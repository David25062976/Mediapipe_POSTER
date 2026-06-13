import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

from .hyp_crossvit_affect import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


class PatchFeatureEncoder49_14(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), # [3, 14, 14] -> [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # [64, 12, 12] -> [64, 6, 6]
            
            nn.Conv2d(64, 128, kernel_size=3), # [64, 6, 6] -> [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3), # [128, 4, 4] -> [256, 2, 2]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # [256, 2, 2] -> [256, 1, 1]
        )

        self.fuse_projection = nn.Sequential(
            nn.Linear(256, 512), # 256 -> 512
            nn.LayerNorm(512), # 增加穩定性
            nn.ReLU(),
            nn.Linear(512, out_dim) # 512 -> 1024
        )

    def forward(self, x):
        # x shape: [B*49, 3, 28, 28]
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fuse_projection(feat) # [B*49, 1024]

class PatchFeatureEncoder49_24(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # [3, 24, 24] -> [32, 24, 24]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [32, 24, 24] -> [32, 12, 12]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # [32, 12, 12] -> [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # [64, 12, 12] -> [64, 6, 6]
            
            nn.Conv2d(64, 128, kernel_size=3),     # [64, 6, 6] -> [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3),    # [128, 4, 4] -> [256, 2, 2]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),          # [256, 2, 2] -> [256, 1, 1]
        )

        self.fuse_projection = nn.Sequential(
            nn.Linear(256, 512), # 256 -> 512
            nn.LayerNorm(512), # 增加穩定性
            nn.ReLU(),
            nn.Linear(512, out_dim) # 512 -> 1024
        )

    def forward(self, x):
        # x shape: [B*49, 3, 28, 28]
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fuse_projection(feat) # [B*49, 1024]

class PatchFeatureEncoder196_14(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        # 針對 14x14 Patch 的輕量化 CNN
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),      # [3, 14, 14] -> [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # [64, 12, 12] -> [64, 6, 6]
            
            nn.Conv2d(64, 128, kernel_size=3),     # [64, 6, 6] -> [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3),    # [128, 4, 4] -> [256, 2, 2]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),          # [256, 2, 2] -> [256, 1, 1]
        )

        # 聚合層：將 4 個特徵點的向量 (256*4) 合併為 1 個 1024 維特徵
        self.fuse_projection = nn.Sequential(
            nn.Linear(256 * 4, out_dim),           # 1024 -> 1024
            nn.LayerNorm(out_dim),                 # 增加穩定性
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # 輸入 x shape: [B * 196, 3, 14, 14]
        N = x.size(0)
        
        # 1. 卷積特徵提取
        feat = self.conv(x)                        # [B * 196, 256, 1, 1]
        feat = feat.view(N, -1)                    # [B * 196, 256]
        
        # 2. 空間聚合 (196 點 -> 49 點)
        # 將每 4 個點一組進行拼接：[B * 49, 4 * 256]
        feat = feat.view(-1, 4 * 256)              # [B * 49, 1024]
        
        # 3. 投影到最終特徵空間
        return self.fuse_projection(feat)          # [B * 49, 1024]

class PatchFeatureEncoder196_24(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        # 針對 28x28 Patch 的 CNN 架構
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # [3, 24, 24] -> [32, 24, 24]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # [32, 24, 24] -> [32, 12, 12]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # [32, 12, 12] -> [64, 12, 12]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                       # [64, 12, 12] -> [64, 6, 6]
            
            nn.Conv2d(64, 128, kernel_size=3),     # [64, 6, 6] -> [128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3),    # [128, 4, 4] -> [256, 2, 2]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),          # [256, 2, 2] -> [256, 1, 1]
        )

        # 聚合層：將 4 個特徵點的向量 (256*4) 合併為 1 個 1024 維特徵
        self.fuse_projection = nn.Sequential(
            nn.Linear(256 * 4, out_dim),                   # 1024 -> 1024
            nn.LayerNorm(out_dim),                         # 增加穩定性
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # 輸入 x shape: [B * 196, 3, 28, 28]
        N = x.size(0)
        
        # 1. 卷積特徵提取
        feat = self.conv(x)                        # [B * 196, 256, 1, 1]
        feat = feat.view(N, -1)                    # [B * 196, 256]
        
        # 2. 空間聚合 (196 點 -> 49 點)
        feat = feat.view(-1, 4 * 256)              # [B * 49, 1024]
        
        # 3. 投影到最終特徵空間
        return self.fuse_projection(feat)          # [B * 49, 1024]

class MediaPipeFeatureExtractor(nn.Module):
    def __init__(self, output_dim=1024, points=196, patch_size=24):
        super().__init__()
        self.points = points
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        if self.points == 196:
            self.target_indices = [
                # 1-2: 中軸線與口鼻核心 (8 pts)
                0, 4, 6, 8,       # 鼻樑與鼻尖
                17, 18, 19, 152,  # 人中與下巴中心

                # 3-10: 左眼區域 (32 pts)
                33, 133, 144, 145,   153, 154, 155, 157, 
                158, 159, 160, 161,   163, 173, 246, 7,
                37, 39, 40, 43,       193, 195, 196, 198,
                199, 201, 204, 206,   164, 165, 111, 112,

                # 11-18: 右眼區域 (32 pts)
                263, 362, 373, 374,   380, 381, 382, 384,
                385, 386, 387, 388,   390, 398, 466, 249,
                267, 269, 270, 273,   417, 419, 420, 421,
                424, 426, 430, 432,   391, 395, 340, 341,

                # 19-24: 左眉毛區域 (24 pts)
                70, 105, 107, 68,     69, 101, 47, 53,
                65, 59, 61, 51,       21, 22, 24, 25,
                28, 29, 32, 97,       87, 118, 120, 123,

                # 25-30: 右眉毛區域 (24 pts)
                300, 334, 336, 298,   299, 330, 277, 283,
                295, 289, 291, 281,   251, 252, 254, 255,
                258, 259, 262, 326,   317, 347, 349, 352,

                # 31-39: 左臉頰與左嘴角 (36 pts)
                146, 91, 181, 182,    185, 186, 187, 188,
                190, 226, 227, 228,   230, 232, 244, 84,
                82, 170, 171, 177,    215, 217, 220, 221,
                223, 225, 136, 138,   139, 142, 143, 127,
                130, 131, 124, 110,   # 110 是補充點位

                # 40-48: 右臉頰與右嘴角 (36 pts)
                375, 321, 405, 406,   409, 410, 411, 412,
                414, 446, 447, 448,   450, 452, 464, 314,
                312, 395, 396, 401,   435, 437, 440, 441,
                443, 445, 365, 367,   368, 371, 372, 356,
                359, 360, 353, 467,

                # 49: 剩餘邊緣點位聚合 (4 pts)
                467, 247, 464, 252    # 確保最後一組湊滿 4 個
            ]
        else :
            # 49個目標點位
            self.target_indices = [0, 4, 17, 33, 35, 37, 39, 40, 61, 63, 66, 84, 91, 133, 144, 153, 158, 160, 181, 221, 223, 225, 228, 230, 232, 245, 263, 265, 267, 269, 270, 291, 293, 296, 314, 321, 362, 373, 380, 385, 387, 405, 441, 443, 445, 448, 450, 452, 465]

        self.symmetry_map = {
            3: 248, 7: 249, 8: 285, 20: 250, 21: 251, 22: 252, 23: 253, 24: 254,
            25: 255, 26: 256, 27: 257, 28: 258, 29: 259, 30: 260, 31: 261, 32: 262,
            33: 263, 34: 264, 35: 265, 36: 266, 37: 267, 38: 268, 39: 269, 40: 270,
            41: 271, 42: 272, 43: 273, 44: 274, 45: 275, 46: 276, 47: 277, 48: 278,
            49: 279, 50: 280, 51: 281, 52: 282, 53: 283, 54: 284, 55: 285, 56: 286,
            57: 287, 58: 288, 59: 289, 60: 290, 61: 291, 62: 308, 63: 293, 64: 294,
            65: 295, 66: 296, 67: 297, 68: 298, 69: 299, 70: 300, 71: 301, 72: 302,
            73: 303, 74: 304, 75: 305, 76: 306, 77: 307, 78: 308, 79: 309, 80: 318,
            81: 311, 82: 312, 83: 313, 84: 314, 85: 315, 86: 316, 87: 317, 88: 318,
            89: 319, 90: 320, 91: 321, 92: 322, 93: 323, 95: 324, 96: 324, 97: 326,
            98: 327, 99: 328, 100: 329, 101: 330, 102: 331, 103: 332, 104: 333, 105: 334,
            106: 335, 107: 336, 108: 337, 109: 338, 110: 339, 111: 340, 112: 341, 113: 342,
            114: 343, 115: 344, 116: 345, 117: 346, 118: 347, 119: 348, 120: 349, 121: 350,
            122: 351, 123: 352, 124: 353, 125: 354, 126: 355, 127: 356, 128: 357, 129: 358,
            130: 359, 131: 360, 132: 361, 133: 362, 134: 363, 135: 364, 136: 365, 137: 366,
            138: 367, 139: 368, 140: 369, 141: 370, 142: 371, 143: 372, 144: 373, 145: 477,
            146: 375, 147: 376, 148: 377, 149: 378, 150: 379, 153: 380, 154: 381, 155: 382,
            156: 383, 157: 384, 158: 385, 159: 386, 160: 387, 161: 466, 162: 389, 163: 390,
            165: 391, 166: 392, 167: 393, 169: 394, 170: 395, 171: 396, 172: 397, 173: 398,
            174: 399, 176: 400, 177: 401, 178: 402, 179: 403, 180: 404, 181: 405, 182: 406,
            183: 415, 184: 407, 185: 408, 186: 410, 187: 411, 188: 412, 189: 413, 190: 414,
            191: 324, 192: 416, 193: 417, 194: 418, 196: 419, 198: 420, 201: 421, 202: 422,
            203: 423, 204: 424, 205: 425, 206: 426, 207: 427, 208: 428, 209: 429, 210: 430,
            211: 431, 212: 432, 213: 433, 214: 434, 215: 435, 216: 436, 217: 437, 218: 438,
            219: 439, 220: 440, 221: 441, 222: 442, 223: 443, 224: 444, 225: 445, 226: 446,
            227: 447, 228: 448, 229: 449, 230: 450, 231: 451, 232: 452, 233: 453, 234: 454,
            235: 455, 236: 456, 237: 457, 238: 458, 239: 459, 240: 460, 241: 461, 242: 462,
            243: 463, 244: 464, 245: 465, 246: 249, 247: 467, 468: 473, 469: 476, 470: 475,
            471: 474, 472: 477,
        }
        self.reverse_map = {v: k for k, v in self.symmetry_map.items()}

        # 建立模型 Buffer 供 GPU 快速查找
        pair_indices_list = []
        has_pair_list = []
        for idx in self.target_indices:
            pair = self.symmetry_map.get(idx) or self.reverse_map.get(idx)
            if pair is not None:
                pair_indices_list.append(pair)
                has_pair_list.append(True)
            else:
                pair_indices_list.append(idx)
                has_pair_list.append(False)

        self.register_buffer('target_indices_tensor', torch.tensor(self.target_indices, dtype=torch.long))
        self.register_buffer('pair_indices_tensor', torch.tensor(pair_indices_list, dtype=torch.long))
        self.register_buffer('has_pair_tensor', torch.tensor(has_pair_list, dtype=torch.bool))

        # Patch CNN 卷積特徵提取器
        if self.points == 49:
            if self.patch_size == 14:
                self.patch_cnn = PatchFeatureEncoder49_14(out_dim=output_dim)
            else:
                self.patch_cnn = PatchFeatureEncoder49_24(out_dim=output_dim)
        else:
            if self.patch_size == 14:
                self.patch_cnn = PatchFeatureEncoder196_14(out_dim=output_dim)
            else:
                self.patch_cnn = PatchFeatureEncoder196_24(out_dim=output_dim)


    def get_valid_landmarks(self, coords_478, img_w=224, img_h=224):
        """完全使用 Tensor 平行運算計算遮蔽與越界"""
        z_threshold = 0.75
        
        # 提取目標點與對稱點的座標 [B, pts, 3]
        coords_target = coords_478[:, self.target_indices_tensor, :]
        pair_coords = coords_478[:, self.pair_indices_tensor, :]

        # 計算 Z 軸遮蔽
        z_diff = coords_target[:, :, 2] - pair_coords[:, :, 2]
        is_occluded = (z_diff > z_threshold) & self.has_pair_tensor

        # 計算越界
        x = coords_target[:, :, 0]
        y = coords_target[:, :, 1]
        out_of_bounds = (x <= 0) | (x >= img_w) | (y <= 0) | (y >= img_h)

        # 遮罩：是否需要被替換 (並在裁切時翻轉)
        replace_mask = (out_of_bounds | is_occluded) & self.has_pair_tensor  # [B, pts]
        
        final_x = torch.where(replace_mask, pair_coords[:, :, 0], coords_target[:, :, 0])
        final_y = torch.where(replace_mask, pair_coords[:, :, 1], coords_target[:, :, 1])
        final_coords = torch.stack([final_x, final_y], dim=-1) # [B, pts, 2]
        
        return final_coords, replace_mask


    def extract_patches(self, x, final_coords, flip_flags):
        """從 Tensor 裁切 Patch，並使用 GPU 並行翻轉 (展平最佳化版)"""
        B, C, H, W = x.shape
        num_pts = final_coords.shape[1] 
        
        # 1. 歸一化座標到 [-1, 1] 供 grid_sample 使用
        norm_coords = torch.zeros((B, num_pts, 2), device=x.device)
        norm_coords[:, :, 0] = (final_coords[:, :, 0] / (W - 1)) * 2 - 1
        norm_coords[:, :, 1] = (final_coords[:, :, 1] / (H - 1)) * 2 - 1
        
        grid_size = self.patch_size
        rel_grid = torch.linspace(-1, 1, grid_size, device=x.device)
        yy, xx = torch.meshgrid(rel_grid, rel_grid, indexing='ij')
        patch_grid = torch.stack([xx, yy], dim=-1).view(1, 1, grid_size, grid_size, 2)
        
        scale = torch.tensor([grid_size/W, grid_size/H], device=x.device).view(1, 1, 1, 1, 2)
        
        final_grid = norm_coords.view(B, num_pts, 1, 1, 2) + patch_grid * scale
        final_grid = final_grid.view(B * num_pts, grid_size, grid_size, 2)
        
        x_expanded = x.unsqueeze(1).expand(-1, num_pts, -1, -1, -1).reshape(B * num_pts, C, H, W)
        
        # 2. 統一裁切 (此時 patches shape 為 [B*num_pts, C, grid_size, grid_size])
        patches = F.grid_sample(x_expanded, final_grid, align_corners=True)
        
        # 3. 全張量並行翻轉
        flipped_patches = torch.flip(patches, dims=[-1])
        
        # 4. 【關鍵修正】把遮罩直接展平，形狀變為 [B*num_pts, 1, 1, 1] 以完美對齊 patches
        flip_mask = flip_flags.view(-1, 1, 1, 1) 
        
        # 5. 根據遮蔽判定決定是否取用翻轉後的 patch
        patches = torch.where(flip_mask, flipped_patches, patches)
        
        # 直接回傳，不需要再 reshape 回去了！
        return patches

    def forward(self, x, coords_478):
        # x shape: [B, 3, 224, 224]
        # coords shape: [B, 196, 2], flip_flags shape: [B, 196]
        final_coords, flip_flags = self.get_valid_landmarks(coords_478, x.shape[3], x.shape[2])
        
        # 直接拿外部傳進來的座標去裁切！
        patches = self.extract_patches(x, final_coords, flip_flags)
        
        features = self.patch_cnn(patches)
        return features.view(x.shape[0], 49, self.output_dim)


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat
    

# --- 修改後的 pyramid_trans_expr ---
class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large", freeze=False, mediapipe_points=196, mediapipe_patch_size=24):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        # 替換為新的提取器
        self.face_landback = MediaPipeFeatureExtractor(output_dim=1024, points=mediapipe_points, patch_size=mediapipe_patch_size)
        
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        if freeze:
            for param in self.ir_back.parameters():
                param.requires_grad = False
        else:
            for param in self.ir_back.parameters():
                param.requires_grad = True

        # 因為 face_landback 現在直接輸出 [B, 49, 1024]
        # ir_back 輸出也是 [B, 49, 1024]，所以不需要 ir_layer 降維了

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=1024,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=1024)
        self.head = ClassificationHead(input_dim=1024, target_dim=self.num_classes)

    def forward(self, x, coords_478):
        x_face = self.face_landback(x, coords_478) # 往下傳遞
        x_ir = self.ir_back(x)
        
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat
