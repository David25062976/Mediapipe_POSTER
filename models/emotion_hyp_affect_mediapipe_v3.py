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


class PatchFeatureEncoder14(nn.Module):
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

class PatchFeatureEncoder28(nn.Module):
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
    def __init__(self, output_dim=1024, patch_size=14):
        super().__init__()
        # 初始化 MediaPipe Task
        base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task', 
                                          delegate=python.BaseOptions.Delegate.GPU)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # 49個目標點位
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
        # 對應點映射表 (左, 右)
        self.symmetry_map = {
            33: 263, 133: 362, 144: 373, 145: 374, 153: 380, 154: 381, 155: 382, 157: 384, 158: 385, 159: 386, 160: 387, 161: 388, 163: 390, 173: 398, 246: 466, 70: 300, 105: 334, 107: 336, 68: 298, 69: 299, 101: 330, 47: 277, 53: 283, 65: 295, 61: 291, 146: 375, 91: 321, 181: 405, 84: 314, 82: 312, 37: 267, 39: 269, 40: 270, 127: 356, 232: 452, 136: 365, 171: 396, 210: 430, 215: 435, 138: 367, 212: 432, 7: 240, 21: 251, 22: 252, 24: 254, 25: 255, 28: 258, 29: 259, 32: 262, 43: 273, 48: 278, 51: 281, 59: 289, 87: 317, 97: 326, 111: 340, 112: 341, 118: 347, 120: 349, 123: 352, 124: 353, 130: 359, 131: 360, 139: 368, 142: 371, 143: 372, 165: 391, 170: 395, 177: 401, 182: 406, 185: 409, 186: 410, 187: 411, 188: 412, 190: 414, 193: 417, 196: 419, 198: 420, 201: 421, 204: 424, 206: 426, 217: 437, 220: 440, 221: 441, 223: 443, 225: 445, 226: 446, 227: 447, 228: 448, 230: 450, 244: 464, 247: 467
        }
        # 建立反向映射
        self.reverse_map = {v: k for k, v in self.symmetry_map.items()}
        
        # Patch CNN 卷積特徵提取器
        if self.patch_size == 14:
            self.patch_cnn = PatchFeatureEncoder14(out_dim=output_dim)
        else:
            self.patch_cnn = PatchFeatureEncoder28(out_dim=output_dim)

    def get_valid_landmarks(self, results, img_w, img_h):
        """
        回傳:
        - final_coords: 196 個點的實際裁切座標 (若遮蔽，此座標為對稱點的座標)
        - flip_flags: 196 個 boolean，標記該 Patch 切下來後是否需要水平翻轉
        """
        coords = np.zeros((478, 3))
        if not results.face_landmarks:
            return np.zeros((len(self.target_indices), 2)), [False] * len(self.target_indices)
        
        raw_lms = results.face_landmarks[0]
        for i, lm in enumerate(raw_lms):
            coords[i] = [lm.x * img_w, lm.y * img_h, lm.z]
        
        final_coords = []
        flip_flags = []
        z_threshold = 0.75

        for idx in self.target_indices:
            x, y, z = coords[idx]
            pair_idx = self.symmetry_map.get(idx) or self.reverse_map.get(idx)
            
            is_occluded = False
            # 判斷遮蔽或超出邊界
            if pair_idx is not None:
                if coords[idx][2] > coords[pair_idx][2] + z_threshold:
                    is_occluded = True
            
            if x <= 0 or x >= img_w or y <= 0 or y >= img_h or is_occluded:
                if pair_idx is not None:
                    # 【修改點】：直接記錄對稱點的座標，並標記需要翻轉
                    pair_x, pair_y, _ = coords[pair_idx]
                    final_coords.append([pair_x, pair_y])
                    flip_flags.append(True)
                else:
                    final_coords.append([x, y])
                    flip_flags.append(False)
            else:
                final_coords.append([x, y])
                flip_flags.append(False)
                
        return np.array(final_coords), flip_flags

    def extract_patches(self, x, coords, flip_flags):
        """從 Tensor 裁切 Patch，並對被標記的 Patch 進行水平翻轉"""
        B, C, H, W = x.shape
        num_pts = coords.shape[1] # 196
        
        norm_coords = torch.zeros((B, num_pts, 2), device=x.device)
        norm_coords[:, :, 0] = (coords[:, :, 0] / (W - 1)) * 2 - 1
        norm_coords[:, :, 1] = (coords[:, :, 1] / (H - 1)) * 2 - 1
        
        grid_size = self.patch_size
        rel_grid = torch.linspace(-1, 1, grid_size, device=x.device)
        yy, xx = torch.meshgrid(rel_grid, rel_grid, indexing='ij')
        patch_grid = torch.stack([xx, yy], dim=-1).view(1, 1, grid_size, grid_size, 2)
        
        scale = torch.tensor([grid_size/W, grid_size/H], device=x.device).view(1, 1, 1, 1, 2)
        
        final_grid = norm_coords.view(B, num_pts, 1, 1, 2) + patch_grid * scale
        final_grid = final_grid.view(B * num_pts, grid_size, grid_size, 2)
        
        x_expanded = x.unsqueeze(1).expand(-1, num_pts, -1, -1, -1).reshape(B * num_pts, C, H, W)
        
        # 1. 統一裁切 (B*196 個 Patch)
        patches = F.grid_sample(x_expanded, final_grid, align_corners=True) # [B*196, 3, 14, 14]
        
        # 2. 針對需要翻轉的 Patch 執行 flip
        # 將 patches 重塑回 [B, 196, 3, 14, 14] 方便依據 flag 處理
        patches = patches.view(B, num_pts, C, grid_size, grid_size)
        
        for b in range(B):
            for p_idx, needs_flip in enumerate(flip_flags[b]):
                if needs_flip:
                    # 對第 4 維 (寬度 W) 進行翻轉
                    patches[b, p_idx] = torch.flip(patches[b, p_idx], dims=[-1])
                    
        return patches.view(B * num_pts, C, grid_size, grid_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x_np = (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
        
        batch_coords = []
        batch_flip_flags = []
        
        for i in range(B):
            img_cont = np.ascontiguousarray(x_np[i])
            mp_image = mp.Image(mp.ImageFormat.SRGB, img_cont)
            res = self.detector.detect(mp_image)
            
            # 取得座標與翻轉標記
            coords, flags = self.get_valid_landmarks(res, W, H)
            batch_coords.append(coords)
            batch_flip_flags.append(flags)
        
        batch_coords_tensor = torch.tensor(np.array(batch_coords), dtype=torch.float32, device=x.device)
        
        # 傳入 flip_flags 給 extract_patches
        patches = self.extract_patches(x, batch_coords_tensor, batch_flip_flags)
        
        features = self.patch_cnn(patches)
        return features.view(B, 49, self.output_dim)


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
    def __init__(self, img_size=224, num_classes=7, type="large", freeze=False, mediapipe_patch_size=14):
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
        self.face_landback = MediaPipeFeatureExtractor(output_dim=1024, patch_size=mediapipe_patch_size)
        
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

    def forward(self, x):
        x_face = self.face_landback(x) # [B, 49, 1024]
        x_ir = self.ir_back(x)         # [B, 49, 1024]
    
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat
