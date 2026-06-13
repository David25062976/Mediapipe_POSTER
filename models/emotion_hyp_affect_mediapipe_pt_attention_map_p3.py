import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

# ★ 關鍵修改：引進支援完全動態序列長度的 P3 版本 Transformer
from .hyp_crossvit_affect_attention_map_p3 import *
from .ir50 import Backbone

def load_pretrained_weights(model, checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    
    matched_layers = []
    discarded_layers = []
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            
        if k in model_dict:
            if model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(f"{k} (Shape 衝突: Checkpoint {list(v.size())} vs 模型 {list(model_dict[k].size())})")
        else:
            discarded_layers.append(f"{k} (目前模型中不存在此層)")
            
    missing_layers = [k for k in model_dict.keys() if k not in matched_layers]
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print("\n" + "=" * 60)
    print("                 載入權重診斷報告 (Weight Loading Report)")
    print("=" * 60)
    print(f"✅ 成功載入層數 (Matched Layers): {len(matched_layers)} layers")
    print(f"❌ 丟棄/無法載入的 Checkpoint 權重: {len(discarded_layers)} layers")
    for discard_layer in discarded_layers: print(discard_layer)
    print(f"⚠️ 模型中未獲得預訓練權重 (將隨機初始化): {len(missing_layers)} layers")
    for missing_layer in missing_layers: print(missing_layer)
    print("=" * 60 + "\n")
    
    return model

class PatchFeatureEncoder478_14(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fuse_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fuse_projection(feat)

class MediaPipeFeatureExtractor(nn.Module):
    def __init__(self, output_dim=1024, points=478, patch_size=14):
        super().__init__()
        self.points = points
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        self.target_indices = list(range(478))
        
        # ★ 完整封裝由 med.csv 修正後的左右臉極精密對稱表
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

        self.patch_cnn = PatchFeatureEncoder478_14(out_dim=output_dim)

    def get_valid_landmarks(self, coords_478, img_w=224, img_h=224, active_indices=None):
        # ★ 階梯式推論關鍵：動態根據 active_indices 篩選點位
        if active_indices is not None:
            t_idx = self.target_indices_tensor[active_indices]
            p_idx = self.pair_indices_tensor[active_indices]
            h_pair = self.has_pair_tensor[active_indices]
        else:
            t_idx = self.target_indices_tensor
            p_idx = self.pair_indices_tensor
            h_pair = self.has_pair_tensor

        coords_target = coords_478[:, t_idx, :]
        pair_coords = coords_478[:, p_idx, :]

        z_diff = coords_target[:, :, 2] - pair_coords[:, :, 2]
        is_occluded = (z_diff > 0.75) & h_pair

        x = coords_target[:, :, 0]
        y = coords_target[:, :, 1]
        out_of_bounds = (x <= 0) | (x >= img_w) | (y <= 0) | (y >= img_h)

        replace_mask = (out_of_bounds | is_occluded) & h_pair
        
        final_x = torch.where(replace_mask, pair_coords[:, :, 0], coords_target[:, :, 0])
        final_y = torch.where(replace_mask, pair_coords[:, :, 1], coords_target[:, :, 1])
        final_coords = torch.stack([final_x, final_y], dim=-1)
        
        return final_coords, replace_mask

    def extract_patches(self, x, final_coords, flip_flags):
        B, C, H, W = x.shape
        N = final_coords.shape[1]
        p = self.patch_size
        half_p = p // 2

        x_padded = F.pad(x, (half_p, half_p, half_p, half_p), mode='constant', value=0)
        cx = torch.clamp(final_coords[:, :, 0] + half_p, half_p, W + half_p - 1)
        cy = torch.clamp(final_coords[:, :, 1] + half_p, half_p, H + half_p - 1)

        patches = []
        for b in range(B):
            img = x_padded[b]
            for n in range(N):
                ix = int(cx[b, n].item())
                iy = int(cy[b, n].item())
                patch = img[:, iy - half_p : iy + half_p, ix - half_p : ix + half_p]
                if flip_flags[b, n]:
                    patch = torch.flip(patch, dims=[2])
                patches.append(patch)

        return torch.stack(patches, dim=0)

    def forward(self, x, coords_478, active_indices=None):
        final_coords, flip_flags = self.get_valid_landmarks(coords_478, x.shape[3], x.shape[2], active_indices)
        patches = self.extract_patches(x, final_coords, flip_flags)
        features = self.patch_cnn(patches)
        
        # ★ 關鍵動態還原：不再寫死 49 或 478，而是根據當前保留的點數 (final_coords.shape[1]) 動態還原維度
        num_points = final_coords.shape[1]
        return features.view(x.shape[0], num_points, self.output_dim)

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

class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=8, type="large", freeze=False, mediapipe_points=478, mediapipe_patch_size=14):
        super().__init__()
        depth = 8
        if type == "small": depth = 4
        if type == "base": depth = 6
        if type == "large": depth = 8

        self.img_size = img_size
        self.num_classes = num_classes

        self.face_landback = MediaPipeFeatureExtractor(output_dim=1024, points=mediapipe_points, patch_size=mediapipe_patch_size)
        
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        if freeze:
            for param in self.ir_back.parameters(): param.requires_grad = False
        else:
            for param in self.ir_back.parameters(): param.requires_grad = True

        self.pyramid_fuse = HyVisionTransformer(in_chans=mediapipe_points, q_chanel=mediapipe_points, embed_dim=1024,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=1024)
        self.head = ClassificationHead(input_dim=1024, target_dim=self.num_classes)

    def forward(self, x, coords_478, return_attention=False, active_indices=None):
        # 1. 取得臉部動態特徵 [B, current_pts, 1024]
        x_face = self.face_landback(x, coords_478, active_indices=active_indices)
        
        # 2. 取得基礎背景特徵 [B, 49, 1024]
        x_ir = self.ir_back(x)

        # 3. ★ 動態插值升維：根據現有臉部點數 N 將背景特徵線性重採樣對齊
        current_pts = x_face.size(1)
        x_ir = x_ir.transpose(1, 2)
        x_ir = F.interpolate(x_ir, size=current_pts, mode='linear', align_corners=False)
        x_ir = x_ir.transpose(1, 2)

        if return_attention:
            y_hat, attn_maps = self.pyramid_fuse(x_ir, x_face, return_attention=True)
        else:
            y_hat = self.pyramid_fuse(x_ir, x_face)

        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        if return_attention:
            return out, y_feat, attn_maps
        return out, y_feat