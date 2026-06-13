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


# 定義一個輕量級 CNN 來處理 28x28 的 Patch
class PatchFeatureEncoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), # 14x14
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2), # 7x7
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        # )
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

        self.fc = nn.Sequential(
            nn.Linear(256, 512), # 256 -> 512
            nn.ReLU(),
            nn.Linear(512, out_dim) # 512 -> 1024
        )

    def forward(self, x):
        # x shape: [B*49, 3, 28, 28]
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat) # [B*49, 1024]

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
        self.target_indices = [0, 4, 17, 33, 35, 37, 39, 40, 61, 63, 66, 84, 91, 133, 144, 153, 158, 160, 181, 221, 223, 225, 228, 230, 232, 245, 263, 265, 267, 269, 270, 291, 293, 296, 314, 321, 362, 373, 380, 385, 387, 405, 441, 443, 445, 448, 450, 452, 465]
        
        # 對應點映射表 (左, 右)
        self.symmetry_map = {
            35: 265, 228: 448, 230: 450, 232: 452, 245: 465, 221: 441, 223: 443, 
            225: 445, 33: 263, 144: 373, 153: 380, 133: 362, 158: 385, 160: 387, 
            63: 293, 66: 296, 61: 291, 91: 321, 181: 405, 84: 314, 40: 270, 
            39: 269, 37: 267
        }
        # 建立反向映射
        self.reverse_map = {v: k for k, v in self.symmetry_map.items()}
        
        # Patch CNN 卷積特徵提取器
        self.patch_cnn = PatchFeatureEncoder(out_dim=output_dim)

    def get_valid_landmarks(self, results, img_w, img_h):
        """
        改良版邏輯：
        1. 判斷座標是否超出邊界。
        2. 比較對稱點的 Z 軸（深度），若目標點太深則視為被遮蔽，改取對稱點並做鏡像處理。
        """
        coords = np.zeros((478, 3)) # 儲存 x, y, z
        if not results.face_landmarks:
            return np.zeros((len(self.target_indices), 2))
        
        raw_lms = results.face_landmarks[0]
        for i, lm in enumerate(raw_lms):
            coords[i] = [lm.x * img_w, lm.y * img_h, lm.z]
        
        # 定義中心 X 座標（以鼻尖點 4 為準）
        center_x = coords[4][0]
        
        final_coords = []
        # 設定深度閾值，若 Z 軸差距超過此值則視為遮蔽（可根據實驗微調，通常 0.05~0.1）
        z_threshold = 0.05 

        for idx in self.target_indices:
            x, y, z = coords[idx]
            
            # 取得該點的對稱點索引
            pair_idx = self.symmetry_map.get(idx) or self.reverse_map.get(idx)
            
            is_occluded = False
            if pair_idx is not None:
                # 比較深度：若目標點比對稱點深很多，代表轉向了另一邊
                if coords[idx][2] > coords[pair_idx][2] + z_threshold:
                    is_occluded = True

            # 判斷條件：超出邊界 OR 被遮蔽
            if x <= 0 or x >= img_w or y <= 0 or y >= img_h or is_occluded:
                if pair_idx is not None:
                    # 使用對稱點的座標
                    pair_x, pair_y, _ = coords[pair_idx]
                    # 執行鏡像翻轉：對稱點相對於中心點的位置，翻轉到另一邊
                    # 公式：New_X = Center_X - (Pair_X - Center_X)
                    reflected_x = 2 * center_x - pair_x
                    final_coords.append([reflected_x, pair_y])
                else:
                    # 若無對稱點（如 0, 4, 17）且超出邊界，則維持原樣或補零
                    final_coords.append([x, y])
            else:
                # 正常點位
                final_coords.append([x, y])
                
        return np.array(final_coords)

    def extract_patches(self, x, coords):
        """使用 grid_sample 從 GPU Tensor 直接裁切 Patch，效率最高"""
        B, C, H, W = x.shape
        num_pts = coords.shape[1] # 49
        
        # 歸一化座標到 [-1, 1] 供 grid_sample 使用
        norm_coords = torch.zeros((B, num_pts, 2), device=x.device)
        norm_coords[:, :, 0] = (coords[:, :, 0] / (W - 1)) * 2 - 1
        norm_coords[:, :, 1] = (coords[:, :, 1] / (H - 1)) * 2 - 1
        
        # 建立 28x28 的採樣網格
        grid_size = self.patch_size
        rel_grid = torch.linspace(-1, 1, grid_size, device=x.device)
        yy, xx = torch.meshgrid(rel_grid, rel_grid, indexing='ij')
        patch_grid = torch.stack([xx, yy], dim=-1).view(1, 1, grid_size, grid_size, 2) # [1, 1, 28, 28, 2]
        
        # 縮放網格大小以符合 28x28 在原圖中的比例 (假設相對於原圖很小)
        scale = torch.tensor([grid_size/W, grid_size/H], device=x.device).view(1, 1, 1, 1, 2)
        
        # 為每個 Batch 的每個點位計算最終網格
        final_grid = norm_coords.view(B, num_pts, 1, 1, 2) + patch_grid * scale
        final_grid = final_grid.view(B * num_pts, grid_size, grid_size, 2)
        
        # 將輸入 x 擴展以匹配網格數量
        x_expanded = x.unsqueeze(1).expand(-1, num_pts, -1, -1, -1).reshape(B * num_pts, C, H, W)
        
        patches = F.grid_sample(x_expanded, final_grid, align_corners=True)
        return patches # [B*49, 3, 28, 28]

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. MediaPipe 偵測 (需在 CPU/Numpy 環境)
        x_np = (x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
        
        batch_coords = []
        for i in range(B):
            img_cont = np.ascontiguousarray(x_np[i])
            mp_image = mp.Image(mp.ImageFormat.SRGB, img_cont)
            res = self.detector.detect(mp_image)
            coords = self.get_valid_landmarks(res, W, H)
            batch_coords.append(coords)
        
        # 2. 轉回 Tensor 並執行裁切
        batch_coords_tensor = torch.tensor(np.array(batch_coords), dtype=torch.float32, device=x.device)
        patches = self.extract_patches(x, batch_coords_tensor) # [B*49, 3, 28, 28]
        
        # 3. 通過 Patch CNN
        features = self.patch_cnn(patches) # [B*49, 1024]
        
        # 4. 重塑為 [B, 49, 1024]
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
    def __init__(self, img_size=224, num_classes=7, type="large", freeze=False):
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
        self.face_landback = MediaPipeFeatureExtractor(output_dim=1024, patch_size=14)
        
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
