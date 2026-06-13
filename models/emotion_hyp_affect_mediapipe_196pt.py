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
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Patch CNN 卷積特徵提取器
        if self.patch_size == 14:
            self.patch_cnn = PatchFeatureEncoder14(out_dim=output_dim)
        else:
            self.patch_cnn = PatchFeatureEncoder28(out_dim=output_dim)

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

    def forward(self, x, coords, flip_flags):
        # x shape: [B, 3, 224, 224]
        # coords shape: [B, 196, 2], flip_flags shape: [B, 196]
        
        # 直接拿外部傳進來的座標去裁切！
        patches = self.extract_patches(x, coords, flip_flags)
        
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

    def forward(self, x, coords, flip_flags):
        x_face = self.face_landback(x, coords, flip_flags) # 往下傳遞
        x_ir = self.ir_back(x)
        
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat
