import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone



def load_IR_pretrained_weights(model, checkpoint):
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

def load_PF_pretrained_weights(model, checkpoint):
    import collections
    
    # 1. 取得原始權重字典
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    
    matched_layers = []
    
    # 定義你要找的子模組前綴
    prefix = 'pyramid_fuse.' 

    for k, v in state_dict.items():
        # 先處理 DataParallel 可能產生的 module. 前綴
        name = k.replace('module.', '')
        
        # 關鍵邏輯：如果這個 key 是屬於 pyramid_fuse 的
        if name.startswith(prefix):
            # 去掉 'pyramid_fuse.' 這 13 個字元，讓它變成 'cls_token', 'pos_embed' 等
            name = name[len(prefix):] 
        
        # 進行匹配
        if name in model_dict:
            if model_dict[name].size() == v.size():
                new_state_dict[name] = v
                matched_layers.append(name)
            else:
                print(f"[Size Mismatch] {name}: ckpt {v.size()} vs model {model_dict[name].size()}")

    # 載入並更新
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    print(f'Successfully loaded {len(matched_layers)} layers into pyramid_fuse.')
    return model


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


class pyramid_trans_expr_window(nn.Module):
    def __init__(self, img_size=224, num_classes=7, window_size=8, type="large", freeze_list=[False, False, False], temporal_type="MLP"):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8    # 10

        self.img_size = img_size
        self.num_classes = num_classes
        self.window_size = window_size
        self.temporal_type = temporal_type

        FL_freeze, IR_freeze, PF_freeze = freeze_list

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        if FL_freeze:
            for param in self.face_landback.parameters():
                param.requires_grad = False
            print("face_landback is frozen")

        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_IR_pretrained_weights(self.ir_back, ir_checkpoint)

        if IR_freeze :
            for param in self.ir_back.parameters():
                param.requires_grad = False
            print("ir_back is frozen")

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)
        
        if PF_freeze is not None :
            # 載入權重到特定的模組 (pyramid_fuse)
            pf_checkpoint = torch.load(PF_freeze, map_location=lambda storage, loc: storage)
            self.pyramid_fuse = load_PF_pretrained_weights(self.pyramid_fuse, pf_checkpoint)

            for param in self.pyramid_fuse.parameters():
                param.requires_grad = False
            print(f"{PF_freeze} has been used as a pre-trained frozen model")


        if temporal_type == "MLP":
            self.temporal = nn.Sequential(
                nn.Linear(self.window_size * 512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.5), 
                nn.Linear(1024, 512), 
                nn.ReLU(),
                nn.Dropout(p=0.5), 
                nn.Linear(512, 512),
            )
            linear_input_dim = 512
        
        elif temporal_type == "CNN":
            # 輸入 Channel: 512, 時間長度: W
            self.temporal = nn.Sequential(
                # Layer 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # W變 W/2, 512變 256
                
                # Layer 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)  # W變 W/4, 256變 128
            )
            # [自動計算 Head 的輸入維度]
            # 經過兩次 Pooling (除以4)
            final_h = self.window_size // 4
            final_w = 512 // 4
            final_c = 64
            
            # 攤平後的向量長度 (例如: 64 * 2 * 128 = 16384)
            linear_input_dim = final_c * final_h * final_w
        
        elif temporal_type == "Transformer":
            # [新增] Transformer 設定
            # d_model 必須等於你的特徵維度 (512)
            # nhead 建議設為 8 (512/8=64)
            encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=3)
            
            # 位置編碼 (Positional Embedding)
            # 設一個夠大的 max_len (例如 100)，forward 時再切
            self.pos_embedding = nn.Parameter(torch.randn(1, 100, 512))
            
            # Transformer 輸出後通常接 Global Average Pooling，所以維度保持 512
            linear_input_dim = 512


        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=linear_input_dim, target_dim=self.num_classes)    # MLP=1024, CNN=128

    def forward(self, x):
        # 修正點：假設 x 原始形狀為 [B, C, W, H, Wd] (例如 [B, 3, 8, 224, 224])
        # 我們將其轉置為 [B, W, C, H, Wd] (即 [B, 8, 3, 224, 224])
        x = x.permute(0, 2, 1, 3, 4)
        
        # x shape 現在是: [B, W, 3, H, W] (例如 [B, 8, 3, 224, 224])
        B, W, C, H, Wd = x.shape

        y_seq = []

        # 逐幀送入你的原模型 (但不經 classification head)
        for t in range(W):
            x_t = x[:, t]              # shape: [B, 3, 224, 224]

            # === Feature Extractor ===
            x_face = F.interpolate(x_t, size=112)
            _, x_face = self.face_landback(x_face)
            x_face = x_face.view(B, -1, 49).transpose(1, 2)

            x_ir = self.ir_back(x_t)
            x_ir = self.ir_layer(x_ir)

            y_hat = self.pyramid_fuse(x_ir, x_face)
            y_hat = self.se_block(y_hat)
            # y_hat: [B, 512]

            y_seq.append(y_hat)

        y_seq = torch.stack(y_seq, dim=1)

        # ==================== Temporal Fusion Selection ====================
        if self.temporal_type == "MLP":
            # Flatten 全全部
            y_seq = y_seq.view(B, self.window_size * 512)    # [B, W * 512]
            y_fused = self.temporal(y_seq)
            out = self.head(y_fused)

        elif self.temporal_type == "CNN":
            # CNN 需要 4D 輸入 [B, C, H, W] -> 這裡變成 [B, 1, W, 512]
            cnn_in = y_seq.unsqueeze(1)
            features = self.temporal(cnn_in)
            
            # Flatten (攤平)
            y_fused = features.view(B, -1) 
            out = self.head(y_fused)

        elif self.temporal_type == "Transformer":
            # 1. 加入位置編碼 (Positional Encoding)
            # y_seq: [B, W, 512]
            # self.pos_embedding: [1, 100, 512] -> 切成 [1, W, 512]
            # PyTorch 的 broadcasting 會自動處理 Batch 維度
            x_trans = y_seq + self.pos_embedding[:, :W, :]

            # 2. Transformer Encoder
            # output: [B, W, 512] (保持序列形狀)
            trans_out = self.temporal(x_trans)

            # 3. 聚合特徵 (Aggregation)
            # 雖然 CNN 你選擇攤平，但在 Transformer 中，通常使用 "Mean Pooling" 效果最好
            # 這樣 Head 的輸入大小 (512) 就不會被 Window Size 綁死
            y_fused = trans_out.mean(dim=1)  # [B, 512]

            # 4. 最終分類
            out = self.head(y_fused)

        return out, y_fused
