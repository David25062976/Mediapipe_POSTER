import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit_test1 import *
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
    def __init__(self, img_size=224, num_classes=7, type="large", num_frames=32):
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

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        self.num_frames = num_frames
        self.embed_dim = 512
        self.time_embed = nn.Embedding(self.num_frames, self.embed_dim)

        for param in self.face_landback.parameters():
            param.requires_grad = False

        ###########################################################################333


        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        # ir_checkpoint = ir_checkpoint["model"]
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                                depth=depth, num_heads=8, mlp_ratio=2.,
                                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)


        self.se_block = SE_block(input_dim=512)    # SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)    # ClassificationHead(input_dim=512, target_dim=self.num_classes)

        self.mlp = nn.Sequential(
            nn.Linear(32 * 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # print(f"[pyramid_trans_expr]  Input video shape: {x.shape}")    # torch.Size([2, 32, 3, 224, 224])
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)    # 攤平
        # print(f"[pyramid_trans_expr]  Input video after flatten: {x.shape}")    # torch.Size([64, 3, 224, 224])

        x_face = F.interpolate(x, size=112)
        # print(f"[pyramid_trans_expr]  Resized face image shape: {x_face.shape}")    # torch.Size([64, 3, 112, 112])
        _, x_face = self.face_landback(x_face)
        # print(f"[pyramid_trans_expr]  Face landmark features shape: {x_face.shape}")    # torch.Size([64, 512, 7, 7])
        x_face = x_face.view(B * T, -1, 49).transpose(1,2)
        # print(f"[pyramid_trans_expr]  Face landmark features reshaped: {x_face.shape}")    # torch.Size([64, 49, 512])
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        # print(f"[pyramid_trans_expr]  IR50 extracted features shape: {x_ir.shape}")    # torch.Size([64, 49, 1024])
        x_ir = self.ir_layer(x_ir)
        # print(f"[pyramid_trans_expr]  IR50 reduced feature shape: {x_ir.shape}")    # torch.Size([64, 49, 512])
        ###############  image x_ir ([B, 49, 512])

        y_hat = self.pyramid_fuse(x_ir, x_face)
        # y1_hat, y2_hat = self.pyramid_fuse(x_ir, x_face)
        # y_hat = torch.cat((y1_hat, y2_hat), dim=1)
        # print(f"[pyramid_trans_expr]  After pyramid transformer shape: {y_hat.shape}")    # torch.Size([64, 1024])

        # 將每 T 幀 reshape 回 [B, T, ...]
        y_hat = y_hat.view(B, T, -1)
        y_hat = y_hat.view(B, -1)

        out = self.mlp(y_hat)         # [b, num_classes]

        # # 對 T 維做 temporal pooling（平均）
        # y_hat = torch.mean(y_hat, dim=1)  # [B, feature_dim]

        # y_hat = self.se_block(y_hat)
        # # print(f"[pyramid_trans_expr]  After SE Block shape: {y_hat.shape}")    # torch.Size([2, 1024])
        # out = self.head(y_hat)
        # # print(f"[pyramid_trans_expr]  Final output shape: {out.shape}")    # torch.Size([2, 6])

        return out, y_hat


