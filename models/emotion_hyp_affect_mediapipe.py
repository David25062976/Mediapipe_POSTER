import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


from .hyp_crossvit_affect import *
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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


class MediaPipeFeatureExtractor(nn.Module):
    def __init__(self, output_dim=1024, num_patches=49):
        super().__init__()
        
        # 設定 MediaPipe BaseOptions 使用 GPU
        base_options = python.BaseOptions(
            model_asset_path='models/face_landmarker.task', # 需要先下載官方的 .task 檔
            delegate=python.BaseOptions.Delegate.GPU # 強制指定使用 GPU
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        
        # 初始化 Landmarker
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.input_dim = 478 * 3 # MediaPipe FaceMesh 點位座標
        self.num_patches = num_patches
        self.output_dim = output_dim
        
        # 投影層：將座標映射到你要求的 [49, 1024]
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.LayerNorm(2048), # 加入 LayerNorm 增加訓練穩定性
            nn.ReLU(),
            nn.Linear(2048, num_patches * output_dim)
        )

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # 1. 將 PyTorch Tensor 轉換為 MediaPipe 影像格式
        # 注意：MediaPipe 需要 uint8 的 RGB 影像
        x_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        
        all_landmarks = []
        for i in range(batch_size):
            # 轉換為 MediaPipe Image 對象
            img = (x_np[i] * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            mp_image = mp.Image(mp.ImageFormat.SRGB, img)
            
            # 執行推理 (這部分會在 GPU 上運行)
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                # 提取 478 個關鍵點座標
                landmarks = []
                for lm in detection_result.face_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])
                all_landmarks.append(landmarks)
            else:
                all_landmarks.append([0.0] * self.input_dim)
        
        # 2. 轉換回 Tensor 並進行投影
        landmarks_tensor = torch.tensor(all_landmarks, dtype=torch.float32).to(device)
        features = self.projection(landmarks_tensor) # [B, 49 * 1024]
        
        # 3. 調整維度為 [B, 49, 1024]
        return features.view(batch_size, self.num_patches, self.output_dim)


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

        # self.face_landback = MobileFaceNet([112, 112],136)
        # face_landback_checkpoint = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage)
        # self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        self.face_landback = MediaPipeFeatureExtractor(output_dim=1024, num_patches=49)

        if freeze:
            for param in self.face_landback.parameters():
                param.requires_grad = False
        else:
            for param in self.face_landback.parameters():
                param.requires_grad = True



        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('./models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        if freeze:
            for param in self.ir_back.parameters():
                param.requires_grad = False
        else:
            for param in self.ir_back.parameters():
                param.requires_grad = True

        # self.ir_layer = nn.Linear(1024,512)


        #############################################################3
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=1024,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=1024)
        self.head = ClassificationHead(input_dim=1024, target_dim=self.num_classes)


    def forward(self, x):

        x_face = self.face_landback(x)
        # B_ = x.shape[0]
        # x_face = F.interpolate(x, size=112)
        # _, x_face = self.face_landback(x_face)
        # x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        # x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat


