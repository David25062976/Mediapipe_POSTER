import torch
import torch.nn as nn
from models.emotion_hyp_test1 import pyramid_trans_expr

class MultiPosterEmotionModel_Independent(nn.Module):
    def __init__(self, poster_factory_fn, num_classes=6, num_posters=9, feature_dim=512):
        super().__init__()
        self.num_posters = num_posters
        self.feature_dim = feature_dim

        # 建立 9 個獨立的 POSTER 模型
        self.posters = nn.ModuleList([poster_factory_fn() for _ in range(num_posters)])

        self.mlp = nn.Sequential(
            nn.Linear(num_posters * feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, num_posters, C, H, W]
        b, n, c, h, w = x.shape
        assert n == self.num_posters, f"Expected {self.num_posters} images, got {n}"

        feats = []
        for i in range(self.num_posters):
            xi = x[:, i]  # [batch_size, C, H, W]
            _, fi = self.posters[i](xi)  # 只拿 feature 向量 [batch_size, 512]
            feats.append(fi)

        feats = torch.cat(feats, dim=1)  # [batch_size, num_posters * 512]
        out = self.mlp(feats)            # [batch_size, num_classes]
        return out
