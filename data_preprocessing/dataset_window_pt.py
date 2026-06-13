import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from facenet_pytorch import MTCNN  # 請確保已安裝: pip install facenet-pytorch
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

class AffectDatasetWindow_pt(data.Dataset):
    def __init__(self, img_root, anno, landmarks_db, window_size=8, stride=8, transform=None):
        """
        Args:
            img_root (str): 圖片所在的根目錄
            anno (str): 標註檔路徑
            landmarks_db (str): generate_landmarks.py 預先生成的 .pt 檔路徑
            window_size (int): 每個樣本包含的連續幀數 (T)
            stride (int): 滑動視窗的步長
            transform: PyTorch transforms
        """
        self.img_root = img_root
        self.anno = anno
        
        self.landmarks_db = torch.load(landmarks_db, weights_only=False)

        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        print(f"Building {anno} dataloader with Window Size {window_size}...")

        # 讀取影片幀檔案清單
        df = pd.read_csv(os.path.join(self.img_root, self.anno), sep=' ', header=None, names=['path', 'label'])
        self.video_paths = df['path'].values
        self.targets = df['label'].values
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for video_rel_path, label in zip(self.video_paths, self.targets):
            video_dir = os.path.join(self.img_root, video_rel_path)
            if not os.path.exists(video_dir): continue
            
            frames = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            if len(frames) < self.window_size: continue
                
            for i in range(0, len(frames) - self.window_size + 1, self.stride):
                window_frames = frames[i : i + self.window_size]
                samples.append((window_frames, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        """
        回傳所有視窗樣本的標籤列表
        """
        return [label for _, label in self.samples]

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        imgs = []
        window_coords = []

        # --- 1. 防呆機制：先掃描視窗內找出「第一個有效的特徵點」 ---
        # 如果影片開頭第一幀剛好沒抓到臉，我們必須有個有效的座標作為基準
        fallback_coords = np.zeros((478, 3), dtype=np.float32)
        for path in frame_paths:
            # 將絕對路徑轉為相對路徑，以對應 landmarks_db 內的 key (例如: '0/image001.jpg')
            rel_path = os.path.relpath(path, self.img_root).replace('\\', '/')
            if rel_path in self.landmarks_db and self.landmarks_db[rel_path]['face_found']:
                fallback_coords = self.landmarks_db[rel_path]['coords'].copy()
                break

        # last_valid_coords 負責跨幀記憶
        last_valid_coords = fallback_coords

        for path in frame_paths:
            image = cv2.imread(path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            rel_path = os.path.relpath(path, self.img_root).replace('\\', '/')

            # --- 3. 提取 478 點座標 ---
            if rel_path in self.landmarks_db and self.landmarks_db[rel_path]['face_found']:
                coords = self.landmarks_db[rel_path]['coords'].copy()
                last_valid_coords = coords # 更新記憶
            else:
                # 該幀遺失人臉，沿用上一幀的座標 (Forward-fill)
                coords = last_valid_coords.copy()
            
            window_coords.append(torch.tensor(coords))

            # --- 4. 影像處理 (強制 Resize 以對齊離線特徵點) ---
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform is not None:
                image = self.transform(image)
            
            imgs.append(image)

        # --- 5. 堆疊成時序 Tensor ---
        # 影像 Tensor: [T, C, H, W] -> 轉換為 3D CNN 常用的 [C, T, H, W]
        data_tensor = torch.stack(imgs, dim=0)    # (T, C, H, W)
        data_tensor = data_tensor.permute(1, 0, 2, 3)    # (C, T, H, W)
        
        # 座標 Tensor: [T, 478, 3]
        coords_tensor = torch.stack(window_coords, dim=0) 

        # 回傳時序影像, 標籤, 與時序座標
        return data_tensor, label, coords_tensor, frame_paths
