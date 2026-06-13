import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import random

class Affectdataset_8class_2(data.Dataset):
    def __init__(self, data_dir, landmarks_db, train=True, transform=None, basic_aug=False, point=196):
        self.transform = transform
        self.basic_aug = basic_aug
        self.train = train

        self.data_dir = data_dir
        self.landmarks_db = torch.load(landmarks_db, weights_only=False)

        self.file_paths, self.target, self.rel_paths = [], [], []

        for label_name in sorted(os.listdir(self.data_dir)):
            label_path = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(label_path): continue
            label = int(label_name)

            for fname in os.listdir(label_path):
                fpath = os.path.join(label_path, fname)
                if os.path.isfile(fpath):
                    self.file_paths.append(fpath)
                    self.target.append(label)
                    self.rel_paths.append(f"{label_name}/{fname}")
        

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return np.array(self.target)

    def __getitem__(self, idx):
        # 安全的尋找迴圈
        for attempt in range(len(self.file_paths)):
            curr_idx = (idx + attempt) % len(self.file_paths)
            
            path = self.file_paths[curr_idx]
            rel_path = self.rel_paths[curr_idx]
            
            image = cv2.imread(path)
            
            if image is None:
                continue
                
            if rel_path not in self.landmarks_db:
                continue
                
            # 檢查：如果在離線生成時沒偵測到臉，就跳過這張
            lm_data = self.landmarks_db[rel_path]
            if not lm_data['face_found']:
                continue
                
            break
        else:
            raise RuntimeError("整個資料集都沒有有效的圖或特徵點！")

        image = cv2.resize(image, (224, 224))
        image = image[:, :, ::-1] 
        target = self.target[curr_idx]

        # 直接取出 478 個原始座標 [478, 3]
        coords_478 = lm_data['coords'].copy()

        # 資料增強 (移除水平翻轉以保護 478 點位對齊)
        if self.train and self.basic_aug:
            if random.uniform(0, 1) > 0.5:
                std = 30**0.5
                noisy = image + np.random.normal(0, std, image.shape)
                image = np.clip(noisy, 0, 255).astype(np.uint8)

        if self.transform is not None:
            image = self.transform(image)

        # 回傳: 影像, 標籤, 478個點座標
        return image, target, torch.tensor(coords_478)

# 資料增強函式
def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def flip_image(image_array):
    return cv2.flip(image_array, 1)
