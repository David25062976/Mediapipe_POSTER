import torch.utils.data as data
import cv2
import numpy as np
import os
import random

class Affectdataset_8class_2(data.Dataset):
    def __init__(self, root, train=True, transform=None, basic_aug=False, train_dataset_path='train_set', valid_dataset_path='valid_set'):
        self.root = root
        self.train = train
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

        # 選擇 train 或 valid 路徑
        if self.train:
            self.data_dir = os.path.join(self.root, train_dataset_path)
        else:
            self.data_dir = os.path.join(self.root, valid_dataset_path)

        # 掃描資料夾 (假設子資料夾名稱為 0,1,2,3,4,5,6,7)
        self.file_paths = []
        self.target = []

        for label_name in sorted(os.listdir(self.data_dir)):
            label_path = os.path.join(self.data_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            label = int(label_name)  # 資料夾名稱轉成整數 label

            for fname in os.listdir(label_path):
                fpath = os.path.join(label_path, fname)
                if os.path.isfile(fpath):
                    self.file_paths.append(fpath)
                    self.target.append(label)

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return np.array(self.target)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)

        if image is None:
            # 壞圖 → 隨機取另一張替代
            return self.__getitem__((idx + 1) % len(self.file_paths))

        image = image[:, :, ::-1]  # BGR → RGB
        target = self.target[idx]

        # 資料增強
        if self.train and self.basic_aug and random.uniform(0, 1) > 0.5:
            image = random.choice(self.aug_func)(image)

        # transform 前處理
        if self.transform is not None:
            image = self.transform(image)

        return image, target

# 資料增強函式
def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def flip_image(image_array):
    return cv2.flip(image_array, 1)
