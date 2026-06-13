import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_preprocessing.dataset_vivit import VivitDataSet
from torchvision import transforms
import random
import os

# 載入 VivitDataSet class 和必要的 augmentation function
# 假設 VivitDataSet 所在檔案叫做 vivit_dataset.py
# from vivit_dataset import VivitDataSet

# 如果你把 class 放在同一個檔案就不用 import

# 設定你的資料根目錄
DATASET_ROOT = './data/vivit_9f/'  # 修改成你資料集的根目錄

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.1)),
])

# 建立資料集實例
dataset = VivitDataSet(root=DATASET_ROOT, train=True, transform=data_transforms, basic_aug=True)

# 使用 DataLoader 包裝，這邊只設 batch_size=1 為了簡單測試
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



# 隨機取得一個 batch
for frames, label in dataloader:
    # frames 是 list of tensors (影片幀)，label 是 tensor
    print(f"Label: {label}")
    print(f"Number of frames: {frames.shape}")

    # # 顯示前 5 幅畫面
    # for i, frame in enumerate(frames[0][:5]):
    #     plt.subplot(1, 5, i+1)
    #     plt.imshow(frame.numpy())
    #     plt.axis('off')
    # plt.suptitle(f"Label: {label.item()}")
    # plt.show()
    # break  # 只顯示一筆
