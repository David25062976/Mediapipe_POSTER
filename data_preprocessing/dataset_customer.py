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

class AffectDatasetWindow(data.Dataset):
    def __init__(self, img_root, train, window_size=8, stride=4, transform=None):
        """
        Args:
            list_file (str): 剛剛生成的 txt 檔路徑 (例如 'train_list.txt')
            img_root (str): 圖片所在的根目錄 (例如 'D:/消費者行為分析原始資料')，用來與 txt 裡的相對路徑組合成絕對路徑
            window_size (int): 每個樣本包含的連續幀數 (T)
            stride (int): 滑動視窗的步長
            transform: PyTorch transforms (應用於每一幀)
        """
        self.img_root = img_root
        self.train = train
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # 1. 讀取 txt 檔案 (路徑 標籤)
        # 假設格式: dislike1-100/folder_name 0
        
        if self.train:
            df = pd.read_csv(os.path.join(self.img_root, "train_list.txt"), sep=' ', header=None, names=['path', 'label'])
        else:
            df = pd.read_csv(os.path.join(self.img_root, "test_list.txt"), sep=' ', header=None, names=['path', 'label'])
            
        self.video_paths = df['path'].values
        self.targets = df['label'].values
        
        # 2. 預處理：將所有影片資料夾展開成多個「時序樣本」
        # samples 裡存的是: ([frame1_path, frame2_path, ...], label)
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        print(f"正在建立時序樣本 (Window={self.window_size}, Stride={self.stride})...")
        
        for video_rel_path, label in zip(self.video_paths, self.targets):
            # 組合完整路徑
            video_dir = os.path.join(self.img_root, video_rel_path)
            
            if not os.path.exists(video_dir):
                continue
                
            # 取得該資料夾內所有圖片並排序
            # 這裡假設檔名是 frame0.jpg, frame1.jpg... 
            # 為了避免 frame10 排在 frame2 前面，建議可以用自定義排序，這裡先用標準 sorted
            frames = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            # 如果影片幀數不足一個 window_size，可以選擇跳過或補齊 (這裡選擇跳過)
            if len(frames) < self.window_size:
                continue
                
            # 3. 滑動視窗 (Sliding Window) 切分
            # 例如: 總共100幀, window=16, stride=4
            # 樣本1: 0~15, 樣本2: 4~19, 樣本3: 8~23 ...
            for i in range(0, len(frames) - self.window_size + 1, self.stride):
                window_frames = frames[i : i + self.window_size]
                samples.append((window_frames, label))
                
        print(f"建立完成：共 {len(samples)} 個時序樣本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 取得一組連續圖片路徑和標籤
        frame_paths, label = self.samples[idx]
        
        imgs = []
        for path in frame_paths:
            # 讀取圖片
            # path = path.replace('\\', '/')
            image = cv2.imread(path)
            if image is None:
                # 簡單的錯誤處理：若讀不到圖，產生全黑圖以免程式崩潰 (這在訓練大量資料時很有用)
                print(f"Warning: Cannot read {path}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            image = image[:, :, ::-1]  # BGR to RGB
            
            # 套用 Transform (如果有)
            # 注意：如果是隨機 Augmentation (如 RandomCrop)，理論上所有 frame 應該要裁切一樣的位置
            # 為了簡化，這裡假設 transform 是 Resize, ToTensor, Normalize 這種確定性的
            if self.transform is not None:
                # 如果 transform 需要 PIL 格式，請在這裡轉換: image = Image.fromarray(image)
                # 這裡假設 transform 接受 numpy 或 Tensor
                image = self.transform(image)
            
            imgs.append(image)

        # 堆疊圖片
        # 如果 transform 輸出是 Tensor (C, H, W)，stack 後變成 (T, C, H, W)
        # 許多 3D CNN (如 R(2+1)D) 預設輸入是 (B, C, T, H, W)，所以需要 permute
        data_tensor = torch.stack(imgs, dim=0) # (T, C, H, W)
        
        # 轉置維度：變成 (C, T, H, W) -> 這是 PyTorch 3D 卷積的標準輸入格式
        data_tensor = data_tensor.permute(1, 0, 2, 3) 

        return data_tensor, label

class AffectDatasetWindow_facecrop(data.Dataset):
    def __init__(self, img_root, anno, window_size=8, stride=8, transform=None, 
                 face_crop=False, crop_method='yolo', scale=1.3):
        """
        Args:
            img_root (str): 圖片所在的根目錄
            anno (str): 標註檔路徑
            window_size (int): 每個樣本包含的連續幀數 (T)
            stride (int): 滑動視窗的步長
            transform: PyTorch transforms
            face_crop (bool): 是否開啟人臉切圖
            crop_method (str): 'yolo' 或 'mtcnn'
            scale (float): 縮放倍率
        """
        self.img_root = img_root
        self.anno = anno
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.face_crop = face_crop
        self.crop_method = crop_method.lower()
        self.scale = scale
        
        print(f"\nBuilding {anno} dataloader...")

        if face_crop:
            if self.crop_method == 'yolo':
                print("Using YOLO for face detection...")
                head_detect_model_name = 'head_detect_medium'
                try:
                    self.face_model = YOLO(f'./data_preprocessing/{head_detect_model_name}.pt')
                    print(f"{head_detect_model_name} model loaded")
                except:
                    print("Warning: Model not found, downloading standard 'yolov8n.pt'")
                    self.face_model = YOLO('yolov8n.pt')
            
            elif self.crop_method == 'mtcnn':
                print("Using MTCNN for face detection...")
                # device 可以根據是否有 GPU 自動調整
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.face_model = MTCNN(keep_all=False, device=device, post_process=False)
        else:
            self.face_model = None

        # 讀取檔案
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

    def _get_yolo_box(self, image):
        """ YOLO 偵測邏輯：回傳評分最高的人臉座標 """
        results = self.face_model.predict(image, conf=0.3, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        
        h, w = image.shape[:2]
        img_center_x, img_center_y = w / 2, h / 2
        best_box = None
        highest_score = -1

        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 評分邏輯 (與你原始代碼一致)
            face_w, face_h = x2 - x1, y2 - y1
            dist = ( (x1+x2)/2 - img_center_x)**2 + ( (y1+y2)/2 - img_center_y)**2
            score = (conf * face_w * face_h) / (dist + 1)
            
            if score > highest_score:
                highest_score = score
                best_box = [x1, y1, x2, y2]
        return best_box

    def _get_mtcnn_box(self, image):
        """ MTCNN 偵測邏輯：回傳信心度最高的座標 """
        # MTCNN 需要 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = self.face_model.detect(image_rgb)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # 取得信心度最高的索引
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = map(int, boxes[best_idx])
        return [x1, y1, x2, y2]

    def crop_face(self, image, path, last_coordinate):
        """
        統一切圖入口
        """
        h, w = image.shape[:2]
        current_box = None

        # 執行偵測
        if self.crop_method == 'yolo':
            current_box = self._get_yolo_box(image)
        elif self.crop_method == 'mtcnn':
            current_box = self._get_mtcnn_box(image)

        # 如果本次偵測失敗，嘗試沿用上一幀座標
        if current_box is None:
            if last_coordinate is not None:
                x1, y1, x2, y2 = last_coordinate
            else:
                # 完全沒抓到過人臉，回傳原圖
                return image, None
        else:
            # 抓到人臉後進行 Scale 放大處理
            x1, y1, x2, y2 = current_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nw, nh = (x2 - x1) * self.scale, (y2 - y1) * self.scale
            
            x1 = max(0, int(cx - nw/2))
            y1 = max(0, int(cy - nh/2))
            x2 = min(w, int(cx + nw/2))
            y2 = min(h, int(cy + nh/2))

        cropped_img = image[y1:y2, x1:x2]
        
        # 防止切出空圖
        if cropped_img.size == 0:
            return image, last_coordinate
            
        return cropped_img, [x1, y1, x2, y2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        imgs = []
        best_crop_coordinate = None

        for path in frame_paths:
            image = cv2.imread(path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            if self.face_crop:
                image, best_crop_coordinate = self.crop_face(image, path, best_crop_coordinate)

            # 轉 RGB 並 transform
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                image = self.transform(image)
            
            imgs.append(image)

        data_tensor = torch.stack(imgs, dim=0) # (T, C, H, W)
        data_tensor = data_tensor.permute(1, 0, 2, 3) # (C, T, H, W)

        return data_tensor, label, frame_paths


# --- 2. 執行邏輯必須放在這裡 ---
if __name__ == '__main__':
    # 這裡放你的設定
    data_root_dir = r'/home/Dataset/customer_ori2'

    # 定義 Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 實例化 Dataset
    train_dataset = AffectDatasetWindow_facecrop(
        img_root=data_root_dir, 
        train=True, 
        window_size=8,   # 你錯誤訊息中顯示的是 8
        stride=8, 
        transform=transform,
        face_crop=True
    )

    # 實例化 DataLoader (num_workers > 0 時，這段必須在 main 裡面)
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0  # 如果還是報錯，可以先設為 0 測試
    )

    # 測試讀取
    print("Start testing DataLoader...")
    for i, (X, y) in enumerate(train_loader):
        print(f"Batch {i}: X shape={X.shape}, y shape={y.shape}")
        if i >= 2: break # 讀個幾筆就停，不然會跑很久