import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_all_landmarks(results, img_w=224, img_h=224):
    """
    將 MediaPipe 的正規化座標轉換為 224x224 的像素座標與深度，
    回傳完整的 478 個點 (x, y, z)。
    """
    coords = np.zeros((478, 3), dtype=np.float32)
    
    if not results.face_landmarks:
        return coords, False

    for i, lm in enumerate(results.face_landmarks[0]):
        coords[i] = [lm.x * img_w, lm.y * img_h, lm.z]
        
    return coords, True

def process_and_save(dataset_root, list_file, save_path):
    # 啟動 MediaPipe
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    db_dict = {}
    
    if not os.path.exists(list_file):
        print(f"Error: {list_file} 不存在！")
        return

    # 讀取 txt 檔案
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 以影片資料夾為單位顯示進度條
    for line in tqdm(lines, desc=f"Processing {list_file}"):
        parts = line.split()
        if len(parts) < 2: continue
        
        rel_folder = parts[0] # 資料夾的相對路徑 (例如: 3_buy/8446796-hd_...)
        label = int(parts[1])
        full_folder_path = os.path.join(dataset_root, rel_folder)
        
        # 檢查該影片資料夾是否存在
        if not os.path.isdir(full_folder_path):
            continue
            
        # 抓取資料夾內的所有圖片，並進行排序以確保時間序列正確
        frame_files = sorted([
            f for f in os.listdir(full_folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # 逐幀處理
        for frame_name in frame_files:
            img_path = os.path.join(full_folder_path, frame_name)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
            
            # 執行偵測
            res = detector.detect(mp_image)
            coords, face_found = get_all_landmarks(res, 224, 224)
            
            # 將每一幀的相對路徑作為 key (例如: 3_buy/8446796.../frame_0001.jpg)
            frame_rel_path = f"{rel_folder}/{frame_name}"
            
            db_dict[frame_rel_path] = {
                'coords': coords, 
                'face_found': face_found,
                'label': label,
                'video_dir': rel_folder # 額外記錄所屬的影片資料夾，方便後續用來分組 (grouping)
            }

    # 儲存
    torch.save(db_dict, save_path)
    print(f"Saved {len(db_dict)} frames to {save_path}")

if __name__ == "__main__":
    # 請確認圖片所在資料夾與兩個 txt 檔案的路徑與你的環境一致
    dataset_root = '/home/lab702/POSTER/data/5_classes' # 如果你的 txt 內的路徑是從 dataset 根目錄開始算

    # 處理 Test / Valid 資料
    process_and_save(
        dataset_root=dataset_root, 
        list_file='/home/lab702/POSTER/data/5_classes/valid_list.txt', 
        save_path='/home/lab702/POSTER/data/5_classes/valid_landmarks.pt'
    )

    # 處理 Train 資料
    process_and_save(
        dataset_root=dataset_root, 
        list_file='/home/lab702/POSTER/data/5_classes/train_list.txt', 
        save_path='/home/lab702/POSTER/data/5_classes/train_landmarks.pt'
    )