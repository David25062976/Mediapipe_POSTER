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
    只負責將 MediaPipe 的正規化座標轉換為 224x224 的像素座標與深度，
    回傳完整的 478 個點 (x, y, z)。
    """
    coords = np.zeros((478, 3), dtype=np.float32)
    
    # 如果 MediaPipe 沒有偵測到臉，回傳全零陣列與 False
    if not results.face_landmarks:
        return coords, False

    for i, lm in enumerate(results.face_landmarks[0]):
        # 保存 x, y (放大至 224 尺度) 以及原始 z 深度
        coords[i] = [lm.x * img_w, lm.y * img_h, lm.z]
        
    return coords, True

def process_and_save(root_dir, split_name):
    data_dir = os.path.join(root_dir, split_name)
    save_path = os.path.join(root_dir, f"{split_name}_landmarks_all.pt")
    
    # 啟動 MediaPipe
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    db_dict = {}
    
    for label_name in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_path): continue
        
        for fname in tqdm(os.listdir(label_path), desc=f"Processing {split_name}/{label_name}"):
            img_path = os.path.join(label_path, fname)
            rel_path = f"{label_name}/{fname}" # 例如 "0/image001.jpg"
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 統一 Resize 到 224x224 (與模型訓練時一致)
            img = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
            
            # 執行偵測
            res = detector.detect(mp_image)
            coords, face_found = get_all_landmarks(res, 224, 224)
            
            # 儲存完整的 478 點座標，以及是否有偵測到臉的標記
            db_dict[rel_path] = {'coords': coords, 'face_found': face_found}

    torch.save(db_dict, save_path)
    print(f"Saved {len(db_dict)} records to {save_path}")

if __name__ == "__main__":
    # 請確認資料夾名稱與你的環境一致
    dataset_root = './data/AffectNet8c_Shopping_test'
    process_and_save(dataset_root, 'train_set')
    process_and_save(dataset_root, 'valid_set')
    process_and_save(dataset_root, 'test_set')
