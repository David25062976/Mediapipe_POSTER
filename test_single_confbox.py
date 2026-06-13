import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg') # 伺服器端繪圖必備
import matplotlib.pyplot as plt
import argparse
import numpy as np
from natsort import natsorted
from collections import OrderedDict
from ultralytics import YOLO

# 引入您的模型定義
from models.emotion_hyp_affect import pyramid_trans_expr

# 定義情緒標籤與顏色
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Like", "Hesitate", "Anger", "Dislike"]
# 對應顏色 (BGR 格式用於 OpenCV繪圖: Blue, Green, Red)
COLORS_BGR = [
    (128, 128, 128), # Neutral (Gray)
    (0, 165, 255),   # Happy (Orange) -> BGR: 0, 165, 255
    (255, 0, 0),     # Sad (Blue)
    (128, 0, 128),   # Surprise (Purple)
    (0, 128, 0),     # Like (Green)
    (255, 255, 0),   # Hesitate (Cyan) -> BGR: 255, 255, 0
    (0, 0, 255),     # Anger (Red)
    (19, 69, 139)    # Dislike (Brown)
]
# Matplotlib 用的顏色 (名稱或 Hex)
COLORS_PLT = ['gray', 'orange', 'blue', 'purple', 'green', 'cyan', 'red', 'brown']

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Emotion Recognition with Tracking')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/customer_ori2/out/let s go/let s go-16',  help='Path to the folder containing video frames')
    parser.add_argument('--checkpoint_single', type=str, default='./checkpoint/20260128-100001/epoch25_acc0.8408.pth', help='Path to the trained model .pth file')
    parser.add_argument('--modeltype', type=str, default='large', help='Model type: small, base, or large')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    return parser.parse_args()

def load_emotion_model(args, device):
    print(f"Loading emotion model: pyramid_trans_expr (type={args.modeltype})...")
    model = pyramid_trans_expr(img_size=224, num_classes=8, type=args.modeltype)
    
    print(f"Loading weights from {args.checkpoint_single}...")
    checkpoint = torch.load(args.checkpoint_single, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def draw_hud(frame, box, probs, predicted_idx):
    """
    在影像上繪製人臉框與詳細的情緒信心儀表板 (HUD)
    """
    x1, y1, x2, y2 = box
    
    # 1. 畫人臉框 (使用預測類別的顏色)
    main_color = COLORS_BGR[predicted_idx]
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    
    # 2. 準備文字背景區塊
    # 計算顯示位置 (優先顯示在人臉右側，如果太靠右則顯示在左側)
    h, w, _ = frame.shape
    hud_x = x2 + 10
    if hud_x + 200 > w: 
        hud_x = x1 - 210 # 移到左邊
    
    hud_y = y1
    line_height = 25
    panel_h = line_height * 8 + 10
    panel_w = 200
    
    # 確保不會畫出邊界
    hud_x = max(0, min(hud_x, w - panel_w))
    hud_y = max(0, min(hud_y, h - panel_h))

    # 畫半透明背景增加文字可讀性
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + panel_w, hud_y + panel_h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 3. 逐行繪製情緒分數
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        text = f"{label}: {prob:.2f}"
        text_y = hud_y + (i + 1) * line_height - 5
        
        if i == predicted_idx:
            # 最高分者：粗體、該情緒顏色
            font_scale = 0.7
            thickness = 2
            text_color = COLORS_BGR[i] # 使用該情緒代表色
            # 畫個小箭頭或標記
            cv2.putText(frame, ">", (hud_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        else:
            # 其他：普通、白色
            font_scale = 0.5
            thickness = 1
            text_color = (255, 255, 255) # White

        cv2.putText(frame, text, (hud_x + 25, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    return frame

def get_best_face_id(boxes, img_shape):
    """
    策略：選擇畫面中心且面積最大的人臉作為追蹤目標
    """
    if not boxes:
        return None
    
    h, w = img_shape[:2]
    center_x, center_y = w / 2, h / 2
    
    best_id = None
    max_score = -1

    for box in boxes:
        if box.id is None: continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2
        
        dist = ((face_cx - center_x)**2 + (face_cy - center_y)**2)**0.5
        area = face_w * face_h
        
        # 分數邏輯：面積越大越好，距離中心越近越好
        score = area / (dist + 1.0)
        
        if score > max_score:
            max_score = score
            best_id = int(box.id.item())
            
    return best_id

def process_video_or_folder(args, model, device, transform):
    # 初始化 Face Model
    print("Loading Face Detection Model...")
    head_detect_model_name = 'head_detect_medium'
    try:
        # 假設您的路徑結構，如果沒有則使用 yolov8n
        face_model_path = f'./data_preprocessing/{head_detect_model_name}.pt'
        if os.path.exists(face_model_path):
             face_model = YOLO(face_model_path)
             print(f"Loaded custom face model: {face_model_path}")
        else:
             print("Custom model not found, downloading YOLOv8n-face/pose...")
             face_model = YOLO('yolov8n-face.pt') 
    except Exception as e:
        print(f"Error loading custom model: {e}. Fallback to yolov8n.pt")
        face_model = YOLO('yolov8n.pt')

    # 設定輸入來源
    input_path = args.image_folder
    if os.path.isdir(input_path):
        # 如果是資料夾，建立圖片列表
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]
        files = natsorted(files)
        image_paths = [os.path.join(input_path, f) for f in files]
        is_video = False
        print(f"Processing folder: {len(image_paths)} images.")
        
        # 設定輸出目錄
        save_dir = input_path.rstrip('/') + '_result'
        
    elif os.path.isfile(input_path):
        # 如果是影片檔案
        cap = cv2.VideoCapture(input_path)
        is_video = True
        print(f"Processing video file: {input_path}")
        
        # 設定輸出目錄 (檔名_result)
        filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
        parent_dir = os.path.dirname(input_path)
        save_dir = os.path.join(parent_dir, filename_no_ext + '_result')
    else:
        print("Error: Input path is not a valid file or directory.")
        return [], [], []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")

    # 追蹤變數
    target_track_id = None
    results_list = []
    frame_indices = []
    all_probs_list = []
    
    frame_count = 0
    scale = 1.2 # Crop 放大倍率

    while True:
        # 1. 讀取影像
        if is_video:
            ret, img_bgr = cap.read()
            if not ret: break
            frame_name = f"frame_{frame_count:05d}.jpg"
        else:
            if frame_count >= len(image_paths): break
            img_path = image_paths[frame_count]
            img_bgr = cv2.imread(img_path)
            frame_name = os.path.basename(img_path)

        if img_bgr is None:
            frame_count += 1
            continue

        h_img, w_img = img_bgr.shape[:2]

        # 2. 人臉追蹤 (YOLO Track)
        # persist=True 讓 tracker 記住上一幀的 ID
        track_results = face_model.track(img_bgr, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        target_box = None
        current_boxes = track_results[0].boxes

        # 3. 鎖定或更新目標 ID
        if target_track_id is None:
            # 尚未鎖定目標，尋找最佳目標
            if current_boxes.id is not None:
                target_track_id = get_best_face_id(current_boxes, img_bgr.shape)
                if target_track_id is not None:
                    print(f"Target Locked! ID: {target_track_id}")

        # 4. 根據 ID 抓取 Box
        found_target = False
        if current_boxes.id is not None and target_track_id is not None:
            ids = current_boxes.id.cpu().numpy().astype(int)
            if target_track_id in ids:
                idx = np.where(ids == target_track_id)[0][0]
                box_tensor = current_boxes.xyxy[idx] # x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, box_tensor)
                
                # 計算放大的 Crop 範圍 (用於模型輸入)
                w_box = x2 - x1
                h_box = y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                new_w_half = (w_box * scale) / 2
                new_h_half = (h_box * scale) / 2
                
                nx1 = max(0, int(cx - new_w_half))
                ny1 = max(0, int(cy - new_h_half))
                nx2 = min(w_img, int(cx + new_w_half))
                ny2 = min(h_img, int(cy + new_h_half))
                
                # 用於 Inference 的圖片
                face_crop_bgr = img_bgr[ny1:ny2, nx1:nx2]
                
                # 用於繪圖的座標 (畫在原圖上，可以用原始座標或放大後的，這裡選用放大後的框以顯示實際輸入範圍)
                draw_coord = (nx1, ny1, nx2, ny2)
                found_target = True

        # 5. 情緒辨識
        if found_target and face_crop_bgr.size > 0:
            # Preprocess
            face_rgb = face_crop_bgr[:, :, ::-1] # BGR to RGB
            input_tensor = transform(face_rgb)
            input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dim

            with torch.no_grad():
                outputs, _ = model(input_tensor)
                probs = F.softmax(outputs, dim=1) # [1, 8]
                probs_np = probs.cpu().numpy()[0]
                _, predicted = torch.max(outputs, 1)
                pred_label_idx = predicted.item()

            # 記錄結果
            results_list.append(pred_label_idx)
            frame_indices.append(frame_count + 1)
            all_probs_list.append(probs_np)

            # 6. 繪圖 (HUD)
            img_draw = img_bgr.copy()
            img_draw = draw_hud(img_draw, draw_coord, probs_np, pred_label_idx)
            
            # 儲存圖片
            save_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(save_path, img_draw)
        
        else:
            # 如果沒抓到人臉，或者是目標遺失
            # 選擇性：如果目標遺失太久，可以重設 target_track_id = None (這裡暫不實作複雜邏輯)
            # 仍然儲存原圖以便查看
            save_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(save_path, img_bgr)

        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}...")
        
        frame_count += 1

    if is_video:
        cap.release()

    return frame_indices, results_list, np.array(all_probs_list), save_dir

def plot_results(x, y, probs_matrix, save_dir):
    save_path = os.path.join(save_dir, 'result_graph.png')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Subplot 1: Predictions
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3, linewidth=1.5, color='black', label='Prediction')
    ax1.set_title("Frame-by-Frame Classification Result", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Emotion", fontsize=12)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(EMOTION_LABELS)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 7.5)

    # Subplot 2: Probabilities
    for class_idx in range(8):
        class_probs = probs_matrix[:, class_idx]
        ax2.plot(x, class_probs, 
                 label=f"{EMOTION_LABELS[class_idx]}", 
                 color=COLORS_PLT[class_idx], 
                 linewidth=1.5, 
                 alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.close()

def main():
    args = parse_args()
    
    # 設定 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 載入情緒模型
    emotion_model = load_emotion_model(args, device)
    transform = get_transforms()

    # 執行推論 (含追蹤與繪圖)
    if os.path.exists(args.image_folder):
        frames, preds, probs, save_dir = process_video_or_folder(args, emotion_model, device, transform)
        
        if len(frames) > 0:
            # 繪製總結折線圖
            plot_results(frames, preds, probs, save_dir)
            print(f"Processing complete. Results saved in: {save_dir}")
        else:
            print("No frames processed.")
    else:
        print(f"Error: Path {args.image_folder} does not exist.")

if __name__ == "__main__":
    main()
