import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse
import numpy as np
from natsort import natsorted
from collections import OrderedDict
from ultralytics import YOLO
import re 
import csv

# 引入您的模型定義
from models.emotion_hyp_affect import pyramid_trans_expr

# EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interesting', 'thinking', 'buying', 'passing']
EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']

COLORS_BGR = [
    (128, 128, 128), (0, 165, 255), (255, 0, 0), (128, 0, 128),
    (0, 128, 0), (255, 255, 0), (0, 0, 255), (19, 69, 139)
]
COLORS_PLT = ['gray', 'orange', 'blue', 'purple', 'green', 'cyan', 'red', 'brown']

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-3', help='Path to image folder')    # /home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-4
    parser.add_argument('--checkpoint_single', type=str, default='./checkpoint/20260427-155438_base_ori/best.pth', help='Path to model')    # ./checkpoint/20260502-132804_Shopping_base/best.pth
    parser.add_argument('--modeltype', type=str, default='large', help='Model type')
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
    return parser.parse_args()

def load_emotion_model(args, device):
    print(f"Loading emotion model: pyramid_trans_expr (type={args.modeltype})...")
    model = pyramid_trans_expr(img_size=224, num_classes=8, type=args.modeltype)
    checkpoint = torch.load(args.checkpoint_single, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
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
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_frame_number(filename):
    """
    從檔名提取數字，例如 "01130.jpg" -> 1130
    """
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1]) 
    return -1

def draw_hud(frame, box, probs, predicted_idx):
    x1, y1, x2, y2 = box
    main_color = COLORS_BGR[predicted_idx]
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    
    h, w, _ = frame.shape
    hud_x = max(0, min(x2 + 10 if x2 + 210 < w else x1 - 210, w - 210))
    hud_y = max(0, min(y1, h - 220))

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + 200, hud_y + 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        text_y = hud_y + (i + 1) * 25 - 5
        color = COLORS_BGR[i] if i == predicted_idx else (255, 255, 255)
        thickness = 2 if i == predicted_idx else 1
        font_scale = 0.7 if i == predicted_idx else 0.5
        
        if i == predicted_idx:
            cv2.putText(frame, ">", (hud_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, f"{label}: {prob:.2f}", (hud_x + 25, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

def get_best_face_id(boxes, img_shape):
    """
    找出最佳的人臉 ID (面積大且靠近中心)
    """
    if not boxes: return None
    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2
    best_id, max_score = None, -1
    
    # 確保 boxes.id 存在
    if boxes.id is None: return None

    for box in boxes:
        # 如果追蹤器還沒分配 ID，跳過
        if box.id is None: continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2
        
        # 分數邏輯：面積 / (距離中心 + 1)
        area = (x2 - x1) * (y2 - y1)
        dist = ((face_cx - cx)**2 + (face_cy - cy)**2)**0.5
        score = area / (dist + 1.0)
        
        if score > max_score:
            max_score = score
            best_id = int(box.id.item())
            
    return best_id

def process_video_or_folder(args, model, device, transform):
    try:
        face_model = YOLO(f'./data_preprocessing/head_detect_medium.pt')
    except:
        print("Using YOLOv8n-face fallback.")
        face_model = YOLO('yolov8n.pt')

    input_path = args.image_folder
    image_paths = []
    is_video = False
    
    if os.path.isdir(input_path):
        files = natsorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])
        image_paths = [os.path.join(input_path, f) for f in files]
        save_dir = input_path.rstrip('/') + '_result_AffectNet'
        print(f"DEBUG: Found {len(image_paths)} images in folder.") 
    elif os.path.isfile(input_path):
        is_video = True
        cap = cv2.VideoCapture(input_path)
        save_dir = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0] + '_result_AffectNet')
    else:
        return [], [], [], ""

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    target_track_id = None
    results_list, frame_indices, all_probs_list = [], [], []
    frame_count = 0
    consecutive_lost_count = 0  # 記錄連續跟丟幾幀

    while True:
        # 1. 決定幀的名稱與來源
        if is_video:
            ret, img_bgr = cap.read()
            if not ret: break
            frame_name = f"frame_{frame_count:05d}.jpg"
            current_frame_id = frame_count + 1
        else:
            if frame_count >= len(image_paths): break
            img_path = image_paths[frame_count]
            frame_name = os.path.basename(img_path)
            extracted = extract_frame_number(frame_name)
            current_frame_id = extracted if extracted != -1 else (frame_count + 1)
            img_bgr = cv2.imread(img_path)

        # 2. 記錄 Frame ID
        frame_indices.append(current_frame_id)

        # 3. 檢查圖片
        if img_bgr is None:
            results_list.append(np.nan)
            all_probs_list.append(np.full(8, np.nan))
            frame_count += 1
            continue

        # 4. 人臉追蹤
        h_img, w_img = img_bgr.shape[:2]
        # 使用 persist=True 保持 ID
        track_results = face_model.track(img_bgr, persist=True, verbose=False, tracker="bytetrack.yaml")
        boxes = track_results[0].boxes

        found_target = False
        face_crop = None
        draw_coord = None
        
        # === 核心邏輯修正：動態更新 target_track_id ===
        current_ids = []
        if boxes.id is not None:
            current_ids = boxes.id.cpu().numpy().astype(int).tolist()

        # 情況 A: 目標 ID 還在畫面中 -> 繼續追
        if target_track_id is not None and target_track_id in current_ids:
            consecutive_lost_count = 0 # 重置跟丟計數
        
        # 情況 B: 目標 ID 不見了 (或尚未鎖定) -> 嘗試尋找新的最佳目標
        else:
            if target_track_id is not None:
                consecutive_lost_count += 1
                # print(f"Target {target_track_id} lost for {consecutive_lost_count} frames...")
            
            # 如果目前有偵測到任何人臉，就嘗試切換目標
            # (這裡可以加一個邏輯：如果跟丟超過 N 幀才切換，或者立刻切換。為了連續性，我們選擇立刻尋找最佳候選人)
            new_best_id = get_best_face_id(boxes, img_bgr.shape)
            
            if new_best_id is not None:
                if target_track_id is not None:
                    print(f"Switching target from ID {target_track_id} -> {new_best_id}")
                else:
                    print(f"Locked on new target ID: {new_best_id}")
                
                target_track_id = new_best_id
                consecutive_lost_count = 0

        # === 執行裁切與推論 ===
        if target_track_id is not None and target_track_id in current_ids:
            idx = current_ids.index(target_track_id)
            box = boxes.xyxy[idx]
            x1, y1, x2, y2 = map(int, box)
            
            # Crop logic
            cx, cy = (x1+x2)//2, (y1+y2)//2
            w_b, h_b = x2-x1, y2-y1
            nx1, ny1 = max(0, int(cx - w_b*0.6)), max(0, int(cy - h_b*0.6))
            nx2, ny2 = min(w_img, int(cx + w_b*0.6)), min(h_img, int(cy + h_b*0.6))
            
            face_crop = img_bgr[ny1:ny2, nx1:nx2]
            draw_coord = (nx1, ny1, nx2, ny2)
            found_target = True

        if found_target and face_crop.size > 0:
            face_rgb = face_crop[:, :, ::-1]
            input_tensor = transform(face_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs, _ = model(input_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = torch.max(outputs, 1)[1].item()

            results_list.append(pred)
            all_probs_list.append(probs)
            img_draw = draw_hud(img_bgr.copy(), draw_coord, probs, pred)
            cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)
        else:
            results_list.append(np.nan)
            all_probs_list.append(np.full(8, np.nan))
            cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)

        if frame_count % 100 == 0: 
            print(f"Processed: {frame_name} -> ID: {current_frame_id}")
            
        frame_count += 1

    if is_video: cap.release()
    return frame_indices, results_list, np.array(all_probs_list), save_dir

def plot_results(x, y, probs_matrix, save_dir):
    # === 1. 新增：儲存 CSV 檔案 ===
    csv_path = os.path.join(save_dir, 'result_data.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 寫入 CSV 標題列 (包含 Frame_ID, 預測結果, 及各情緒的欄位名稱)
        header = ['Frame_ID', 'Classification', 'Prediction'] + EMOTION_LABELS
        writer.writerow(header)
        
        for i in range(len(x)):
            frame_id = x[i]
            pred_idx = y[i]
            
            # 處理無法偵測到人臉的情況 (pred_idx 會是 NaN)
            if np.isnan(pred_idx):
                pred_label = 'NaN'
            else:
                pred_label = EMOTION_LABELS[int(pred_idx)]
            
            # 將資料組合成一個列表並寫入該 row
            row = [frame_id, int(pred_idx), pred_label] + probs_matrix[i].tolist()
            writer.writerow(row)
            
    print(f"Data saved to {csv_path}")

    # === 2. 原本的：儲存圖表檔案 ===
    save_path = os.path.join(save_dir, 'result_graph.png')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # === Subplot 1: 預測類別 ===
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3, linewidth=1.5, color='black', label='Prediction')
    ax1.set_title("Frame-by-Frame Classification Result", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Emotion", fontsize=12)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(EMOTION_LABELS)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 7.5)

    # === Subplot 2: 信心程度 ===
    for class_idx in range(8):
        class_probs = probs_matrix[:, class_idx]
        ax2.plot(x, class_probs, 
                 label=f"{EMOTION_LABELS[class_idx]}", 
                 color=COLORS_PLT[class_idx], 
                 linewidth=1.5, 
                 alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame ID", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    # === 強制設定 X 軸範圍 ===
    if len(x) > 0:
        x_min, x_max = min(x), max(x)
        ax1.set_xlim(x_min - 1, x_max + 1)
        print(f"DEBUG: Force X-axis limit to {x_min} - {x_max}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_emotion_model(args, device)
    
    if os.path.exists(args.image_folder):
        frames, preds, probs, sdir = process_video_or_folder(args, model, device, get_transforms())
        if len(frames) > 0:
            plot_results(frames, preds, probs, sdir)
            print(f"Done! Results in {sdir}")
    else:
        print("Path not found.")
