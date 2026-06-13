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
import mediapipe as mp
from facenet_pytorch import MTCNN

# 引入您的模型定義
from models.emotion_hyp_affect import pyramid_trans_expr

# === 標籤與顏色定義 ===
EMOTION_LABELS_8 = ["Neutral", "Happy", "Sad", "Surprise", "Like", "Hesitate", "Anger", "Dislike"]
COLORS_BGR_8 = [
    (128, 128, 128), (0, 165, 255), (255, 0, 0), (128, 0, 128),
    (0, 128, 0), (255, 255, 0), (0, 0, 255), (19, 69, 139)
]

# 3類模型專用 (Like, Hesitate, Dislike)
EMOTION_LABELS_3 = ["Like", "Hesitate", "Dislike"]
COLORS_BGR_3 = [(0, 128, 0), (255, 255, 0), (19, 69, 139)]
# 將 3 類的 Index 映射到 8 類的 Index (4: Like, 5: Hesitate, 7: Dislike)，以利繪圖
MAP_3_TO_8 = {0: 4, 1: 5, 2: 7}

COLORS_PLT = ['gray', 'orange', 'blue', 'purple', 'green', 'cyan', 'red', 'brown']

def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid Inference for Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/customer_ori2/out/RESTOCKING MY SHOWER/RESTOCKING MY SHOWER-7', help='Path to video or image folder')
    parser.add_argument('--checkpoint_8class', type=str, default='./checkpoint/20260128-100001/epoch25_acc0.8408.pth', help='Path to 8-class model')
    parser.add_argument('--checkpoint_3class', type=str, default='./checkpoint/20260302-113908/epoch10_acc0.9937.pth', help='Path to 3-class model (Like, Hesitate, Dislike)')
    parser.add_argument('--modeltype', type=str, default='large', help='Model type')
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID')
    return parser.parse_args()

def load_emotion_model(checkpoint_path, model_type, num_classes, device):
    print(f"Loading emotion model (Classes: {num_classes}) from {checkpoint_path}...")
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=model_type)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1]) 
    return -1

def get_pure_yaw_rgb(image_rgb, face_mesh):
    """
    接收 RGB 圖片，使用 MediaPipe 與 PnP 解算 Yaw 角度
    """
    img_h, img_w, _ = image_rgb.shape
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    
    # 建立標準 3D 模型
    face_3d_model = np.array([
        (0.0, 0.0, 0.0),             # 鼻尖
        (0.0, -330.0, -65.0),        # 下巴
        (-225.0, 170.0, -135.0),     # 左眼左角
        (225.0, 170.0, -135.0),      # 右眼右角
        (-150.0, -150.0, -125.0),    # 左嘴角
        (150.0, -150.0, -125.0)      # 右嘴角
    ], dtype=np.float64)

    key_indices = [1, 199, 33, 263, 61, 291]
    
    face_2d = []
    for idx in key_indices:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])

    face_2d = np.array(face_2d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    yaw = angles[1] * 360
    return yaw

def draw_hud(frame, box, probs, predicted_idx, labels, colors, pose_status):
    x1, y1, x2, y2 = box
    
    # 決定主要顏色
    if pose_status == "Back Head":
        main_color = (100, 100, 100) # 灰色代表後腦杓
    else:
        main_color = colors[predicted_idx]
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
    
    # 在框的上方寫上姿態狀態
    cv2.putText(frame, pose_status, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
    
    # 如果是後腦杓，不畫 HUD 機率表
    if probs is None:
        return frame
        
    h, w, _ = frame.shape
    hud_x = max(0, min(x2 + 10 if x2 + 210 < w else x1 - 210, w - 210))
    hud_y = max(0, min(y1, h - 220))

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + 200, hud_y + 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        text_y = hud_y + (i + 1) * 25 - 5
        color = colors[i] if i == predicted_idx else (255, 255, 255)
        thickness = 2 if i == predicted_idx else 1
        font_scale = 0.7 if i == predicted_idx else 0.5
        
        if i == predicted_idx:
            cv2.putText(frame, ">", (hud_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, f"{label}: {prob:.2f}", (hud_x + 25, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame

def get_best_face_id(boxes, img_shape):
    if not boxes: return None
    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2
    best_id, max_score = None, -1
    if boxes.id is None: return None

    for box in boxes:
        if box.id is None: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_cx, face_cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        dist = ((face_cx - cx)**2 + (face_cy - cy)**2)**0.5
        score = area / (dist + 1.0)
        if score > max_score:
            max_score = score
            best_id = int(box.id.item())
    return best_id

def process_video_or_folder(args, model_8c, model_3c, device, transform):
    # 載入 YOLOv8
    try:
        yolo_model = YOLO(f'./data_preprocessing/head_detect_medium.pt')
    except:
        print("Using YOLOv8n-face fallback.")
        yolo_model = YOLO('yolov8n.pt')

    # 初始化 MTCNN 與 MediaPipe
    mtcnn = MTCNN(keep_all=False, device=device)
    mp_face_mesh = mp.solutions.face_mesh
    # 對於影片序列，static_image_mode 設為 False 效能較佳且具備時序平滑
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    input_path = args.image_folder
    image_paths = []
    is_video = False
    
    if os.path.isdir(input_path):
        files = natsorted([f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])
        image_paths = [os.path.join(input_path, f) for f in files]
        save_dir = input_path.rstrip('/') + '_result'
    elif os.path.isfile(input_path):
        is_video = True
        cap = cv2.VideoCapture(input_path)
        save_dir = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0] + '_result')
    else:
        return [], [], [], ""

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    target_track_id = None
    results_list, frame_indices, all_probs_list = [], [], []
    frame_count, consecutive_lost_count = 0, 0

    while True:
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

        frame_indices.append(current_frame_id)

        if img_bgr is None:
            results_list.append(np.nan)
            all_probs_list.append(np.full(8, np.nan))
            frame_count += 1
            continue

        h_img, w_img = img_bgr.shape[:2]
        track_results = yolo_model.track(img_bgr, persist=True, verbose=False, tracker="bytetrack.yaml")
        boxes = track_results[0].boxes

        found_target, face_crop, draw_coord = False, None, None
        current_ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else []

        if target_track_id is not None and target_track_id in current_ids:
            consecutive_lost_count = 0 
        else:
            if target_track_id is not None: consecutive_lost_count += 1
            new_best_id = get_best_face_id(boxes, img_bgr.shape)
            if new_best_id is not None:
                target_track_id = new_best_id
                consecutive_lost_count = 0

        if target_track_id is not None and target_track_id in current_ids:
            idx = current_ids.index(target_track_id)
            box = boxes.xyxy[idx]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            w_b, h_b = x2-x1, y2-y1
            nx1, ny1 = max(0, int(cx - w_b*0.6)), max(0, int(cy - h_b*0.6))
            nx2, ny2 = min(w_img, int(cx + w_b*0.6)), min(h_img, int(cy + h_b*0.6))
            
            face_crop = img_bgr[ny1:ny2, nx1:nx2]
            draw_coord = (nx1, ny1, nx2, ny2)
            found_target = True

        if found_target and face_crop.size > 0:
            face_rgb = face_crop[:, :, ::-1]
            
            # --- 判斷機制核心區塊 ---
            # 1. MTCNN 驗證是否為後腦杓
            boxes_mtcnn, _ = mtcnn.detect(face_rgb)
            
            if boxes_mtcnn is None:
                # 判定為後腦杓
                img_draw = draw_hud(img_bgr.copy(), draw_coord, None, None, None, None, "Back Head")
                results_list.append(np.nan)
                all_probs_list.append(np.full(8, np.nan))
                cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)
            else:
                # 2. MediaPipe 判斷正側臉
                yaw = get_pure_yaw_rgb(face_rgb, face_mesh)
                abs_yaw = abs(yaw) if yaw is not None else 99999
                
                input_tensor = transform(face_rgb).unsqueeze(0).to(device)
                
                # 延續原本 sort_faces.py 的閾值標準
                if yaw is not None and abs_yaw <= 12000:
                    # 正臉：使用 8 類模型
                    with torch.no_grad():
                        outputs, _ = model_8c(input_tensor)
                        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                        pred = torch.max(outputs, 1)[1].item()

                    img_draw = draw_hud(img_bgr.copy(), draw_coord, probs, pred, EMOTION_LABELS_8, COLORS_BGR_8, "Front Face")
                    results_list.append(pred)
                    all_probs_list.append(probs)
                else:
                    # 側臉：使用 3 類模型 (Like, Hesitate, Dislike)
                    with torch.no_grad():
                        outputs, _ = model_3c(input_tensor)
                        probs_3c = F.softmax(outputs, dim=1).cpu().numpy()[0]
                        pred_3c = torch.max(outputs, 1)[1].item()
                    
                    img_draw = draw_hud(img_bgr.copy(), draw_coord, probs_3c, pred_3c, EMOTION_LABELS_3, COLORS_BGR_3, "Side Face")
                    
                    # 將 3 類的結果轉譯回 8 類的格式，確保圖表正確繪製
                    mapped_pred = MAP_3_TO_8[pred_3c]
                    mapped_probs = np.zeros(8)
                    for idx_3c, prob_val in enumerate(probs_3c):
                        mapped_probs[MAP_3_TO_8[idx_3c]] = prob_val
                        
                    results_list.append(mapped_pred)
                    all_probs_list.append(mapped_probs)

                cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)
        else:
            results_list.append(np.nan)
            all_probs_list.append(np.full(8, np.nan))
            cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)

        if frame_count % 100 == 0: 
            print(f"Processed: {frame_name} -> ID: {current_frame_id}")
        frame_count += 1

    if is_video: cap.release()
    face_mesh.close() # 釋放資源
    return frame_indices, results_list, np.array(all_probs_list), save_dir

def plot_results(x, y, probs_matrix, save_dir):
    # 此部分程式碼完全不用動，因為已經透過 Mapper 轉換好了
    save_path = os.path.join(save_dir, 'result_graph.png')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3, linewidth=1.5, color='black', label='Prediction')
    ax1.set_title("Frame-by-Frame Classification Result (Hybrid Model)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Emotion", fontsize=12)
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(EMOTION_LABELS_8)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 7.5)

    for class_idx in range(8):
        class_probs = probs_matrix[:, class_idx]
        ax2.plot(x, class_probs, label=f"{EMOTION_LABELS_8[class_idx]}", color=COLORS_PLT[class_idx], linewidth=1.5, alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame ID", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    if len(x) > 0:
        x_min, x_max = min(x), max(x)
        ax1.set_xlim(x_min - 1, x_max + 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 分別載入兩個模型
    model_8class = load_emotion_model(args.checkpoint_8class, args.modeltype, num_classes=8, device=device)
    model_3class = load_emotion_model(args.checkpoint_3class, args.modeltype, num_classes=3, device=device)
    
    if os.path.exists(args.image_folder):
        frames, preds, probs, sdir = process_video_or_folder(args, model_8class, model_3class, device, get_transforms())
        if len(frames) > 0:
            plot_results(frames, preds, probs, sdir)
            print(f"Done! Results in {sdir}")
    else:
        print("Path not found.")
