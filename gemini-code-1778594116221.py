"""
test_single_confbox_mediapipe_pt.py

架構參考 test_single_confbox_shopping.py，
模型改為 emotion_hyp_affect_mediapipe_pt.py（train_affect_mediapipe_pt.py 訓練出的版本）。

主要差異：
  - 舊版 (shopping)   : model(input_tensor)            → 只需要影像
  - 本版 (mediapipe)  : model(input_tensor, coords_478) → 需要影像 + 即時 MediaPipe 偵測

流程：
  YOLO 偵測/追蹤人臉
  → 裁切人臉 (224×224)
  → MediaPipe 偵測人臉 478 個 landmarks
  → 組成 coords_478 tensor [1, 478, 3]
  → model(image_tensor, coords_478_tensor) → 預測
"""

import os
import re
import csv
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from natsort import natsorted

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from models.emotion_hyp_affect_mediapipe_pt import pyramid_trans_expr

# ══════════════════════════════════════════════════════════════════
#  상수
# ══════════════════════════════════════════════════════════════════
EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interesting', 'thinking', 'buying', 'passing']

COLORS_BGR = [
    (128, 128, 128),  # Neutral  - 灰
    (0, 165, 255),    # Happy    - 橘
    (255, 0, 0),      # Sad      - 藍
    (128, 0, 128),    # Surprise - 紫
    (0, 128, 0),      # Interest - 綠
    (255, 255, 0),    # Thinking - 青
    (0, 0, 255),      # Anger    - 紅
    (19, 69, 139),    # Passing  - 棕
]

COLORS_PLT = ['gray', 'orange', 'blue', 'purple',
              'green', 'cyan', 'red', 'brown']


# ══════════════════════════════════════════════════════════════════
#  引數
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Mediapipe-PT Emotion Model')

    parser.add_argument('--image_folder',         type=str, default='/home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-3',
                        help='影像資料夾路徑 或 影片檔案路徑')    # /home/Dataset/customer_ori2/out/let s go/let s go-16
    parser.add_argument('--name_notes',           type=str, default='_result',
                        help='result 資料夾的名稱備註')
    parser.add_argument('--checkpoint',           type=str, default='./checkpoint/20260430-220136_Shopping_49_14/best.pth',
                        help='訓練好的 .pth checkpoint 路徑')    # 20260430-220136_Shopping_49_14, 20260429-184718_Shopping_49_24, 20260501-060739_Shopping_196_14, 20260429-184618_Shopping_196_24

    # ── 模型超參數（需與訓練時完全一致）──
    parser.add_argument('--mediapipe_points',     type=int, default=49,
                        choices=[49, 196],
                        help='訓練時使用的 mediapipe_points (49 or 196)')
    parser.add_argument('--mediapipe_patch_size', type=int, default=14,
                        choices=[14, 24],
                        help='訓練時使用的 mediapipe_patch_size (14 or 24)')
    parser.add_argument('--num_classes',          type=int, default=8)
    parser.add_argument('--modeltype',            type=str, default='large',
                        choices=['small', 'base', 'large'])

    # ── MediaPipe 模型路徑 ──
    parser.add_argument('--mediapipe_model_path', type=str,
                        default='models/face_landmarker.task',
                        help='face_landmarker.task 檔案路徑')

    # ── YOLO 模型路徑 ──
    parser.add_argument('--yolo_model',           type=str,
                        default='./data_preprocessing/head_detect_medium.pt',
                        help='YOLO 人臉偵測模型路徑')

    parser.add_argument('--gpu',                  type=str, default='0')

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
#  MediaPipe 初始化
# ══════════════════════════════════════════════════════════════════
def build_mediapipe_detector(model_path: str):
    """
    建立 MediaPipe FaceLandmarker。
    使用 CPU Delegate（GPU Delegate 在部分環境下會與 PyTorch CUDA 衝突）。
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(detector, face_bgr: np.ndarray, target_size: int = 224, last_coords_478=None):
    """
    輸入：BGR 格式的人臉裁切影像（任意大小）
    輸出：coords_478  np.ndarray [478, 3]  (x, y 為 target_size 空間的像素座標)
          face_found  bool
    """
    # resize 到模型期望的輸入尺寸
    face_resized = cv2.resize(face_bgr, (target_size, target_size))
    face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(mp.ImageFormat.SRGB, np.ascontiguousarray(face_rgb))

    result = detector.detect(mp_image)

    coords_478 = np.zeros((478, 3), dtype=np.float32)
    if not result.face_landmarks:
        if last_coords_478 is None:
            return coords_478, False
        else:
            return last_coords_478, False

    for i, lm in enumerate(result.face_landmarks[0]):
        coords_478[i] = [
            lm.x * target_size,   # pixel x
            lm.y * target_size,   # pixel y
            lm.z,                 # depth (歸一化)
        ]
    return coords_478, True


# ══════════════════════════════════════════════════════════════════
#  감정 모델 로드
# ══════════════════════════════════════════════════════════════════
def load_emotion_model(args, device):
    print(f"載入感情模型 pyramid_trans_expr "
          f"(type={args.modeltype}, points={args.mediapipe_points}, "
          f"patch_size={args.mediapipe_patch_size})...")

    model = pyramid_trans_expr(
        img_size=224,
        num_classes=args.num_classes,
        type=args.modeltype,
        mediapipe_points=args.mediapipe_points,
        mediapipe_patch_size=args.mediapipe_patch_size,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 移除 DataParallel 的 "module." 前綴
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("模型載入完成。")
    return model


# ══════════════════════════════════════════════════════════════════
#  Transform（與訓練 val transform 完全一致）
# ══════════════════════════════════════════════════════════════════
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════════════════════════
def extract_frame_number(filename: str) -> int:
    """從檔名提取數字，例如 "01130.jpg" -> 1130"""
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1


def get_best_face_id(boxes, img_shape):
    """找出面積最大且最靠近畫面中心的追蹤 ID。"""
    if boxes.id is None:
        return None

    h, w    = img_shape[:2]
    cx, cy  = w / 2, h / 2
    best_id, max_score = None, -1

    for box in boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2
        area    = (x2 - x1) * (y2 - y1)
        dist    = ((face_cx - cx) ** 2 + (face_cy - cy) ** 2) ** 0.5
        score   = area / (dist + 1.0)

        if score > max_score:
            max_score = score
            best_id   = int(box.id.item())

    return best_id


def draw_hud(frame, box, probs, predicted_idx, coords=None, face_224=None, face_found=False, show_landmarks=True):
    """在影像上繪製偵測框 + 浮動式 HUD 機率列表。"""
    x1, y1, x2, y2  = box
    main_color = COLORS_BGR[predicted_idx]
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)

    h, w, _ = frame.shape
    hud_x = max(0, min(x2 + 10 if x2 + 210 < w else x1 - 210, w - 210))
    hud_y = max(0, min(y1, h - 220))

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y),
                  (hud_x + 200, hud_y + 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        text_y      = hud_y + (i + 1) * 25 - 5
        color       = COLORS_BGR[i] if i == predicted_idx else (255, 255, 255)
        thickness   = 2 if i == predicted_idx else 1
        font_scale  = 0.7 if i == predicted_idx else 0.5

        if i == predicted_idx:
            cv2.putText(frame, ">", (hud_x + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, f"{label}: {prob:.2f}", (hud_x + 25, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # ── 新增：在 face_224 上畫特徵點，並疊加至左上角 ──
    if show_landmarks and coords is not None and face_224 is not None:
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
            
        if coords.ndim == 3:
            coords = coords.squeeze(0)
            
        # 1. 在 face_224 (224x224) 上畫點
        for pt in coords:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < 224 and 0 <= y < 224:
                # radius=1 讓點極小, thickness=-1 為實心, 白色不干擾表情
                cv2.circle(face_224, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
        
        # 2. 將 face_224 疊加到主畫面 (frame) 的左上角
        pip_h, pip_w = face_224.shape[:2]
        if h >= pip_h and w >= pip_w:
            frame[0:pip_h, 0:pip_w] = face_224
            # 畫一個白框框住子畫面，讓它看起來更像一個 HUD 面板
            if face_found:
                cv2.rectangle(frame, (0, 0), (pip_w, pip_h), (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (0, 0), (pip_w, pip_h), (0, 0, 255), 2)

    return frame


# ══════════════════════════════════════════════════════════════════
#  主推理循環
# ══════════════════════════════════════════════════════════════════
def process_video_or_folder(args, model, device, transform, mp_detector):
    # ── YOLO 載入 ──
    try:
        face_model = YOLO(args.yolo_model)
    except Exception:
        print("指定的 YOLO 模型載入失敗，改用 yolov8n.pt。")
        face_model = YOLO('yolov8n.pt')

    input_path = args.image_folder
    image_paths = []
    is_video    = False

    if os.path.isdir(input_path):
        files       = natsorted([f for f in os.listdir(input_path)
                                  if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])
        image_paths = [os.path.join(input_path, f) for f in files]
        save_dir    = input_path.rstrip('/') + args.name_notes
        print(f"找到 {len(image_paths)} 張影像。")

    elif os.path.isfile(input_path):
        is_video = True
        cap      = cv2.VideoCapture(input_path)
        save_dir = os.path.join(
            os.path.dirname(input_path),
            os.path.splitext(os.path.basename(input_path))[0] + args.name_notes
        )
    else:
        print(f"路徑不存在：{input_path}")
        return [], [], [], ""

    os.makedirs(save_dir, exist_ok=True)

    target_track_id       = None
    consecutive_lost_count = 0
    results_list          = []   # 每幀的預測類別 index（無法偵測為 nan）
    frame_indices         = []   # 每幀的 frame ID
    all_probs_list        = []   # 每幀的 softmax 機率向量
    frame_count           = 0

    last_coords_478 = None
    last_coords_id = 0

    while True:
        # ── 1. 取得影像 ──────────────────────────────────────────
        if is_video:
            ret, img_bgr = cap.read()
            if not ret:
                break
            frame_name      = f"frame_{frame_count:05d}.jpg"
            current_frame_id = frame_count + 1
        else:
            if frame_count >= len(image_paths):
                break
            img_path        = image_paths[frame_count]
            frame_name      = os.path.basename(img_path)
            extracted       = extract_frame_number(frame_name)
            current_frame_id = extracted if extracted != -1 else (frame_count + 1)
            img_bgr         = cv2.imread(img_path)

        frame_indices.append(current_frame_id)

        if img_bgr is None:
            results_list.append(np.nan)
            all_probs_list.append(np.full(args.num_classes, np.nan))
            frame_count += 1
            continue

        h_img, w_img = img_bgr.shape[:2]

        # ── 2. YOLO 追蹤 ─────────────────────────────────────────
        track_results = face_model.track(img_bgr, persist=True,
                                          verbose=False, tracker="bytetrack.yaml")
        boxes = track_results[0].boxes

        current_ids = []
        if boxes.id is not None:
            current_ids = boxes.id.cpu().numpy().astype(int).tolist()

        # 目標 ID 是否仍在畫面中
        if target_track_id is not None and target_track_id in current_ids:
            consecutive_lost_count = 0
        else:
            if target_track_id is not None:
                consecutive_lost_count += 1

            new_best_id = get_best_face_id(boxes, img_bgr.shape)
            if new_best_id is not None:
                if target_track_id is not None:
                    print(f"  目標切換：ID {target_track_id} → {new_best_id}")
                else:
                    print(f"  鎖定目標 ID：{new_best_id}")
                target_track_id        = new_best_id
                consecutive_lost_count = 0

        # ── 3. 人臉裁切 ──────────────────────────────────────────
        found_target = False
        face_crop    = None
        draw_coord   = None

        if target_track_id is not None and target_track_id in current_ids:
            idx         = current_ids.index(target_track_id)
            box         = boxes.xyxy[idx]
            x1, y1, x2, y2 = map(int, box)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w_b, h_b = x2 - x1, y2 - y1
            nx1 = max(0, int(cx - w_b * 0.6))
            ny1 = max(0, int(cy - h_b * 0.6))
            nx2 = min(w_img, int(cx + w_b * 0.6))
            ny2 = min(h_img, int(cy + h_b * 0.6))

            face_crop  = img_bgr[ny1:ny2, nx1:nx2]
            draw_coord = (nx1, ny1, nx2, ny2)
            found_target = True

        # ── 4. MediaPipe + 감정 모델 추론 ────────────────────────
        if found_target and face_crop is not None and face_crop.size > 0:

            # 4-a. MediaPipe landmarks 偵測
            coords_478, face_found = detect_landmarks(mp_detector, face_crop, target_size=224, last_coords_478=last_coords_478)

            if face_found:
                last_coords_478 = coords_478
                last_coords_id = current_frame_id
            
                # 4-b. 準備模型輸入
                face_rgb      = face_crop[:, :, ::-1]          # BGR → RGB
                face_224      = cv2.resize(face_crop, (224, 224))
                input_tensor  = transform(face_rgb).unsqueeze(0).to(device)
                coords_tensor = torch.tensor(coords_478,
                                              dtype=torch.float32).unsqueeze(0).to(device)  # [1, 478, 3]

                # 4-c. 模型推論
                with torch.no_grad():
                    outputs, _ = model(input_tensor, coords_tensor)
                    probs  = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred   = torch.max(outputs, 1)[1].item()

                    # ── 新增：只篩選出模型真正使用的 49 或 196 個點 ──
                    if isinstance(model, torch.nn.DataParallel):
                        target_idx = model.module.face_landback.target_indices_tensor
                    else:
                        target_idx = model.face_landback.target_indices_tensor
                        
                    filtered_coords = coords_tensor[:, target_idx, :]

                results_list.append(pred)
                all_probs_list.append(probs)

                # 正常繪製 HUD
                img_draw = draw_hud(img_bgr.copy(), draw_coord, probs, pred, coords=filtered_coords, face_224=face_224, face_found=face_found, show_landmarks=True)
                cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)
            
            else:
                # ── MediaPipe 未偵測到人臉時的處理 ──
                print(f"  [警告] 第 {current_frame_id} 幀 MediaPipe 未偵測到人臉，跳過模型推論。")
                results_list.append(np.nan)
                all_probs_list.append(np.full(args.num_classes, np.nan))
                
                # 不呼叫 draw_hud，直接回傳/儲存原圖
                cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)

        else:
            # YOLO 也沒抓到人臉
            results_list.append(np.nan)
            all_probs_list.append(np.full(args.num_classes, np.nan))
            cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)

        if frame_count % 100 == 0:
            print(f"  已處理：{frame_name}  (Frame ID: {current_frame_id})")

        frame_count += 1

    if is_video:
        cap.release()

    return frame_indices, results_list, np.array(all_probs_list), save_dir


# ══════════════════════════════════════════════════════════════════
#  結果儲存（CSV + 圖表）
# ══════════════════════════════════════════════════════════════════
def plot_results(x, y, probs_matrix, save_dir, num_classes):
    # ── CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, 'result_data.csv')
    labels   = EMOTION_LABELS[:num_classes]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_ID', 'Classification', 'Prediction', 'Face_Detected'] + labels)

        for i in range(len(x)):
            pred_idx = y[i]
            if np.isnan(pred_idx):
                pred_label   = 'NaN'
                face_detected = False
            else:
                pred_label   = labels[int(pred_idx)]
                face_detected = True

            row = [x[i], int(pred_idx) if not np.isnan(pred_idx) else -1, pred_label, face_detected] + probs_matrix[i].tolist()
            writer.writerow(row)

    print(f"CSV 儲存完成 → {csv_path}")

    # ── 圖表 ─────────────────────────────────────────────────────
    save_path = os.path.join(save_dir, 'result_graph.png')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # Subplot 1：預測類別
    # Matplotlib 預設在遇到 NaN 時會自動斷開連線，這正好符合我們的需求！
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3,
             linewidth=1.5, color='black', label='Prediction')
    ax1.set_title("Frame-by-Frame Classification Result", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Emotion", fontsize=12)
    ax1.set_yticks(range(num_classes))
    ax1.set_yticklabels(labels)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, num_classes - 0.5)

    # Subplot 2：각 클래스 신뢰도
    for class_idx in range(num_classes):
        ax2.plot(x, probs_matrix[:, class_idx],
                 label=labels[class_idx],
                 color=COLORS_PLT[class_idx % len(COLORS_PLT)],
                 linewidth=1.5, alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame ID", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    if len(x) > 0:
        valid_x = [xi for xi in x if not (isinstance(xi, float) and np.isnan(xi))]
        if valid_x:
            ax1.set_xlim(min(valid_x) - 1, max(valid_x) + 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"圖表儲存完成 → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  進入點
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # ── 模型初始化 ──
    mp_detector = build_mediapipe_detector(args.mediapipe_model_path)
    model       = load_emotion_model(args, device)
    transform   = get_transforms()

    # ── 推論 ──
    if not os.path.exists(args.image_folder):
        print(f"路徑不存在：{args.image_folder}")
    else:
        frames, preds, probs, sdir = process_video_or_folder(
            args, model, device, transform, mp_detector
        )
        if len(frames) > 0:
            plot_results(frames, preds, probs, sdir, args.num_classes)
            print(f"\n完成！結果儲存於：{sdir}")
        else:
            print("未處理任何幀，請確認輸入路徑與資料內容。")