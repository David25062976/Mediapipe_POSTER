"""
test_single_confbox_mediapipe_pt.py

架構參考 test_single_confbox_shopping.py，
模型改為 emotion_hyp_affect_mediapipe_pt.py（train_affect_mediapipe_pt.py 訓練出的版本）。

主要差異：
  - 舊版 (shopping)   : model(input_tensor)            → 只需要影像
  - 本版 (mediapipe)  : model(input_tensor, coords_478) → 需要影像 + 即時 MediaPipe 偵測

新增功能：
  - 同時使用時序特徵分析模型 (pyramid_trans_expr_window_pt) 進行偵測
  - HUD 同時顯示兩個模型的機率列表（左欄：單幀 / 右欄：時序）
  - 影片結束後統計兩個模型各自偵測最多次的類別，輸出影片層級分類結果

流程：
  YOLO 偵測/追蹤人臉
  → 裁切人臉 (224×224)
  → MediaPipe 偵測人臉 478 個 landmarks
  → 組成 coords_478 tensor [1, 478, 3]
  → [單幀模型] model(image_tensor, coords_478_tensor)   → 每幀預測
  → [時序模型] model(window_tensor, coords_window)      → 每幀預測（buffer 滿後）
  → 累積計票 → 影片層級分類結果

使用範例
─────────────────────────────────────────────────────────
  python test_single_confbox_mediapipe_pt.py \
      --image_folder     /path/to/images_or_video \
      --checkpoint       ./checkpoint/20240101-120000/best.pth \
      --temporal_checkpoint  ./checkpoint/temporal/best.pth \
      --temporal_window_size 8 \
      --temporal_type    MLP \
      --mediapipe_points    196 \
      --mediapipe_patch_size 24 \
      --num_classes   8 \
      --modeltype     large \
      --gpu           0
─────────────────────────────────────────────────────────
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
from collections import OrderedDict, Counter, deque
from natsort import natsorted

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
from models.emotion_hyp_affect_mediapipe_pt import pyramid_trans_expr
from models.emotion_hyp_window_pt import pyramid_trans_expr_window_pt

# ══════════════════════════════════════════════════════════════════
#  常數（兩個模型各自獨立的類別定義）
# ══════════════════════════════════════════════════════════════════

# ── 單幀影像幀模型：8 類 ─────────────────────────────────────────
SINGLE_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise',
                 'interesting', 'thinking', 'buying', 'passing']

SINGLE_COLORS_BGR = [
    (128, 128, 128),  # Neutral     - 灰
    (0, 165, 255),    # Happy       - 橘
    (255, 0, 0),      # Sad         - 藍
    (128, 0, 128),    # Surprise    - 紫
    (0, 128, 0),      # interesting - 綠
    (255, 255, 0),    # thinking    - 青
    (0, 0, 255),      # buying      - 紅
    (19, 69, 139),    # passing     - 棕
]

SINGLE_COLORS_PLT = ['gray', 'orange', 'blue', 'purple',
                     'green', 'cyan', 'red', 'brown']

# ── 時序特徵分析模型：5 類 ───────────────────────────────────────
TEMPORAL_LABELS = ['browsing', 'interesting', 'thinking', 'buying', 'passing']

TEMPORAL_COLORS_BGR = [
    (0, 215, 255),    # browsing    - 金黃
    (0, 128, 0),      # interesting - 綠
    (255, 255, 0),    # thinking    - 青
    (0, 0, 255),      # buying      - 紅
    (19, 69, 139),    # passing     - 棕
]

TEMPORAL_COLORS_PLT = ['gold', 'green', 'cyan', 'red', 'brown']


# ══════════════════════════════════════════════════════════════════
#  引數
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Mediapipe-PT Emotion Model (Single + Temporal)')

    parser.add_argument('--image_folder',         type=str,
                        default='/home/Dataset/customer_ori2/out/let s go/let s go-3',
                        help='影像資料夾路徑 或 影片檔案路徑')
    parser.add_argument('--name_notes',           type=str, default='_result',
                        help='result 資料夾的名稱備註')

    # ── 單幀模型 ──
    parser.add_argument('--checkpoint',           type=str,
                        default='./checkpoint/20260501-060739_Shopping_196_14/best.pth',
                        help='單幀模型 .pth checkpoint 路徑')

    # ── 時序模型 ──
    parser.add_argument('--temporal_checkpoint',  type=str,
                        default='./checkpoint/window_20260521-120629_w8s8/best.pth',
                        help='時序特徵分析模型 .pth checkpoint 路徑（留空則不啟用）')
    parser.add_argument('--temporal_window_size', type=int, default=8,
                        help='時序特徵模型的滑動視窗大小（幀數）')
    parser.add_argument('--temporal_type',        type=str, default='MLP',
                        choices=['MLP', 'CNN', 'Transformer'],
                        help='時序特徵模型的時序融合方式')

    # ── 共用模型超參數（需與訓練時完全一致）──
    # num_classes 由各自的 label 列表長度自動決定：
    #   單幀模型  = len(SINGLE_LABELS)  = 8
    #   時序模型  = len(TEMPORAL_LABELS) = 5
    parser.add_argument('--mediapipe_points',     type=int, default=49,
                        choices=[49, 196],
                        help='訓練時使用的 mediapipe_points (49 or 196)')
    parser.add_argument('--mediapipe_patch_size', type=int, default=14,
                        choices=[14, 24],
                        help='訓練時使用的 mediapipe_patch_size (14 or 24)')
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
            lm.z,                 # depth（歸一化）
        ]
    return coords_478, True


# ══════════════════════════════════════════════════════════════════
#  單幀情緒模型載入
# ══════════════════════════════════════════════════════════════════
def load_emotion_model(args, device):
    print(f"載入單幀情緒模型 pyramid_trans_expr "
          f"(type={args.modeltype}, points={args.mediapipe_points}, "
          f"patch_size={args.mediapipe_patch_size})...")

    model = pyramid_trans_expr(
        img_size=224,
        num_classes=len(SINGLE_LABELS),
        type=args.modeltype,
        mediapipe_points=args.mediapipe_points,
        mediapipe_patch_size=args.mediapipe_patch_size,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("單幀模型載入完成。")
    return model


# ══════════════════════════════════════════════════════════════════
#  時序特徵分析模型載入
# ══════════════════════════════════════════════════════════════════
def load_temporal_model(args, device):
    """
    載入 pyramid_trans_expr_window_pt 時序模型。
    共用 mediapipe_points / mediapipe_patch_size / modeltype / num_classes。
    """
    print(f"\n載入時序特徵分析模型 pyramid_trans_expr_window_pt "
          f"(type={args.modeltype}, points={args.mediapipe_points}, "
          f"patch_size={args.mediapipe_patch_size}, "
          f"window_size={args.temporal_window_size}, "
          f"temporal_type={args.temporal_type})...")

    model = pyramid_trans_expr_window_pt(
        img_size=224,
        num_classes=len(TEMPORAL_LABELS),
        window_size=args.temporal_window_size,
        type=args.modeltype,
        freeze_list=[False, False, False],   # 推理時不需要凍結
        mediapipe_points=args.mediapipe_points,
        mediapipe_patch_size=args.mediapipe_patch_size,
        temporal_type=args.temporal_type,
    )

    ckpt = torch.load(args.temporal_checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("時序模型載入完成。")
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
    """從檔名提取數字，例如 \"01130.jpg\" -> 1130"""
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


def draw_hud(frame, box, probs, predicted_idx,
             coords=None, face_224=None, face_found=False, show_landmarks=True,
             temporal_probs=None, temporal_pred=None, window_size=8):
    """
    在影像上繪製偵測框 + 浮動式 HUD 機率列表。

    左欄：單幀模型機率  (always shown when probs is valid)
    右欄：時序模型機率  (only shown when temporal_probs is not None)
    左上角：face_224 子畫面（含 landmarks）

    Parameters
    ----------
    temporal_probs  : np.ndarray or None  時序模型 softmax 機率向量
    temporal_pred   : int or None         時序模型預測類別 index
    window_size     : int                 時序模型視窗大小（標題顯示用）
    """
    x1, y1, x2, y2 = box
    main_color = SINGLE_COLORS_BGR[predicted_idx]
    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)

    h, w, _ = frame.shape

    # ── 計算 HUD 面板數量與總寬 ──────────────────────────────────
    use_temporal = (temporal_probs is not None and temporal_pred is not None)
    panel_w   = 200          # 每個機率面板的寬度
    panel_gap = 5            # 兩個面板之間的間距
    n_panels  = 2 if use_temporal else 1
    total_w   = panel_w * n_panels + panel_gap * (n_panels - 1)

    # ── 決定 HUD 起始座標（優先放在偵測框右側，若超出則放左側）──
    if x2 + total_w + 15 < w:
        hud_x = x2 + 10
    else:
        hud_x = max(0, x1 - total_w - 10)

    # 面板高度以類別數較多的模型為準，確保兩欄等高
    n_single   = len(SINGLE_LABELS)
    n_temporal = len(TEMPORAL_LABELS)
    n_max      = n_single if not use_temporal else max(n_single, n_temporal)
    panel_h    = n_max * 25 + 30   # 每行 25px + 標題行 30px
    hud_y      = max(0, min(y1, h - panel_h - 5))

    # ── 繪製單幀模型面板 ─────────────────────────────────────────
    _draw_prob_panel(
        frame, hud_x, hud_y, panel_w, panel_h,
        probs, predicted_idx,
        labels=SINGLE_LABELS, colors_bgr=SINGLE_COLORS_BGR,
        title="Single Frame"
    )

    # ── 繪製時序模型面板 ─────────────────────────────────────────
    if use_temporal:
        temp_x = hud_x + panel_w + panel_gap
        _draw_prob_panel(
            frame, temp_x, hud_y, panel_w, panel_h,
            temporal_probs, temporal_pred,
            labels=TEMPORAL_LABELS, colors_bgr=TEMPORAL_COLORS_BGR,
            title=f"Temporal (W={window_size})"
        )

    # ── 在 face_224 上畫特徵點，並疊加至左上角 ──────────────────
    if show_landmarks and coords is not None and face_224 is not None:
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()

        if coords.ndim == 3:
            coords = coords.squeeze(0)

        for pt in coords:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < 224 and 0 <= py < 224:
                cv2.circle(face_224, (px, py), radius=1, color=(255, 255, 255), thickness=-1)

        pip_h, pip_w = face_224.shape[:2]
        if h >= pip_h and w >= pip_w:
            frame[0:pip_h, 0:pip_w] = face_224
            border_color = (255, 255, 255) if face_found else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (pip_w, pip_h), border_color, 2)

    return frame


def _draw_prob_panel(frame, hud_x, hud_y, panel_w, panel_h, probs, pred_idx,
                     labels, colors_bgr, title=""):
    """
    在 frame 上繪製單個半透明機率列表面板。

    Parameters
    ----------
    labels     : 該模型的類別名稱列表
    colors_bgr : 對應各類別的 BGR 顏色列表
    title      : 顯示在面板頂部的標題文字
    """
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (hud_x, hud_y),
                  (hud_x + panel_w, hud_y + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 面板標題
    cv2.putText(frame, title,
                (hud_x + 5, hud_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 各類別機率
    for i, (label, prob) in enumerate(zip(labels, probs)):
        text_y     = hud_y + 30 + i * 25
        color      = colors_bgr[i] if i == pred_idx else (255, 255, 255)
        thickness  = 2 if i == pred_idx else 1
        font_scale = 0.6 if i == pred_idx else 0.48

        if i == pred_idx:
            cv2.putText(frame, ">",
                        (hud_x + 4, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        cv2.putText(frame, f"{label}: {prob:.2f}",
                    (hud_x + 22, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ══════════════════════════════════════════════════════════════════
#  主推理循環
# ══════════════════════════════════════════════════════════════════
def process_video_or_folder(args, model, device, transform, mp_detector,
                             temporal_model=None):
    """
    處理影片或影像資料夾，同時使用單幀模型與時序模型進行推理。

    Returns
    -------
    frame_indices       : list[int]       每幀的 Frame ID
    results_list        : list[float]     單幀模型預測類別 index（無法偵測為 nan）
    all_probs_list      : np.ndarray      單幀模型 softmax 機率向量矩陣
    temporal_res_list   : list[float]     時序模型預測類別 index（buffer 未滿或無法偵測為 nan）
    temporal_probs_list : np.ndarray      時序模型 softmax 機率向量矩陣
    save_dir            : str             結果儲存路徑
    """
    # ── YOLO 載入 ──
    try:
        face_model = YOLO(args.yolo_model)
    except Exception:
        print("指定的 YOLO 模型載入失敗，改用 yolov8n.pt。")
        face_model = YOLO('yolov8n.pt')

    input_path  = args.image_folder
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
        return [], [], [], [], [], ""

    os.makedirs(save_dir, exist_ok=True)

    # ── 狀態初始化 ──────────────────────────────────────────────
    target_track_id        = None
    consecutive_lost_count = 0
    results_list           = []   # 單幀：每幀預測類別 index
    frame_indices          = []
    all_probs_list         = []   # 單幀：softmax 機率向量
    temporal_res_list      = []   # 時序：每幀預測類別 index
    temporal_probs_list    = []   # 時序：softmax 機率向量
    frame_count            = 0

    last_coords_478 = None
    last_coords_id  = 0

    # ── 時序滑動視窗 buffer ──────────────────────────────────────
    use_temporal      = (temporal_model is not None)
    window_size       = args.temporal_window_size
    frame_buf         = deque(maxlen=window_size)   # 存放 transformed tensor [3,224,224]
    coords_buf        = deque(maxlen=window_size)   # 存放 raw coords_478 [478,3]

    # ============================================================
    while True:
        # ── 1. 取得影像 ─────────────────────────────────────────
        if is_video:
            ret, img_bgr = cap.read()
            if not ret:
                break
            frame_name       = f"frame_{frame_count:05d}.jpg"
            current_frame_id = frame_count + 1
        else:
            if frame_count >= len(image_paths):
                break
            img_path         = image_paths[frame_count]
            frame_name       = os.path.basename(img_path)
            extracted        = extract_frame_number(frame_name)
            current_frame_id = extracted if extracted != -1 else (frame_count + 1)
            img_bgr          = cv2.imread(img_path)

        frame_indices.append(current_frame_id)

        if img_bgr is None:
            results_list.append(np.nan)
            all_probs_list.append(np.full(len(SINGLE_LABELS), np.nan))
            temporal_res_list.append(np.nan)
            temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))
            frame_count += 1
            continue

        h_img, w_img = img_bgr.shape[:2]

        # ── 2. YOLO 追蹤 ────────────────────────────────────────
        track_results = face_model.track(img_bgr, persist=True,
                                          verbose=False, tracker="bytetrack.yaml")
        boxes = track_results[0].boxes

        current_ids = []
        if boxes.id is not None:
            current_ids = boxes.id.cpu().numpy().astype(int).tolist()

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

        # ── 3. 人臉裁切 ─────────────────────────────────────────
        found_target = False
        face_crop    = None
        draw_coord   = None

        if target_track_id is not None and target_track_id in current_ids:
            idx             = current_ids.index(target_track_id)
            box             = boxes.xyxy[idx]
            x1, y1, x2, y2 = map(int, box)

            cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
            w_b, h_b = x2 - x1, y2 - y1
            nx1 = max(0, int(cx - w_b * 0.6))
            ny1 = max(0, int(cy - h_b * 0.6))
            nx2 = min(w_img, int(cx + w_b * 0.6))
            ny2 = min(h_img, int(cy + h_b * 0.6))

            face_crop    = img_bgr[ny1:ny2, nx1:nx2]
            draw_coord   = (nx1, ny1, nx2, ny2)
            found_target = True

        # ── 4. MediaPipe + 雙模型推理 ───────────────────────────
        if found_target and face_crop is not None and face_crop.size > 0:

            # 4-a. MediaPipe landmarks
            coords_478, face_found = detect_landmarks(
                mp_detector, face_crop,
                target_size=224,
                last_coords_478=last_coords_478
            )

            if face_found:
                last_coords_478 = coords_478
                last_coords_id  = current_frame_id
            else:
                print(f"  [警告] 第 {current_frame_id} 幀 MediaPipe 未偵測到人臉，"
                      f"使用第 {last_coords_id} 幀座標。")

            # 4-b. 準備共用輸入
            face_rgb     = face_crop[:, :, ::-1]           # BGR → RGB
            face_224     = cv2.resize(face_crop, (224, 224))
            input_tensor = transform(face_rgb).unsqueeze(0).to(device)  # [1, 3, 224, 224]
            coords_tensor = torch.tensor(
                coords_478, dtype=torch.float32
            ).unsqueeze(0).to(device)                      # [1, 478, 3]

            # ── 4-c. 單幀模型推理 ──────────────────────────────
            with torch.no_grad():
                outputs, _ = model(input_tensor, coords_tensor)
                probs      = F.softmax(outputs, dim=1).cpu().numpy()[0]
                pred       = torch.max(outputs, 1)[1].item()

                # 取得篩選後的 landmark 座標（供 HUD 顯示）
                if isinstance(model, torch.nn.DataParallel):
                    target_idx = model.module.face_landback.target_indices_tensor
                else:
                    target_idx = model.face_landback.target_indices_tensor
                filtered_coords = coords_tensor[:, target_idx, :]

            results_list.append(pred)
            all_probs_list.append(probs)

            # ── 4-d. 時序模型推理 ──────────────────────────────
            # 將當前幀放入滑動視窗 buffer
            temporal_probs_cur = None
            temporal_pred_cur  = None

            if use_temporal:
                # 儲存當前幀的 tensor（已在 CPU，節省 GPU 記憶體）
                frame_buf.append(input_tensor.squeeze(0).cpu())
                coords_buf.append(coords_tensor.squeeze(0).cpu())   # [478, 3]

                if len(frame_buf) == window_size:
                    # 組合視窗：x shape → [1, 3, W, 224, 224]
                    x_window = torch.stack(list(frame_buf), dim=0)        # [W, 3, 224, 224]
                    x_window = x_window.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, W, 224, 224]
                    x_window = x_window.to(device)

                    # coords shape → [1, W, 478, 3]
                    c_window = torch.stack(list(coords_buf), dim=0)       # [W, 478, 3]
                    c_window = c_window.unsqueeze(0).to(device)           # [1, W, 478, 3]

                    with torch.no_grad():
                        t_outputs, _ = temporal_model(x_window, c_window)
                        temporal_probs_cur = F.softmax(t_outputs, dim=1).cpu().numpy()[0]
                        temporal_pred_cur  = torch.max(t_outputs, 1)[1].item()

                    temporal_res_list.append(temporal_pred_cur)
                    temporal_probs_list.append(temporal_probs_cur)
                else:
                    # buffer 尚未填滿，無法預測
                    temporal_res_list.append(np.nan)
                    temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))
            else:
                temporal_res_list.append(np.nan)
                temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))

            # ── 4-e. 繪製結果並儲存 ────────────────────────────
            img_draw = draw_hud(
                img_bgr.copy(),
                draw_coord, probs, pred,
                coords=filtered_coords, face_224=face_224,
                face_found=face_found, show_landmarks=True,
                temporal_probs=temporal_probs_cur,
                temporal_pred=temporal_pred_cur,
                window_size=window_size,
            )
            cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)

        else:
            # 未偵測到人臉
            results_list.append(np.nan)
            all_probs_list.append(np.full(len(SINGLE_LABELS), np.nan))
            temporal_res_list.append(np.nan)
            temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))

            # 仍將空幀推入 buffer（以 NaN 向量佔位，不破壞時序連續性）
            # 策略：跳過本幀，不推入 buffer，讓視窗保持有效幀
            cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)

        if frame_count % 100 == 0:
            print(f"  已處理：{frame_name}  (Frame ID: {current_frame_id})")

        frame_count += 1

    if is_video:
        cap.release()

    return (frame_indices,
            results_list,
            np.array(all_probs_list),
            temporal_res_list,
            np.array(temporal_probs_list),
            save_dir)


# ══════════════════════════════════════════════════════════════════
#  結果儲存（CSV + 圖表）
# ══════════════════════════════════════════════════════════════════
def plot_results(x, y, probs_matrix, save_dir,
                 temporal_y=None, temporal_probs_matrix=None):
    """
    儲存 CSV 及結果圖表。

    如果提供 temporal_y / temporal_probs_matrix，
    則圖表增加時序模型的預測曲線與機率曲線。
    """
    use_temporal   = (temporal_y is not None and temporal_probs_matrix is not None)
    single_labels  = SINGLE_LABELS
    single_n       = len(single_labels)
    temporal_labels = TEMPORAL_LABELS
    temporal_n      = len(temporal_labels)

    # ── CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, 'result_data.csv')
    single_prob_header   = [f"Single_{l}" for l in single_labels]
    temporal_prob_header = [f"Temporal_{l}" for l in temporal_labels] if use_temporal else []

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['Frame_ID', 'Single_Prediction', 'Face_Detected']
            + single_prob_header
            + (['Temporal_Prediction'] + temporal_prob_header if use_temporal else [])
        )

        for i in range(len(x)):
            pred_idx = y[i]
            if np.isnan(pred_idx):
                pred_label    = 'NaN'
                face_detected = False
            else:
                pred_label    = single_labels[int(pred_idx)]
                face_detected = True

            row = [x[i], pred_label, face_detected] + probs_matrix[i].tolist()

            if use_temporal:
                t_pred_idx = temporal_y[i]
                t_label    = 'NaN' if np.isnan(t_pred_idx) else temporal_labels[int(t_pred_idx)]
                row += [t_label] + temporal_probs_matrix[i].tolist()

            writer.writerow(row)

    print(f"CSV 儲存完成 → {csv_path}")

    # ── 圖表 ────────────────────────────────────────────────────
    save_path = os.path.join(save_dir, 'result_graph.png')
    n_rows    = 4 if use_temporal else 2
    height_ratios = [1, 2, 1, 2] if use_temporal else [1, 2]

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(18, 6 * n_rows), sharex=True,
        gridspec_kw={'height_ratios': height_ratios}
    )
    axes = list(axes)

    # ── Subplot 1：單幀模型預測類別 ─────────────────────────────
    ax1 = axes[0]
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3,
             linewidth=1.5, color='steelblue', label='Single Frame')
    ax1.set_title("Single Frame Model - Classification Result", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Emotion", fontsize=12)
    ax1.set_yticks(range(single_n))
    ax1.set_yticklabels(single_labels)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, single_n - 0.5)

    # ── Subplot 2：單幀模型各類別信心度 ─────────────────────────
    ax2 = axes[1]
    for class_idx in range(single_n):
        ax2.plot(x, probs_matrix[:, class_idx],
                 label=single_labels[class_idx],
                 color=SINGLE_COLORS_PLT[class_idx % len(SINGLE_COLORS_PLT)],
                 linewidth=1.5, alpha=0.8)
    ax2.set_title("Single Frame Model - Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    if use_temporal:
        # ── Subplot 3：時序模型預測類別 ─────────────────────────
        ax3 = axes[2]
        ax3.plot(x, temporal_y, linestyle='-', marker='s', markersize=3,
                 linewidth=1.5, color='darkorange', label='Temporal')
        ax3.set_title("Temporal Model - Classification Result", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Emotion", fontsize=12)
        ax3.set_yticks(range(temporal_n))
        ax3.set_yticklabels(temporal_labels)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax3.set_ylim(-0.5, temporal_n - 0.5)

        # ── Subplot 4：時序模型各類別信心度 ─────────────────────
        ax4 = axes[3]
        for class_idx in range(temporal_n):
            ax4.plot(x, temporal_probs_matrix[:, class_idx],
                     label=temporal_labels[class_idx],
                     color=TEMPORAL_COLORS_PLT[class_idx % len(TEMPORAL_COLORS_PLT)],
                     linewidth=1.5, alpha=0.8, linestyle='--')
        ax4.set_title("Temporal Model - Confidence Probability per Class", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Frame ID", fontsize=12)
        ax4.set_ylabel("Probability", fontsize=12)
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, linestyle=':', alpha=0.3)
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")
    else:
        axes[1].set_xlabel("Frame ID", fontsize=12)

    if len(x) > 0:
        valid_x = [xi for xi in x if not (isinstance(xi, float) and np.isnan(xi))]
        if valid_x:
            axes[0].set_xlim(min(valid_x) - 1, max(valid_x) + 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"圖表儲存完成 → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  影片層級分類統計
# ══════════════════════════════════════════════════════════════════
def print_video_classification(y_single, y_temporal, save_dir):
    """
    統計整支影片中，兩個模型各自偵測最多次的類別，
    輸出至 console 並儲存為 video_classification.txt。
    """
    lines = []
    lines.append("=" * 55)
    lines.append("  影片層級分類結果（Video-Level Classification）")
    lines.append("=" * 55)

    # ── 單幀模型統計 ────────────────────────────────────────────
    valid_single = [int(p) for p in y_single if not (isinstance(p, float) and np.isnan(p))]
    if valid_single:
        counter          = Counter(valid_single)
        total            = len(valid_single)
        top_idx, top_cnt = counter.most_common(1)[0]
        lines.append(f"\n[單幀影像幀模型]  有效幀：{total}")
        lines.append(f"  最終分類：{SINGLE_LABELS[top_idx]}  (出現 {top_cnt} 次 / {top_cnt/total*100:.1f}%)")
        lines.append("  各類別票數：")
        for cls_idx, lbl in enumerate(SINGLE_LABELS):
            cnt = counter.get(cls_idx, 0)
            bar = "█" * int(cnt / total * 30)
            lines.append(f"    {lbl:12s} {cnt:5d}  {bar}")
    else:
        lines.append("\n[單幀影像幀模型]  無有效預測。")

    # ── 時序模型統計 ────────────────────────────────────────────
    valid_temporal = [int(p) for p in y_temporal if not (isinstance(p, float) and np.isnan(p))]
    if valid_temporal:
        counter          = Counter(valid_temporal)
        total            = len(valid_temporal)
        top_idx, top_cnt = counter.most_common(1)[0]
        lines.append(f"\n[時序特徵分析模型] 有效幀：{total}")
        lines.append(f"  最終分類：{TEMPORAL_LABELS[top_idx]}  (出現 {top_cnt} 次 / {top_cnt/total*100:.1f}%)")
        lines.append("  各類別票數：")
        for cls_idx, lbl in enumerate(TEMPORAL_LABELS):
            cnt = counter.get(cls_idx, 0)
            bar = "█" * int(cnt / total * 30)
            lines.append(f"    {lbl:12s} {cnt:5d}  {bar}")
    else:
        lines.append("\n[時序特徵分析模型] 無有效預測（可能未啟用時序模型）。")

    lines.append("=" * 55)

    report = "\n".join(lines)
    print(report)

    txt_path = os.path.join(save_dir, 'video_classification.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n分類報告儲存完成 → {txt_path}")


# ══════════════════════════════════════════════════════════════════
#  進入點
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    # ── 模型初始化 ──────────────────────────────────────────────
    mp_detector    = build_mediapipe_detector(args.mediapipe_model_path)
    model          = load_emotion_model(args, device)
    transform      = get_transforms()

    temporal_model = None
    if args.temporal_checkpoint:
        temporal_model = load_temporal_model(args, device)
    else:
        print("\n未指定 --temporal_checkpoint，時序模型不啟用。")

    # ── 推理 ────────────────────────────────────────────────────
    if not os.path.exists(args.image_folder):
        print(f"路徑不存在：{args.image_folder}")
    else:
        (frames, preds, probs,
         temp_preds, temp_probs, sdir) = process_video_or_folder(
            args, model, device, transform, mp_detector,
            temporal_model=temporal_model
        )

        if len(frames) > 0:
            plot_results(
                frames, preds, probs, sdir,
                temporal_y=temp_preds if temporal_model else None,
                temporal_probs_matrix=temp_probs if temporal_model else None,
            )
            print_video_classification(preds, temp_preds, sdir)
            print(f"\n完成！結果儲存於：{sdir}")
        else:
            print("未處理任何幀，請確認輸入路徑與資料內容。")
