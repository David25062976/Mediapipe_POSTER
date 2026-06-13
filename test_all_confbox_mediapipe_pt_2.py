"""
test_single_confbox_mediapipe_pt.py

模型：
  - 單幀模型  pyramid_trans_expr          (8 類)
  - 時序模型  pyramid_trans_expr_window_pt (5 類)

執行模式
─────────────────────────────────────────────────
A. 單一資料夾 / 影片模式（原有功能）
   --image_folder /path/to/frames_or_video

B. 批次評估模式（valid_list.txt）
   --eval_list  /path/to/valid_list.txt
   --eval_root  /path/to/dataset_root
   → 依 GT 標籤計算兩個模型的影片級準確度

座標來源（--coords_mode）
─────────────────────────────────────────────────
  mediapipe   : 使用 YOLO 偵測人臉 + MediaPipe 即時計算 478 點（預設）
  precomputed : 讀取預先計算好的 .pt 檔（--landmarks_db），不需 YOLO

單幀模型類別映射到 5 類空間（評估用）
  Neutral / Happy / Sad / Surprise  →  browsing
  interesting                       →  interesting
  thinking                          →  thinking
  buying                            →  buying
  passing                           →  passing

使用範例
─────────────────────────────────────────────────
  # A. 即時推理
  python test_single_confbox_mediapipe_pt.py \
      --image_folder /data/video_frames \
      --checkpoint ./ckpt/single/best.pth \
      --temporal_checkpoint ./ckpt/temporal/best.pth \
      --coords_mode mediapipe

  # B. 批次評估（使用預先計算座標）
  python test_single_confbox_mediapipe_pt.py \
      --eval_list  /data/valid_list.txt \
      --eval_root  /data \
      --checkpoint ./ckpt/single/best.pth \
      --temporal_checkpoint ./ckpt/temporal/best.pth \
      --coords_mode precomputed \
      --landmarks_db /data/landmarks.pt
─────────────────────────────────────────────────
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
#  常數
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

# ── 單幀 8 類 → 時序 5 類 映射（評估用）────────────────────────
# Neutral(0) / Happy(1) / Sad(2) / Surprise(3) → browsing(0)
# interesting(4) → interesting(1)
# thinking(5)    → thinking(2)
# buying(6)      → buying(3)
# passing(7)     → passing(4)
SINGLE_TO_TEMPORAL = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4}


# ══════════════════════════════════════════════════════════════════
#  引數
# ══════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference with Mediapipe-PT Emotion Model (Single + Temporal)')

    # ── 執行模式 ──────────────────────────────────────────────────
    parser.add_argument('--image_folder', type=str, default='',
                        help='[模式 A] 影像資料夾路徑 或 影片檔案路徑')
    parser.add_argument('--eval_list',   type=str, default='./data/5_classes/train_list.txt',
                        help='[模式 B] valid_list.txt 路徑（留空則用模式 A）')
    parser.add_argument('--eval_root',   type=str, default='./data/5_classes',
                        help='[模式 B] eval_list 中影片資料夾的根目錄')
    parser.add_argument('--name_notes',  type=str, default='_result',
                        help='[模式 A] result 資料夾的名稱備註')

    # ── 座標來源 ──────────────────────────────────────────────────
    parser.add_argument('--coords_mode', type=str, default='precomputed',
                        choices=['mediapipe', 'precomputed'],
                        help='人臉座標來源：mediapipe=即時偵測 / precomputed=讀取 .pt 檔')
    parser.add_argument('--landmarks_db', type=str, default='./data/5_classes/train_landmarks.pt',
                        help='預先計算的 landmarks .pt 檔路徑（coords_mode=precomputed 時使用）')

    # ── 單幀模型 ──────────────────────────────────────────────────
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint/20260501-060739_Shopping_196_14/best.pth',
                        help='單幀模型 .pth checkpoint 路徑')

    # ── 時序模型 ──────────────────────────────────────────────────
    parser.add_argument('--temporal_checkpoint',  type=str, default='./checkpoint/window_20260521-120629_w8s8/best.pth',
                        help='時序特徵分析模型 .pth checkpoint 路徑（留空則不啟用）')
    parser.add_argument('--temporal_window_size', type=int, default=8,
                        help='時序特徵模型的滑動視窗大小（幀數）')
    parser.add_argument('--temporal_type',        type=str, default='MLP',
                        choices=['MLP', 'CNN', 'Transformer'],
                        help='時序特徵模型的時序融合方式')

    # ── 共用模型超參數（需與訓練時完全一致）──────────────────────
    # num_classes 自動決定：單幀=len(SINGLE_LABELS)=8 / 時序=len(TEMPORAL_LABELS)=5
    parser.add_argument('--mediapipe_points',     type=int, default=196,
                        choices=[49, 196])
    parser.add_argument('--mediapipe_patch_size', type=int, default=14,
                        choices=[14, 24])
    parser.add_argument('--modeltype', type=str, default='large',
                        choices=['small', 'base', 'large'])

    # ── MediaPipe / YOLO 模型路徑 ────────────────────────────────
    parser.add_argument('--mediapipe_model_path', type=str,
                        default='./models/face_landmarker.task')
    parser.add_argument('--yolo_model', type=str,
                        default='./data_preprocessing/head_detect_medium.pt')

    parser.add_argument('--gpu', type=str, default='1')

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
#  MediaPipe 初始化與即時偵測
# ══════════════════════════════════════════════════════════════════
def build_mediapipe_detector(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(detector, face_bgr, target_size=224, last_coords_478=None):
    """
    即時 MediaPipe 偵測，回傳 [478, 3] coords 與 face_found bool。
    x, y 為 target_size 空間的像素座標；z 為歸一化深度。
    """
    face_resized = cv2.resize(face_bgr, (target_size, target_size))
    face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(mp.ImageFormat.SRGB, np.ascontiguousarray(face_rgb))
    result       = detector.detect(mp_image)

    coords_478 = np.zeros((478, 3), dtype=np.float32)
    if not result.face_landmarks:
        return (last_coords_478 if last_coords_478 is not None else coords_478), False

    for i, lm in enumerate(result.face_landmarks[0]):
        coords_478[i] = [lm.x * target_size, lm.y * target_size, lm.z]
    return coords_478, True


# ══════════════════════════════════════════════════════════════════
#  預先計算座標的讀取
# ══════════════════════════════════════════════════════════════════
def load_landmarks_db(pt_path: str) -> dict:
    """載入 generate_landmarks.py 預先生成的 .pt 檔。"""
    print(f"載入預先計算 landmarks DB：{pt_path} ...")
    db = torch.load(pt_path, weights_only=False)
    print(f"  共 {len(db)} 筆紀錄。")
    return db


def get_precomputed_coords(landmarks_db, abs_img_path, img_root, last_coords):
    """
    從 landmarks_db 查詢該幀的 478 點座標。
    key 格式：相對路徑（以 img_root 為基準），路徑分隔符號統一為 '/'。

    回傳 (coords [478,3], face_found bool)
    """
    rel_path = os.path.relpath(abs_img_path, img_root).replace('\\', '/')
    fallback = last_coords if last_coords is not None else np.zeros((478, 3), dtype=np.float32)

    entry = landmarks_db.get(rel_path)
    if entry is None or not entry.get('face_found', False):
        return fallback, False

    coords = entry['coords']
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    return coords.astype(np.float32), True


# ══════════════════════════════════════════════════════════════════
#  模型載入
# ══════════════════════════════════════════════════════════════════
def _load_state_dict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(new_sd, strict=False)
    return model


def load_emotion_model(args, device):
    print(f"載入單幀模型 (type={args.modeltype}, "
          f"points={args.mediapipe_points}, patch={args.mediapipe_patch_size})...")
    model = pyramid_trans_expr(
        img_size=224, num_classes=len(SINGLE_LABELS),
        type=args.modeltype,
        mediapipe_points=args.mediapipe_points,
        mediapipe_patch_size=args.mediapipe_patch_size,
    )
    model = _load_state_dict(model, args.checkpoint, device)
    model = torch.nn.DataParallel(model).to(device).eval()
    print("  單幀模型載入完成。")
    return model


def load_temporal_model(args, device):
    print(f"載入時序模型 (type={args.modeltype}, "
          f"points={args.mediapipe_points}, patch={args.mediapipe_patch_size}, "
          f"window={args.temporal_window_size}, temporal_type={args.temporal_type})...")
    model = pyramid_trans_expr_window_pt(
        img_size=224, num_classes=len(TEMPORAL_LABELS),
        window_size=args.temporal_window_size,
        type=args.modeltype,
        freeze_list=[False, False, False],
        mediapipe_points=args.mediapipe_points,
        mediapipe_patch_size=args.mediapipe_patch_size,
        temporal_type=args.temporal_type,
    )
    model = _load_state_dict(model, args.temporal_checkpoint, device)
    model = torch.nn.DataParallel(model).to(device).eval()
    print("  時序模型載入完成。")
    return model


# ══════════════════════════════════════════════════════════════════
#  Transform
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
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1


def get_best_face_id(boxes, img_shape):
    if boxes.id is None:
        return None
    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2
    best_id, max_score = None, -1
    for box in boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area  = (x2 - x1) * (y2 - y1)
        dist  = (((x1+x2)/2 - cx)**2 + ((y1+y2)/2 - cy)**2)**0.5
        score = area / (dist + 1.0)
        if score > max_score:
            max_score = score
            best_id   = int(box.id.item())
    return best_id


def single_pred_to_temporal_idx(single_pred_idx: int) -> int:
    """將單幀模型的 8 類預測 index 映射至時序模型的 5 類 index。"""
    return SINGLE_TO_TEMPORAL[single_pred_idx]


# ══════════════════════════════════════════════════════════════════
#  HUD 繪製
# ══════════════════════════════════════════════════════════════════
def draw_hud(frame, box, probs, predicted_idx,
             coords=None, face_224=None, face_found=False, show_landmarks=True,
             temporal_probs=None, temporal_pred=None, window_size=8):
    """
    box: (x1,y1,x2,y2) 或 None（precomputed 模式無 YOLO 框）
    """
    frame = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_LINEAR)
    h, w, _ = frame.shape

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), SINGLE_COLORS_BGR[predicted_idx], 2)
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    use_temporal = (temporal_probs is not None and temporal_pred is not None)
    panel_w  = 200
    panel_gap = 5
    n_panels  = 2 if use_temporal else 1
    total_w   = panel_w * n_panels + panel_gap * (n_panels - 1)

    hud_x = x2 + 10 if x2 + total_w + 15 < w else max(0, x1 - total_w - 10)

    n_max   = max(len(SINGLE_LABELS), len(TEMPORAL_LABELS)) if use_temporal else len(SINGLE_LABELS)
    panel_h = n_max * 25 + 30

    if box is None:
        # 當沒有 box 時，固定顯示在右上角 (預留 10 pixel 邊距)
        hud_x = w - total_w - 10
        hud_y = 10
    else:
        # 有 box 時，依賴框的位置動態顯示在側邊
        hud_x = x2 + 10 if x2 + total_w + 15 < w else max(0, x1 - total_w - 10)
        hud_y = max(0, min(y1, h - panel_h - 5))

    _draw_prob_panel(frame, hud_x, hud_y, panel_w, panel_h,
                     probs, predicted_idx,
                     labels=SINGLE_LABELS, colors_bgr=SINGLE_COLORS_BGR,
                     title="Single Frame")

    if use_temporal:
        _draw_prob_panel(frame, hud_x + panel_w + panel_gap, hud_y, panel_w, panel_h,
                         temporal_probs, temporal_pred,
                         labels=TEMPORAL_LABELS, colors_bgr=TEMPORAL_COLORS_BGR,
                         title=f"Temporal (W={window_size})")

    if show_landmarks and coords is not None and face_224 is not None:
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
        if coords.ndim == 3:
            coords = coords.squeeze(0)
        for pt in coords:
            px, py = int(pt[0]), int(pt[1])
            if 0 <= px < 224 and 0 <= py < 224:
                cv2.circle(face_224, (px, py), 1, (255, 255, 255), -1)
        ph, pw = face_224.shape[:2]
        if h >= ph and w >= pw:
            frame[0:ph, 0:pw] = face_224
            cv2.rectangle(frame, (0, 0), (pw, ph),
                          (255, 255, 255) if face_found else (0, 0, 255), 2)

    return frame


def _draw_prob_panel(frame, hud_x, hud_y, panel_w, panel_h, probs, pred_idx,
                     labels, colors_bgr, title=""):
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y),
                  (hud_x + panel_w, hud_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (hud_x + 5, hud_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        text_y     = hud_y + 30 + i * 25
        color      = colors_bgr[i] if i == pred_idx else (255, 255, 255)
        thickness  = 2 if i == pred_idx else 1
        font_scale = 0.6 if i == pred_idx else 0.48
        if i == pred_idx:
            cv2.putText(frame, ">", (hud_x + 4, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(frame, f"{label}: {prob:.2f}", (hud_x + 22, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ══════════════════════════════════════════════════════════════════
#  核心推理函式（單一影片資料夾）
# ══════════════════════════════════════════════════════════════════
def process_one_video(args, model, device, transform,
                      image_paths, img_root, save_dir,
                      mp_detector=None, face_model=None, landmarks_db=None,
                      temporal_model=None):
    """
    處理單一影片資料夾（影像幀清單），支援兩種座標來源。

    Parameters
    ----------
    image_paths  : 已排序的絕對路徑清單
    img_root     : precomputed 模式下計算 rel_path 的根目錄
    save_dir     : 輸出影像儲存路徑
    mp_detector  : MediaPipe detector（coords_mode='mediapipe' 時使用）
    face_model   : YOLO model（coords_mode='mediapipe' 時使用）
    landmarks_db : 預先計算的 dict（coords_mode='precomputed' 時使用）
    temporal_model: 時序模型（None 則不使用）

    Returns
    -------
    frame_indices, results_list, all_probs_list,
    temporal_res_list, temporal_probs_list
    """
    os.makedirs(save_dir, exist_ok=True)

    use_mediapipe  = (args.coords_mode == 'mediapipe')
    use_temporal   = (temporal_model is not None)
    window_size    = args.temporal_window_size

    frame_indices       = []
    results_list        = []
    all_probs_list      = []
    temporal_res_list   = []
    temporal_probs_list = []

    last_coords_478     = None
    frame_buf           = deque(maxlen=window_size)
    coords_buf          = deque(maxlen=window_size)

    # mediapipe 模式的 YOLO 追蹤狀態
    target_track_id     = None

    for frame_count, img_path in enumerate(image_paths):
        frame_name       = os.path.basename(img_path)
        extracted        = extract_frame_number(frame_name)
        current_frame_id = extracted if extracted != -1 else (frame_count + 1)
        frame_indices.append(current_frame_id)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            results_list.append(np.nan)
            all_probs_list.append(np.full(len(SINGLE_LABELS), np.nan))
            temporal_res_list.append(np.nan)
            temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))
            continue

        h_img, w_img = img_bgr.shape[:2]

        # ── A. 取得 face_crop + coords_478 ──────────────────────
        if use_mediapipe:
            # A1. YOLO 追蹤
            track_results = face_model.track(img_bgr, persist=True,
                                              verbose=False, tracker="bytetrack.yaml")
            boxes       = track_results[0].boxes
            current_ids = []
            if boxes.id is not None:
                current_ids = boxes.id.cpu().numpy().astype(int).tolist()

            if target_track_id not in current_ids:
                new_id = get_best_face_id(boxes, img_bgr.shape)
                if new_id is not None:
                    target_track_id = new_id

            # A2. 人臉裁切
            face_crop  = None
            draw_coord = None
            if target_track_id is not None and target_track_id in current_ids:
                idx             = current_ids.index(target_track_id)
                x1, y1, x2, y2 = map(int, boxes.xyxy[idx])
                cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
                wb, hb   = x2 - x1, y2 - y1
                nx1 = max(0, int(cx - wb * 0.6))
                ny1 = max(0, int(cy - hb * 0.6))
                nx2 = min(w_img, int(cx + wb * 0.6))
                ny2 = min(h_img, int(cy + hb * 0.6))
                face_crop  = img_bgr[ny1:ny2, nx1:nx2]
                draw_coord = (nx1, ny1, nx2, ny2)

            if face_crop is None or face_crop.size == 0:
                results_list.append(np.nan)
                all_probs_list.append(np.full(len(SINGLE_LABELS), np.nan))
                temporal_res_list.append(np.nan)
                temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))
                cv2.imwrite(os.path.join(save_dir, frame_name), img_bgr)
                continue

            # A3. MediaPipe landmarks
            coords_478, face_found = detect_landmarks(
                mp_detector, face_crop, 224, last_coords_478)
            if face_found:
                last_coords_478 = coords_478

            face_rgb = face_crop[:, :, ::-1]
            face_224 = cv2.resize(face_crop, (224, 224))

        else:
            # B. Precomputed 模式：影像已是面部裁切，直接讀取
            coords_478, face_found = get_precomputed_coords(
                landmarks_db, img_path, img_root, last_coords_478)
            if face_found:
                last_coords_478 = coords_478
            else:
                if last_coords_478 is None:
                    last_coords_478 = coords_478  # zeros fallback

            face_resized = cv2.resize(img_bgr, (224, 224))
            face_rgb     = face_resized[:, :, ::-1]
            face_224     = face_resized.copy()
            draw_coord   = None   # 無 YOLO 框

        # ── B. 組合 tensor ──────────────────────────────────────
        input_tensor  = transform(face_rgb).unsqueeze(0).to(device)     # [1,3,224,224]
        coords_tensor = torch.tensor(
            coords_478, dtype=torch.float32).unsqueeze(0).to(device)    # [1,478,3]

        # ── C. 單幀模型推理 ──────────────────────────────────────
        with torch.no_grad():
            outputs, _ = model(input_tensor, coords_tensor)
            probs      = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred       = torch.max(outputs, 1)[1].item()

            if isinstance(model, torch.nn.DataParallel):
                target_idx = model.module.face_landback.target_indices_tensor
            else:
                target_idx = model.face_landback.target_indices_tensor
            filtered_coords = coords_tensor[:, target_idx, :]

        results_list.append(pred)
        all_probs_list.append(probs)

        # ── D. 時序模型推理 ──────────────────────────────────────
        temporal_probs_cur = None
        temporal_pred_cur  = None

        if use_temporal:
            frame_buf.append(input_tensor.squeeze(0).cpu())
            coords_buf.append(coords_tensor.squeeze(0).cpu())

            if len(frame_buf) == window_size:
                x_win = torch.stack(list(frame_buf), dim=0)           # [W,3,224,224]
                x_win = x_win.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1,3,W,224,224]
                c_win = torch.stack(list(coords_buf), dim=0).unsqueeze(0).to(device)  # [1,W,478,3]

                with torch.no_grad():
                    t_out, _          = temporal_model(x_win, c_win)
                    temporal_probs_cur = F.softmax(t_out, dim=1).cpu().numpy()[0]
                    temporal_pred_cur  = torch.max(t_out, 1)[1].item()

                temporal_res_list.append(temporal_pred_cur)
                temporal_probs_list.append(temporal_probs_cur)
            else:
                temporal_res_list.append(np.nan)
                temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))
        else:
            temporal_res_list.append(np.nan)
            temporal_probs_list.append(np.full(len(TEMPORAL_LABELS), np.nan))

        # ── E. 繪製 HUD 並儲存 ───────────────────────────────────
        img_draw = draw_hud(
            img_bgr.copy(), draw_coord, probs, pred,
            coords=filtered_coords, face_224=face_224,
            face_found=face_found, show_landmarks=True,
            temporal_probs=temporal_probs_cur,
            temporal_pred=temporal_pred_cur,
            window_size=window_size,
        )
        cv2.imwrite(os.path.join(save_dir, frame_name), img_draw)

        if frame_count % 100 == 0:
            print(f"    [{frame_count+1}/{len(image_paths)}] {frame_name}")

    return (frame_indices,
            results_list,
            np.array(all_probs_list),
            temporal_res_list,
            np.array(temporal_probs_list))


# ══════════════════════════════════════════════════════════════════
#  模式 A：單一資料夾 / 影片
# ══════════════════════════════════════════════════════════════════
def run_single_folder(args, model, temporal_model, device, transform,
                      mp_detector, face_model, landmarks_db):
    input_path = args.image_folder

    if os.path.isdir(input_path):
        files       = natsorted([f for f in os.listdir(input_path)
                                  if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])
        image_paths = [os.path.join(input_path, f) for f in files]
        save_dir    = input_path.rstrip('/') + args.name_notes
        img_root    = input_path

    elif os.path.isfile(input_path):
        # 影片檔：先解帧到暫存資料夾
        save_dir = os.path.join(
            os.path.dirname(input_path),
            os.path.splitext(os.path.basename(input_path))[0] + args.name_notes)
        frame_dir = save_dir + '_frames'
        os.makedirs(frame_dir, exist_ok=True)
        cap, cnt = cv2.VideoCapture(input_path), 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frame_dir, f"frame_{cnt:05d}.jpg"), f)
            cnt += 1
        cap.release()
        image_paths = natsorted([os.path.join(frame_dir, f)
                                  for f in os.listdir(frame_dir)
                                  if f.lower().endswith('.jpg')])
        img_root = frame_dir
    else:
        print(f"路徑不存在：{input_path}")
        return

    print(f"找到 {len(image_paths)} 張影像，開始推理...")
    frames, preds, probs, t_preds, t_probs = process_one_video(
        args, model, device, transform,
        image_paths, img_root, save_dir,
        mp_detector=mp_detector, face_model=face_model,
        landmarks_db=landmarks_db, temporal_model=temporal_model)

    if len(frames) > 0:
        plot_results(frames, preds, probs, save_dir,
                     temporal_y=t_preds if temporal_model else None,
                     temporal_probs_matrix=t_probs if temporal_model else None)
        print_video_classification(preds, t_preds, save_dir)
        print(f"\n完成！結果儲存於：{save_dir}")


# ══════════════════════════════════════════════════════════════════
#  模式 B：批次評估（valid_list.txt）
# ══════════════════════════════════════════════════════════════════
def run_eval_list(args, model, temporal_model, device, transform,
                  mp_detector, face_model, landmarks_db):
    """
    讀取 valid_list.txt，對每支影片資料夾執行推理，
    統計兩個模型的影片級分類準確度。

    GT 標籤對應（5 類空間）：
      0=browsing, 1=interesting, 2=thinking, 3=buying, 4=passing

    單幀模型預測會透過 SINGLE_TO_TEMPORAL 映射到 5 類再投票。
    """
    list_path = args.eval_list
    img_root  = args.eval_root

    # 讀取 valid_list.txt
    video_entries = []   # [(video_rel_path, gt_label), ...]
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            video_entries.append((parts[0], int(parts[1])))

    print(f"\n共 {len(video_entries)} 支影片需評估。")

    eval_results = []   # 每支影片的結果 dict

    for vi, (video_rel, gt_label) in enumerate(video_entries):
        video_dir = os.path.join(img_root, video_rel)
        if not os.path.isdir(video_dir):
            print(f"  [跳過] 資料夾不存在：{video_dir}")
            continue

        files = natsorted([f for f in os.listdir(video_dir)
                            if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])
        image_paths = [os.path.join(video_dir, f) for f in files]

        if not image_paths:
            print(f"  [跳過] 無影像：{video_dir}")
            continue

        save_dir = video_dir.rstrip('/') + args.name_notes
        print(f"\n[{vi+1}/{len(video_entries)}] {video_rel}  GT={TEMPORAL_LABELS[gt_label]}")
        print(f"  影像幀數：{len(image_paths)}")

        _, s_preds, _, t_preds, _ = process_one_video(
            args, model, device, transform,
            image_paths, img_root, save_dir,
            mp_detector=mp_detector, face_model=face_model,
            landmarks_db=landmarks_db, temporal_model=temporal_model)

        # ── 單幀模型：映射到 5 類後投票 ─────────────────────────
        valid_s = [single_pred_to_temporal_idx(int(p))
                   for p in s_preds if not _is_nan(p)]
        if valid_s:
            s_counter   = Counter(valid_s)
            s_video_pred, s_top_cnt = s_counter.most_common(1)[0]
            s_correct   = (s_video_pred == gt_label)
        else:
            s_video_pred, s_top_cnt, s_correct = -1, 0, False

        # ── 時序模型：直接投票 ───────────────────────────────────
        valid_t = [int(p) for p in t_preds if not _is_nan(p)]
        if valid_t and temporal_model is not None:
            t_counter   = Counter(valid_t)
            t_video_pred, t_top_cnt = t_counter.most_common(1)[0]
            t_correct   = (t_video_pred == gt_label)
        else:
            t_video_pred, t_top_cnt, t_correct = -1, 0, False

        eval_results.append({
            'video':        video_rel,
            'gt_label':     gt_label,
            'gt_name':      TEMPORAL_LABELS[gt_label],
            's_pred':       s_video_pred,
            's_pred_name':  TEMPORAL_LABELS[s_video_pred] if s_video_pred >= 0 else 'N/A',
            's_cnt':        s_top_cnt,
            's_total':      len(valid_s),
            's_correct':    s_correct,
            't_pred':       t_video_pred,
            't_pred_name':  TEMPORAL_LABELS[t_video_pred] if t_video_pred >= 0 else 'N/A',
            't_cnt':        t_top_cnt,
            't_total':      len(valid_t),
            't_correct':    t_correct,
        })

        # 即時顯示本影片結果
        s_mark = "✓" if s_correct else "✗"
        t_mark = "✓" if t_correct else "✗"
        print(f"  Single  : {eval_results[-1]['s_pred_name']:12s} [{s_mark}]  "
              f"({s_top_cnt}/{len(valid_s)} 幀)")
        if temporal_model:
            print(f"  Temporal: {eval_results[-1]['t_pred_name']:12s} [{t_mark}]  "
                  f"({t_top_cnt}/{len(valid_t)} 幀)")

    # 儲存報告
    report_dir = os.path.dirname(os.path.abspath(list_path))
    print_eval_report(eval_results, report_dir,
                      use_temporal=(temporal_model is not None))


def _is_nan(v):
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ══════════════════════════════════════════════════════════════════
#  批次評估報告
# ══════════════════════════════════════════════════════════════════
def print_eval_report(eval_results, save_dir, use_temporal=True):
    """
    輸出影片級準確度比較報告，並儲存 CSV。
    """
    if not eval_results:
        print("無評估結果。")
        return

    n          = len(eval_results)
    s_correct  = sum(r['s_correct'] for r in eval_results)
    t_correct  = sum(r['t_correct'] for r in eval_results) if use_temporal else 0

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  影片級準確度比較報告（Video-Level Accuracy Report）")
    lines.append("=" * 80)

    # ── 表頭 ────────────────────────────────────────────────────
    header_s = "SingleFrame(mapped)"
    header_t = "Temporal" if use_temporal else "(未啟用)"
    lines.append(f"{'影片資料夾':<45} {'GT':^12} {header_s:^22} {header_t:^22}")
    lines.append("-" * 80)

    # ── 每行 ────────────────────────────────────────────────────
    for r in eval_results:
        s_str = f"{r['s_pred_name']} ({'✓' if r['s_correct'] else '✗'})"
        t_str = (f"{r['t_pred_name']} ({'✓' if r['t_correct'] else '✗'})"
                 if use_temporal else "-")
        vid_short = r['video'][-42:] if len(r['video']) > 42 else r['video']
        lines.append(f"{vid_short:<45} {r['gt_name']:^12} {s_str:^22} {t_str:^22}")

    lines.append("=" * 80)
    lines.append(f"  [單幀模型]  影片級準確度：{s_correct}/{n} = {s_correct/n*100:.1f}%")
    if use_temporal:
        lines.append(f"  [時序模型]  影片級準確度：{t_correct}/{n} = {t_correct/n*100:.1f}%")

    # ── 各類別準確度 ─────────────────────────────────────────────
    lines.append("\n  各類別準確度（依 GT 類別分組）：")
    for cls_idx, cls_name in enumerate(TEMPORAL_LABELS):
        subset = [r for r in eval_results if r['gt_label'] == cls_idx]
        if not subset:
            continue
        ns = len(subset)
        ss = sum(r['s_correct'] for r in subset)
        ts = sum(r['t_correct'] for r in subset) if use_temporal else 0
        t_str = f"  Temporal {ts}/{ns}={ts/ns*100:.0f}%" if use_temporal else ""
        lines.append(f"    {cls_name:12s}  "
                     f"Single {ss}/{ns}={ss/ns*100:.0f}%{t_str}")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    # ── 儲存 TXT ─────────────────────────────────────────────────
    txt_path = os.path.join(save_dir, 'eval_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n報告儲存 → {txt_path}")

    # ── 儲存 CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, 'eval_report.csv')
    fieldnames = ['video', 'gt_label', 'gt_name',
                  's_pred', 's_pred_name', 's_cnt', 's_total', 's_correct']
    if use_temporal:
        fieldnames += ['t_pred', 't_pred_name', 't_cnt', 't_total', 't_correct']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(eval_results)
    print(f"CSV 儲存 → {csv_path}")


# ══════════════════════════════════════════════════════════════════
#  plot_results（模式 A 使用）
# ══════════════════════════════════════════════════════════════════
def plot_results(x, y, probs_matrix, save_dir,
                 temporal_y=None, temporal_probs_matrix=None):
    use_temporal    = (temporal_y is not None and temporal_probs_matrix is not None)
    single_n        = len(SINGLE_LABELS)
    temporal_n      = len(TEMPORAL_LABELS)

    csv_path = os.path.join(save_dir, 'result_data.csv')
    s_hdr    = [f"Single_{l}" for l in SINGLE_LABELS]
    t_hdr    = [f"Temporal_{l}" for l in TEMPORAL_LABELS] if use_temporal else []

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_ID', 'Single_Prediction', 'Face_Detected'] + s_hdr
                        + (['Temporal_Prediction'] + t_hdr if use_temporal else []))
        for i in range(len(x)):
            pi = y[i]
            if _is_nan(pi):
                row = [x[i], 'NaN', False] + probs_matrix[i].tolist()
            else:
                row = [x[i], SINGLE_LABELS[int(pi)], True] + probs_matrix[i].tolist()
            if use_temporal:
                ti = temporal_y[i]
                row += ['NaN' if _is_nan(ti) else TEMPORAL_LABELS[int(ti)]] \
                       + temporal_probs_matrix[i].tolist()
            writer.writerow(row)
    print(f"CSV 儲存 → {csv_path}")

    n_rows = 4 if use_temporal else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 6 * n_rows), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2, 1, 2] if use_temporal else [1, 2]})
    axes = list(axes)

    axes[0].plot(x, y, '-o', ms=3, lw=1.5, color='steelblue')
    axes[0].set_title("Single Frame Model - Classification Result", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Emotion")
    axes[0].set_yticks(range(single_n))
    axes[0].set_yticklabels(SINGLE_LABELS)
    axes[0].set_ylim(-0.5, single_n - 0.5)
    axes[0].grid(True, axis='y', ls='--', alpha=0.5)

    for ci in range(single_n):
        axes[1].plot(x, probs_matrix[:, ci], label=SINGLE_LABELS[ci],
                     color=SINGLE_COLORS_PLT[ci], lw=1.5, alpha=0.8)
    axes[1].set_title("Single Frame Model - Confidence per Class", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, ls=':', alpha=0.3)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if use_temporal:
        axes[2].plot(x, temporal_y, '-s', ms=3, lw=1.5, color='darkorange')
        axes[2].set_title("Temporal Model - Classification Result", fontsize=14, fontweight='bold')
        axes[2].set_ylabel("Emotion")
        axes[2].set_yticks(range(temporal_n))
        axes[2].set_yticklabels(TEMPORAL_LABELS)
        axes[2].set_ylim(-0.5, temporal_n - 0.5)
        axes[2].grid(True, axis='y', ls='--', alpha=0.5)

        for ci in range(temporal_n):
            axes[3].plot(x, temporal_probs_matrix[:, ci], label=TEMPORAL_LABELS[ci],
                         color=TEMPORAL_COLORS_PLT[ci], lw=1.5, alpha=0.8, ls='--')
        axes[3].set_title("Temporal Model - Confidence per Class", fontsize=14, fontweight='bold')
        axes[3].set_xlabel("Frame ID")
        axes[3].set_ylabel("Probability")
        axes[3].set_ylim(0, 1.05)
        axes[3].grid(True, ls=':', alpha=0.3)
        axes[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        axes[1].set_xlabel("Frame ID")

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'result_graph.png')
    plt.savefig(save_path, dpi=300)
    print(f"圖表儲存 → {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  影片層級統計（模式 A 使用）
# ══════════════════════════════════════════════════════════════════
def print_video_classification(y_single, y_temporal, save_dir):
    lines = []
    lines.append("=" * 55)
    lines.append("  影片層級分類結果（Video-Level Classification）")
    lines.append("=" * 55)

    valid_s = [int(p) for p in y_single if not _is_nan(p)]
    if valid_s:
        c = Counter(valid_s); total = len(valid_s)
        top_idx, top_cnt = c.most_common(1)[0]
        lines.append(f"\n[單幀影像幀模型]  有效幀：{total}")
        lines.append(f"  最終分類：{SINGLE_LABELS[top_idx]}  ({top_cnt}/{total} = {top_cnt/total*100:.1f}%)")
        lines.append("  各類別票數：")
        for i, lbl in enumerate(SINGLE_LABELS):
            cnt = c.get(i, 0)
            lines.append(f"    {lbl:12s} {cnt:5d}  {'█'*int(cnt/total*30)}")
    else:
        lines.append("\n[單幀影像幀模型]  無有效預測。")

    valid_t = [int(p) for p in y_temporal if not _is_nan(p)]
    if valid_t:
        c = Counter(valid_t); total = len(valid_t)
        top_idx, top_cnt = c.most_common(1)[0]
        lines.append(f"\n[時序特徵分析模型] 有效幀：{total}")
        lines.append(f"  最終分類：{TEMPORAL_LABELS[top_idx]}  ({top_cnt}/{total} = {top_cnt/total*100:.1f}%)")
        lines.append("  各類別票數：")
        for i, lbl in enumerate(TEMPORAL_LABELS):
            cnt = c.get(i, 0)
            lines.append(f"    {lbl:12s} {cnt:5d}  {'█'*int(cnt/total*30)}")
    else:
        lines.append("\n[時序特徵分析模型] 無有效預測（可能未啟用時序模型）。")

    lines.append("=" * 55)
    report = "\n".join(lines)
    print(report)

    txt_path = os.path.join(save_dir, 'video_classification.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n分類報告儲存 → {txt_path}")


# ══════════════════════════════════════════════════════════════════
#  進入點
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device} GPU {args.gpu}")
    print(f"座標來源：{args.coords_mode}")

    # ── 確認執行模式 ────────────────────────────────────────────
    if not args.eval_list and not args.image_folder:
        print("請指定 --eval_list 或 --image_folder。")
        exit(1)

    # ── 座標來源初始化 ──────────────────────────────────────────
    mp_detector  = None
    face_model   = None
    landmarks_db = None

    if args.coords_mode == 'mediapipe':
        mp_detector = build_mediapipe_detector(args.mediapipe_model_path)
        try:
            face_model = YOLO(args.yolo_model)
        except Exception:
            print("指定 YOLO 模型載入失敗，改用 yolov8n.pt。")
            face_model = YOLO('yolov8n.pt')
    else:
        if not args.landmarks_db:
            print("coords_mode=precomputed 需要指定 --landmarks_db。")
            exit(1)
        landmarks_db = load_landmarks_db(args.landmarks_db)

    # ── 情緒模型載入 ────────────────────────────────────────────
    model = load_emotion_model(args, device)
    transform = get_transforms()

    temporal_model = None
    if args.temporal_checkpoint:
        temporal_model = load_temporal_model(args, device)
    else:
        print("\n未指定 --temporal_checkpoint，時序模型不啟用。")

    # ── 執行 ────────────────────────────────────────────────────
    if args.eval_list:
        run_eval_list(args, model, temporal_model, device, transform,
                      mp_detector, face_model, landmarks_db)
    else:
        run_single_folder(args, model, temporal_model, device, transform,
                          mp_detector, face_model, landmarks_db)
