# ──────────────────────────────────────────────────────────────────────────────
# Hand + Object Detection Module (可呼叫函式)
# ──────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
from ultralytics import YOLO
import cv2
import numpy as np

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 初始化 YOLOv8 (載入通用物件偵測模型)
yolo_model = YOLO("/home/lab702/POSTER/models/yolov8m.pt")  # 你可以換 yolov8n.pt 或自己訓練的版本

def detect_hand_objects(frame_bgr: np.ndarray):
    """
    輸入: BGR 影格
    輸出:
      - annotated_frame: 已畫框影像
      - hand_obj_map: dict 例如 {'left': 'cup', 'right': 'phone'}
    """
    h, w, _ = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_results = hands_model.process(frame_rgb)

    # Step 1：取得手掌中心
    hand_centers = []
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for lm, handed in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handed.classification[0].label.lower()  # left / right
            cx = int(lm.landmark[mp_hands.HandLandmark.WRIST].x * w)
            cy = int(lm.landmark[mp_hands.HandLandmark.WRIST].y * h)
            hand_centers.append({"id": label, "center": (cx, cy)})
            mp_draw.draw_landmarks(frame_bgr, lm, mp_hands.HAND_CONNECTIONS)

    # Step 2：YOLO 偵測物體
    yolo_results = yolo_model(frame_bgr, verbose=False)
    objects = []
    for r in yolo_results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < 0.4:
                continue
            cls = int(box.cls)
            name = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append({"name": name, "box": [x1, y1, x2, y2]})
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"{name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 3：距離匹配 (找出手最近的物體)
    hand_obj_map = {}
    for hand in hand_centers:
        hx, hy = hand["center"]
        nearest, dist_min = None, 1e9
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            ox, oy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.hypot(hx - ox, hy - oy)
            if dist < dist_min:
                nearest, dist_min = obj["name"], dist
        hand_obj_map[hand["id"]] = nearest or "None"

        # 在畫面上標記
        cv2.circle(frame_bgr, (hx, hy), 8, (255, 0, 0), -1)
        cv2.putText(frame_bgr, f"{hand['id']} → {hand_obj_map[hand['id']]}",
                    (hx - 60, hy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    return frame_bgr, hand_obj_map
