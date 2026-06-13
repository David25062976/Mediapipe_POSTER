import os
import torch
import warnings
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision import transforms
from models.emotion_hyp import pyramid_trans_expr
from utils import *
import torch.nn.functional as F
import mediapipe as mp

warnings.filterwarnings("ignore")

# === 人臉偵測器 ===
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === 預設模型與 label 函數 ===
current_model_path = "checkpoint/Mix_7_0.8971.pth"
get_label_func = None  # 初始化為 None，稍後由 switch_model 設定

def get_label_model1():  # 直接回傳 labels list
    return ["neatral", "happy", "sad", "surprise", "Fear", "disgust", "angry", "contempt"]  #0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

def get_label_model2():
    return ["surprise", "Fear", "Disgust", "Happy", "sad", "angry", "neutral"]

def get_label_default():
    return ["neutral", "angry", "sad", "happy", "worried", "surprise", "pdface"]


# === 模型對應 label 函數表 ===
model_config = {
    "Mix_7_0.8971.pth": get_label_default,
    "affect8_best.pth": get_label_model1,
    "epoch3_acc0.6122.pth": get_label_model2,
    "rafdb_best.pth": get_label_model2,
    "rafbasic_acc0.9078.pth": get_label_model2,
    "affect8_best.pth": get_label_model1,
    "affect_acc0.6344.pth": get_label_model1,
}

def load_model(device, model_path, num_classes):
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type="large")
    print("Loading pretrained weights...", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model.to(device)
    model.eval()
    return model

# === 影像預處理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 單張圖片預測 ===
def predict_probs(model, image, device):
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, _ = model(image)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted = np.argmax(probs)
    return predicted, probs

# === 視覺輸出文字 ===
def overlay_icon(frame, text, x, y):
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    font_path = "C:\\Windows\\Fonts\\msyh.ttc"
    font = ImageFont.truetype(font_path, 25)
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

# === 畫面更新 ===
mp_face_mesh = mp.solutions.face_mesh# 初始化 mediapipe face mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# 替代 update_frame（建議整段取代）
def update_frame():
    global running, cap, model, device, get_label_func
    if not running:
        cap.release()
        return

    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                # --- 擷取五官邊界框（以整體輪廓估算） ---
                x_coords = [int(lm.x * iw) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * ih) for lm in face_landmarks.landmark]
                x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, iw)
                y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, ih)

                face_roi = frame[y_min:y_max, x_min:x_max]
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                predicted_class, probs = predict_probs(model, face_pil, device)
                emotion = get_label_func(predicted_class)
                print(f"Predicted class: {predicted_class}, Emotion: {emotion}")

                frame = overlay_icon(frame, emotion, x_min, y_min - 10)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                for idx, p in enumerate(probs):
                    label = get_label_func(idx)
                    text = f"{label}: {p*100:.1f}%"
                    cv2.putText(frame, text, (x_max + 10, y_min + 20 + idx * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # --- 額外：畫出五官點位（可選）---
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(50, update_frame)


# === 模型切換 ===
def switch_model(model_path):
    global model, current_model_path, get_label_func
    print(f"Switching model to: {model_path}")
    current_model_path = model_path

    labels = get_label_default()  # fallback
    for key, label_func in model_config.items():
        if key in model_path:
            labels = label_func()
            break

    get_label_func = lambda idx: labels[idx] if 0 <= idx < len(labels) else "unknown"
    model = load_model(device, model_path, num_classes=len(labels))

# === GUI 與啟動 ===
def stop_detection():
    global running
    running = False
    root.quit()

root = tk.Tk()
root.title("Emotion Detection")
root.geometry("800x700")

video_label = tk.Label(root)
video_label.pack()

btn_model1 = tk.Button(root, text="rafdb_best", command=lambda: switch_model("checkpoint/rafdb_best.pth"), font=("Arial", 14))
btn_model1.pack()

btn_model2 = tk.Button(root, text="rafdb_0.9078", command=lambda: switch_model("checkpoint/rafbasic_acc0.9078.pth"), font=("Arial", 14))
btn_model2.pack()

btn_model3 = tk.Button(root, text="affect8_best", command=lambda: switch_model("checkpoint/affect8_best.pth"), font=("Arial", 14))
btn_model3.pack()

btn_model4 = tk.Button(root, text="affect_acc0.6344", command=lambda: switch_model("checkpoint/affect_acc0.6344.pth"), font=("Arial", 14))
btn_model4.pack()

'''btn_model5 = tk.Button(root, text="預設模型", command=lambda: switch_model("checkpoint/Mix_7_0.8971.pth"), font=("Arial", 14))
btn_model5.pack()'''

btn_stop = tk.Button(root, text="結束程式", command=stop_detection, font=("Arial", 14))
btn_stop.pack(expand=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
switch_model(current_model_path)
cap = cv2.VideoCapture(0)
running = True
update_frame()

root.mainloop()
