import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
from natsort import natsorted
from collections import OrderedDict
from ultralytics import YOLO 

# 引入您的模型定義
from models.emotion_hyp_window import pyramid_trans_expr_window

# 定義類別標籤與顏色 (假設 3 類)
CLASS_LABELS = ["Dislike", "Hesitate", "Like"]
COLORS = ['red', 'orange', 'green'] # 對應 Dislike, Hesitate, Like 的顏色

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Window-based Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/face_croped_customer_ori2/pexels_hesitate/face_croped_8801813-uhd_2160_3840_24fps_face299',  help='Path to the folder containing video frames')
    parser.add_argument('--checkpoint_window', type=str,  default='/home/lab702/POSTER/checkpoint/window_20260128-095759/epoch41_acc0.9304.pth', help='Path to the trained model .pth file')
    parser.add_argument('--yolo_path', type=str, default='/home/lab702/POSTER/data_preprocessing/head_detect_medium.pt', help='Path to yolov8 face/nano model')
    
    # 模型結構參數 (需與訓練一致)
    parser.add_argument('--modeltype', type=str, default='large', help='Model type: small, base, large')
    parser.add_argument('--temporal_type', type=str, default='MLP', help='Temporal type: MLP / CNN / Transformer')
    parser.add_argument('--window_size', type=int, default=8, help='Window size')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--face_crop', action='store_true', default=True, help='Apply face crop (Recommended if trained with it)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    return parser.parse_args()

class FaceCropper:
    """
    複製訓練程式碼中的裁切邏輯，確保測試時輸入一致
    """
    def __init__(self, model_path, scale=1.3):
        try:
            self.model = YOLO(model_path)
        except:
            print(f"Warning: {model_path} not found, downloading yolov8n.pt")
            self.model = YOLO('yolov8n.pt')
        self.scale = scale

    def crop(self, image, best_crop_coordinate):
        # image is BGR (OpenCV format)
        results = self.model.predict(image, conf=0.01, verbose=False)
        
        # 如果沒抓到臉，使用上一次最好的座標
        if not results or len(results[0].boxes) == 0:
            if best_crop_coordinate is not None:
                x1, y1, x2, y2 = best_crop_coordinate
                return image[y1:y2, x1:x2], best_crop_coordinate
            else:
                return image, best_crop_coordinate

        h, w = image.shape[:2]
        img_center_x, img_center_y = w / 2, h / 2
        highest_score = -1
        current_best_box = None
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                face_center_x = (x2 + x1) / 2
                face_center_y = (y2 + y1) / 2
                
                face_w = x2 - x1
                face_h = y2 - y1
                
                # 計算距離分數
                dist = ( (face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2 )**0.5
                score = (conf * face_w * face_h) / (dist + 1)
                
                if score > highest_score:
                    highest_score = score
                    # 擴大範圍
                    new_w_half = (face_w * self.scale) / 2
                    new_h_half = (face_h * self.scale) / 2
                    
                    nx1 = max(0, int(face_center_x - new_w_half))
                    ny1 = max(0, int(face_center_y - new_h_half))
                    nx2 = min(w, int(face_center_x + new_w_half))
                    ny2 = min(h, int(face_center_y + new_h_half))
                    current_best_box = [nx1, ny1, nx2, ny2]

        if current_best_box:
            x1, y1, x2, y2 = current_best_box
            return image[y1:y2, x1:x2], current_best_box
        
        return image, best_crop_coordinate

def load_model(args, device, window_size):
    print(f"Loading model: window_size={window_size}, type={args.modeltype}, temporal={args.temporal_type}")
    
    # 定義 freeze_list (測試時其實不影響，但初始化需要參數)
    freeze_list = [True, True, None] 
    
    model = pyramid_trans_expr_window(
        img_size=224, 
        num_classes=3, 
        window_size=window_size, 
        type=args.modeltype, 
        freeze_list=freeze_list, 
        temporal_type=args.temporal_type
    )
    
    print(f"Loading weights from {args.checkpoint_window}...")
    checkpoint = torch.load(args.checkpoint_window, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 移除 DataParallel 的 'module.' 前綴
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False) # strict=False 以容許部分不匹配(如果有的話)
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

def process_and_predict(model, folder_path, args, window_size, stride, device):
    # 1. 取得所有圖片並自然排序
    valid_ext = ('.jpg', '.png', '.jpeg')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    try:
        files = natsorted(files)
    except:
        files = sorted(files)
    
    full_paths = [os.path.join(folder_path, f) for f in files]
    total_frames = len(full_paths)
    print(f"Total frames found: {total_frames}")

    # 初始化工具
    cropper = FaceCropper(args.yolo_path) if args.face_crop else None
    transform = get_transforms()
    
    frame_indices = []
    results = []
    all_probs = [] # 【新增】 用來儲存機率矩陣
    
    # 2. 滑動視窗迴圈
    for i in range(0, total_frames - window_size + 1, stride):
        window_paths = full_paths[i : i + window_size]
        
        # 準備一個 batch 的資料
        window_imgs = []
        best_crop_coordinate = None
        
        for path in window_paths:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Face Crop
            if args.face_crop:
                img_bgr, best_crop_coordinate = cropper.crop(img_bgr, best_crop_coordinate)
            
            # BGR -> RGB
            img_rgb = img_bgr[:, :, ::-1]
            
            # Transform
            img_tensor = transform(img_rgb)
            window_imgs.append(img_tensor)
        
        # 堆疊成 Tensor: (1, C, T, H, W)
        data_tensor = torch.stack(window_imgs, dim=0)
        data_tensor = data_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        data_tensor = data_tensor.to(device)
        
        # 推論
        with torch.no_grad():
            outputs, _ = model(data_tensor)
            
            # 【新增】 計算機率分佈
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            label = predicted.item()
        
        # 紀錄結果 (只記錄 Window 的最後一幀)
        last_frame_index = i + window_size 
        
        frame_indices.append(last_frame_index)
        results.append(label)
        
        # 【新增】 儲存該 Window 的機率
        all_probs.append(probs.cpu().numpy()[0])

        if (i // stride + 1) % 5 == 0:
            print(f"Processed window starting at frame {i+1} -> Prediction: {label}")

    return frame_indices, results, np.array(all_probs)

def plot_graph(x, y, probs_matrix, save_path='window_result.png'):
    # 【修改】 建立雙子圖，共享 X 軸
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # === 子圖 1: 分類結果 (階梯圖) ===
    ax1.step(x, y, where='mid', color='black', linewidth=1.5, label='Prediction')
    ax1.plot(x, y, 'o', markersize=3, color='gray', alpha=0.5) # 標出每個點
    
    ax1.set_title("Sequence Classification Result (Window Based)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Class ID", fontsize=12)
    ax1.set_yticks(range(0, 3))
    ax1.set_yticklabels(CLASS_LABELS) # 使用文字標籤
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 2.5)
    
    # === 子圖 2: 信心程度曲線 ===
    # 遍歷 3 個類別 (0: Dislike, 1: Hesitate, 2: Like)
    for class_idx in range(3):
        class_probs = probs_matrix[:, class_idx]
        ax2.plot(x, class_probs, 
                 label=f"{CLASS_LABELS[class_idx]}", 
                 color=COLORS[class_idx], 
                 linewidth=2, 
                 alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Probability (0.0 - 1.0)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # 圖例
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.close()

def draw_window(args, window_size, stride):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = load_model(args, device, window_size)
    
    # 2. Inference
    if os.path.exists(args.image_folder):
        # 【修改】 接收三個回傳值
        frames, preds, all_probs = process_and_predict(model, args.image_folder, args, window_size, stride, device)
        
        # 3. Plot
        if len(frames) > 0:
            # save_dir = os.path.join('draw_result', os.path.basename(args.image_folder))
            save_dir = args.image_folder + '_draw'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f'result_window{window_size}_{stride}_conf.png')
            
            # 【修改】 傳入機率矩陣
            plot_graph(frames, preds, all_probs, save_path)
        else:
            print("No complete windows found (check folder size vs window_size).")
    else:
        print("Image folder not found.")
    
    return frames, preds, all_probs

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = load_model(args, device, args.window_size)
    
    # 2. Inference
    if os.path.exists(args.image_folder):
        window_size, stride = args.window_size, args.stride
        # 【修改】 接收三個回傳值
        frames, preds, all_probs = process_and_predict(model, args.image_folder, args, window_size, stride, device)
        
        # 3. Plot
        if len(frames) > 0:
            # save_dir = os.path.join('draw_result', os.path.basename(args.image_folder))
            save_dir = args.image_folder + '_draw'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f'result_window{window_size}_{stride}_conf.png')
            
            # 【修改】 傳入機率矩陣
            plot_graph(frames, preds, all_probs, save_path)
        else:
            print("No complete windows found.")
    else:
        print("Image folder not found.")
