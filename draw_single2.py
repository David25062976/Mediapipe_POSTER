import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg') # 如果是在沒有螢幕的伺服器上跑，這行很重要
import matplotlib.pyplot as plt
import argparse
import numpy as np # 新增 numpy 用於處理矩陣
from natsort import natsorted
from collections import OrderedDict
from ultralytics import YOLO

# 引入您的模型定義
from models.emotion_hyp_affect import pyramid_trans_expr

# 定義情緒標籤 (用於圖例)
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Like", "Hesitate", "Anger", "Dislike"]
COLORS = ['gray', 'orange', 'blue', 'purple', 'green', 'cyan', 'red', 'brown'] # 對應每個情緒的顏色

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/face_croped_customer_ori2/pexels_hesitate/face_croped_8801813-uhd_2160_3840_24fps_face299', help='Path to the folder containing video frames')
    parser.add_argument('--checkpoint_single', type=str, default='./checkpoint/20260128-100001/epoch25_acc0.8408.pth', help='Path to the trained model .pth file')
    parser.add_argument('--modeltype', type=str, default='large', help='Model type: small, base, or large (must match training)')
    parser.add_argument('--face_crop', action='store_true', default=False, help='Crop face or not')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    return parser.parse_args()

def load_model(args, device):
    print(f"Loading model architecture: pyramid_trans_expr (type={args.modeltype})...")
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

def crop_face(face_model, image, path, best_crop_coordinate, scale=1.2):
    """
    核心邏輯：偵測人臉 -> 取最高信心度 -> 放大 1.3 倍 -> 切圖
    """

    results = face_model.predict(image, conf=0.01, verbose=False)

    if not results or len(results[0].boxes) == 0:
        if best_crop_coordinate is not None:
            x1, y1, x2, y2 = best_crop_coordinate
            best_crop_img = image[y1:y2, x1:x2]
            return best_crop_img, best_crop_coordinate
        else:
            print(f"Warning: Crop failed {path}")
            return image, best_crop_coordinate

    best_crop_img = None
    highest_score = 0
    
    # 4. 尋找信心程度最高的人頭 (假設類別 0 是你想偵測的目標)
    # 注意：在標準 YOLOv8n.pt 中，0 是 'person'。如果是專門的人頭模型，通常 0 也是 'head'。
    for result in results:
        boxes = result.boxes

        path = result.path
        filename = os.path.basename(path)
        
        # result.orig_img 是 BGR 格式的 NumPy 陣列 (OpenCV 格式)
        img_bgr = result.orig_img
        h, w = img_bgr.shape[:2]
        img_center_x, img_center_y = w / 2, h / 2

        final_crop_img = None
        boxes = result.boxes

        for box in boxes:
            conf = float(box.conf[0])
            # 取得座標 [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            face_center_x = ( x2 + x1 ) / 2
            face_center_y = ( y2 + y1 ) / 2

            half_x_scaled = ( x2 - x1 ) / 2 * scale
            half_y_scaled = ( y2 - y1 ) / 2 * scale
            h, w, _ = image.shape
            dist_to_img_center = ( (face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2 )**0.5
            
            # 計算縮放後的範圍
            face_w = x2 - x1
            face_h = y2 - y1
            new_w_half = (face_w * scale) / 2
            new_h_half = (face_h * scale) / 2
            
            # 【修正：評分邏輯】信心度高、面積大、且離中心近的優先
            # 分母加 1 避免距離為 0 的除錯
            score = (conf * face_w * face_h) / (dist_to_img_center + 1)
            if score > highest_score:
                highest_score = score
                # 限制座標邊界
                nx1 = max(0, int(face_center_x - new_w_half))
                ny1 = max(0, int(face_center_y - new_h_half))
                nx2 = min(w, int(face_center_x + new_w_half))
                ny2 = min(h, int(face_center_y + new_h_half))
                current_best_box = [nx1, ny1, nx2, ny2]
        
        if current_best_box:
            best_crop_coordinate = current_best_box
            x1, y1, x2, y2 = current_best_box
            final_crop_img = img_bgr[y1:y2, x1:x2]
        
    return final_crop_img, best_crop_coordinate

def predict_folder(model, folder_path, device, transform, face_crop):
    results = []
    frame_indices = []
    all_probs = [] # 【新增】 用來儲存每一幀的所有類別機率
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if face_crop:
        print("Loading head_detect_nano model for dataloader...")
        head_detect_model_name = 'head_detect_medium'
        try:
            face_model = YOLO(f'./data_preprocessing/{head_detect_model_name}.pt')
            print(f"{head_detect_model_name} model loaded")
        except:
            print("Warning: 'head_detect_nano.pt' not found, downloading standard 'yolov8n.pt' (might be less accurate for faces)")
            face_model = YOLO('yolov8n.pt') 
    else:
        face_model = None

    try:
        image_files = natsorted(image_files)
    except:
        print("Warning: natsort not installed, using default sort.")
        image_files = sorted(image_files)

    print(f"Found {len(image_files)} images. Starting inference...")

    with torch.no_grad():
        best_crop_coordinate = None
        for i, filename in enumerate(image_files):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot read {img_path}")
                # 產生一張全黑圖 (224, 224) 避免報錯，尺寸可依需求調整
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            # img = Image.open(img_path).convert('RGB')

            if face_crop:
                img, best_crop_coordinate = crop_face(face_model, img, img_path, best_crop_coordinate, scale=1.2)

            img = img[:, :, ::-1]
            input_tensor = transform(img)
            input_tensor = input_tensor.unsqueeze(0).cuda()
            
            outputs, _ = model(input_tensor)
            
            # 計算機率 (Softmax)
            probs = F.softmax(outputs, dim=1)
            
            # 取得預測類別
            _, predicted = torch.max(outputs, 1)
            pred_label = predicted.item()
            
            results.append(pred_label)
            frame_indices.append(i + 1)
            
            # 【新增】 將機率轉為 numpy array 並存入 list
            # probs shape: [1, 8] -> 取 [0] 變成 [8]
            all_probs.append(probs.cpu().numpy()[0]) 
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_files)} frames...")

    # 將 list 轉為 numpy array，方便後續切片繪圖 (Shape: [Frames, 8])
    return frame_indices, results, np.array(all_probs)

def plot_results(x, y, probs_matrix, save_path='result_graph.png'):
    """
    x: Frame indices
    y: Predicted labels (0-7)
    probs_matrix: Numpy array of shape (N_frames, 8)
    """
    # 建立一個有兩個子圖 (Subplots) 的畫布，共享 X 軸
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # === 子圖 1: 原始分類結果 (Top Prediction) ===
    ax1.plot(x, y, linestyle='-', marker='o', markersize=3, linewidth=1.5, color='black', label='Prediction')
    ax1.set_title("Frame-by-Frame Classification Result", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Class ID", fontsize=12)
    ax1.set_yticks(range(0, 8))
    ax1.set_yticklabels(EMOTION_LABELS) # 將 Y 軸數字換成文字標籤
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 7.5)

    # === 子圖 2: 所有類別的信心程度 (Confidence Scores) ===
    # 遍歷 8 個類別，畫出 8 條線
    for class_idx in range(8):
        class_probs = probs_matrix[:, class_idx] # 取出所有幀在該類別的機率
        ax2.plot(x, class_probs, 
                 label=f"{EMOTION_LABELS[class_idx]}", 
                 color=COLORS[class_idx], 
                 linewidth=1.5, 
                 alpha=0.8)

    ax2.set_title("Confidence Probability per Class", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Probability (0.0 - 1.0)", fontsize=12)
    ax2.set_ylim(0, 1.05) # 機率範圍 0~1
    ax2.grid(True, linestyle=':', alpha=0.3)
    
    # 設定圖例 (Legend) 放在圖表外側，避免遮擋線條
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Emotions")

    plt.tight_layout()
    
    # 存檔
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.close() # 關閉畫布釋放記憶體

def draw_single(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = load_model(args, device)
    val_transform = get_transforms()

    if os.path.exists(args.image_folder):
        # 【修改】 接收三個回傳值：indices, labels, probabilities
        frames, predictions, all_probs = predict_folder(model, args.image_folder, device, val_transform, args.face_crop)
        
        if len(frames) > 0:
            # save_dir = os.path.join('draw_result', os.path.basename(args.image_folder))
            save_dir = args.image_folder + '_draw'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, 'result_single_conf.png')
            
            # 【修改】 傳入 probabilities 進行繪圖
            plot_results(frames, predictions, all_probs, save_path)
        else:
            print("No valid images found in the directory.")
    else:
        print(f"Error: Folder {args.image_folder} does not exist.")
    
    return frames, predictions

if __name__ == "__main__":
    args = parse_args()
    draw_single(args)
