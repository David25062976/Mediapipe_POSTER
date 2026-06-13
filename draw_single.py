import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from natsort import natsorted  # 建議安裝 pip install natsort
from collections import OrderedDict

# 引入您的模型定義
from models.emotion_hyp_affect import pyramid_trans_expr

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/face_croped_customer_ori2/istock_like/face_croped_istockphoto-833170332-640_adpp_is_face1310', help='Path to the folder containing video frames')
    parser.add_argument('--checkpoint_single', type=str, default='./checkpoint/20260115-160251/epoch32_acc0.84025.pth', help='Path to the trained model .pth file')
    parser.add_argument('--modeltype', type=str, default='large', help='Model type: small, base, or large (must match training)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    return parser.parse_args()

def load_model(args, device):
    print(f"Loading model architecture: pyramid_trans_expr (type={args.modeltype})...")
    # 初始化模型結構
    model = pyramid_trans_expr(img_size=224, num_classes=8, type=args.modeltype)
    
    print(f"Loading weights from {args.checkpoint_single}...")
    checkpoint = torch.load(args.checkpoint_single, map_location=device)
    
    # 處理原本訓練時儲存的格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # --- 關鍵修正：處理 DataParallel 的 'module.' 前綴 ---
    # 因為原本訓練用了 DataParallel，權重名稱會有 module.xxx
    # 如果現在單卡跑或 CPU 跑，需要移除 module.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print('load_weight', len(new_state_dict))
    model.to(device)
    model.eval() # 設定為評估模式
    return model

def get_transforms():
    # 必須與訓練時的 data_transforms_val 完全一致
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_folder(model, folder_path, device, transform):
    results = []
    frame_indices = []
    
    # 讀取並使用自然排序 (確保 frame_2 在 frame_10 之前)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    try:
        image_files = natsorted(image_files)
    except:
        print("Warning: natsort not installed, using default sort.")
        image_files = sorted(image_files)

    print(f"Found {len(image_files)} images. Starting inference...")

    with torch.no_grad():
        for i, filename in enumerate(image_files):
            img_path = os.path.join(folder_path, filename)
            
            try:
                # 讀取圖片並轉為 RGB
                img = Image.open(img_path).convert('RGB')
                
                # 預處理
                input_tensor = transform(img).unsqueeze(0).to(device) # [1, 3, 224, 224]
                
                # 推論
                outputs, _ = model(input_tensor) # 根據您的模型 output 是 (outputs, features)
                
                # 取得預測結果
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                pred_label = predicted.item()
                
                results.append(pred_label)
                frame_indices.append(i + 1) # 假設 frame 從 1 開始計數
                
                # 顯示進度
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(image_files)} frames...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return frame_indices, results

def plot_results(x, y, save_path='result_graph.png'):
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, linestyle='-', marker='o', markersize=2, linewidth=1, color='royalblue')
    
    plt.title("Emotion Classification per Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Class ID (0-7)")
    
    # 設定 Y 軸只顯示整數類別
    plt.yticks(range(0, 8), labels=["Neutral", "Happy", "Sad", "Surprise", "Like", "Hesitate", "Anger", "Dislike"])
    plt.ylim(-0.5, 7.5)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # plt.show()
    # 如果想存檔，可以取消下面註解
    plt.savefig(save_path)
    print("Graph displayed.")

def draw_single(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = load_model(args, device)
    val_transform = get_transforms()

    if os.path.exists(args.image_folder):
        frames, predictions = predict_folder(model, args.image_folder, device, val_transform)
        
        # 4. 畫圖
        if len(frames) > 0:
            save_dir = os.path.join('draw_result', os.path.basename(args.image_folder))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, 'result_single.png')
            plot_results(frames, predictions, save_path)
        else:
            print("No valid images found in the directory.")
    else:
        print(f"Error: Folder {args.image_folder} does not exist.")
    
    return frames, predictions

if __name__ == "__main__":
    args = parse_args()
    
    # 設定裝置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. 載入模型
    model = load_model(args, device)
    
    # 2. 準備 transform
    val_transform = get_transforms()
    
    # 3. 執行推論
    if os.path.exists(args.image_folder):
        frames, predictions = predict_folder(model, args.image_folder, device, val_transform)
        
        # 4. 畫圖
        if len(frames) > 0:
            plot_results(frames, predictions)
        else:
            print("No valid images found in the directory.")
    else:
        print(f"Error: Folder {args.image_folder} does not exist.")
