# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
# export __GLX_VENDOR_LIBRARY_NAME=nvidia

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import torch.utils.data as data
from torchvision import transforms
import os
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_preprocessing.sam import SAM
from torchsampler import ImbalancedDatasetSampler

# 替換為你的 Dataset 匯入路徑 (若有不同請自行修正)
from data_preprocessing.dataset_affectnet_8class_pt import Affectdataset_8class_2

# ★ 關鍵修改 1：匯入我們剛剛改好的 P3 版本主模型
from models.emotion_hyp_affect_mediapipe_pt_attention_map_p3 import pyramid_trans_expr as pyramid_trans_expr_attention_map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet8class_shopping', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/20260605-190214/best.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')

    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0000015, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=999, help='Tolerating no progress for n epochs.')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze ir layer or not')
    parser.add_argument('--mediapipe_points', type=int, default=478, help='設定最大點數 478')
    parser.add_argument('--mediapipe_patch_size', type=int, default=14, help='14 or 24')
    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=str, default=123, help='torch.manual_seed({seed})')
    parser.add_argument('--save_dir', type=str, default="phase3_results_2", help='model save directory')
    parser.add_argument('--remark', type=str, default="phase3 dataset train, more points", help='attention map Phase 2')
    
    # === P2/P3 新增參數 ===
    parser.add_argument('--dynamic_droptoken', action='store_true', default=False, help='Enable dynamic drop token training (Phase 2)')
    parser.add_argument('--importance_order', type=str, default='./checkpoint/20260605-012234/symmetrized_global_points_importance_order.pt', help='Path to the importance order pt file')
    
    # === ★ Phase 3 專屬參數 ===
    parser.add_argument('--phase3_test', action='store_true', default=True, help='執行第三階段：階梯式點數推論測試與繪圖')

    return parser.parse_args()


def evaluate_stepped_inference(model, test_loader, importance_order, save_dir):
    """
    執行階梯式點數測試，測量準確度與推論時間，並繪製雙軸圖表。
    """
    print("\n" + "="*50)
    print("🚀 啟動第三階段：階梯式推論測試與效能評估")
    print(f"總點數: {importance_order}")
    print("="*50)
    
    # 設定你想測試的點數階梯 (從 478 一路往下砍到 49)
    test_steps = [478, 450, 425, 400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 49, 25, 20, 15, 10]
    
    acc_results = []
    time_results = []
    
    model.eval()
    with torch.no_grad():
        for N in test_steps:
            print(f"\n>>> 正在測試保留點數 N = {N} ...")
            
            # 1. 取出前 N 個重要的點，並重新排序保持空間對齊
            top_n_indices = importance_order[:N]
            active_indices, _ = torch.sort(top_n_indices)
            
            # 2. 測速專用 Warm-up (先跑幾次讓 GPU 時脈升上來，不紀錄時間)
            warmup_batches = 3
            for i, (imgs, targets, coords) in enumerate(test_loader):
                if i >= warmup_batches: break
                _ = model(imgs.cuda(), coords.cuda(), active_indices=active_indices)
            
            # 3. 正式開始測量
            total_time_ms = 0.0
            test_pre_labels, test_gt_labels = [], []
            
            # 設定 CUDA Events 精準測量 GPU 時間
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            test_loader_tqdm = tqdm(test_loader, desc=f"Testing N={N}", leave=False)
            for imgs, targets, coords in test_loader_tqdm:
                imgs, targets, coords = imgs.cuda(), targets.cuda(), coords.cuda()
                
                # 確保 GPU 前面的任務都做完了再開始計時
                torch.cuda.synchronize()
                start_event.record()
                
                outputs, _ = model(imgs, coords, active_indices=active_indices)
                
                end_event.record()
                torch.cuda.synchronize() # 等待 GPU 運算完成
                
                total_time_ms += start_event.elapsed_time(end_event)
                
                _, predicts = torch.max(outputs, 1)
                test_pre_labels.extend(predicts.cpu().tolist())
                test_gt_labels.extend(targets.cpu().tolist())
                
            # 計算準確度與平均每個 Batch 的推論時間 (ms)
            acc = accuracy_score(test_gt_labels, test_pre_labels)
            avg_time = total_time_ms / len(test_loader)
            
            acc_results.append(acc)
            time_results.append(avg_time)
            
            print(f"結果 -> 準確度: {acc*100:.2f}% | 平均推論時間: {avg_time:.2f} ms/batch")

    # ==========================================
    # ★ 繪製精美的雙軸圖表 (Dual-axis Chart)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X 軸設定為字串，確保各階梯間距平均
    x_labels = [str(n) for n in test_steps]
    x_pos = np.arange(len(x_labels))

    # 繪製左側 Y 軸：Accuracy (折線圖)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of MediaPipe Points Kept (Top-N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', color=color1, fontsize=12, fontweight='bold')
    # 將準確度轉為百分比
    acc_percentages = [acc * 100 for acc in acc_results]
    line1 = ax1.plot(x_pos, acc_percentages, color=color1, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 設定 Y 軸上下限讓變化更明顯 (可依實際結果微調)
    min_acc = min(acc_percentages) - 2
    max_acc = max(acc_percentages) + 2
    ax1.set_ylim([min_acc, max_acc])

    # 繪製右側 Y 軸：Inference Time (長條圖或另一條折線圖，這裡用折線圖對比更清晰)
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Inference Time (ms / batch)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(x_pos, time_results, color=color2, marker='s', linestyle='--', linewidth=2, label='Inference Time')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 合併圖例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Impact of Token Pruning on Accuracy and Inference Speed', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, x_labels)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    save_path = os.path.join(save_dir, 'phase3_stepped_test_result.png')
    plt.savefig(save_path, dpi=300)
    print("="*50)
    print(f"✅ 第三階段完成！雙軸圖表已完美儲存至: {save_path}")
    print("="*50)


def run_training():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join('./checkpoint', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "affectnet8class_shopping":
        # 如果是 Phase3 測試，其實只要載入 test_set 即可
        test_data_dir = './data/AffectNet8c_Shopping_test/train_set'
        test_landmarks_db = './data/AffectNet8c_Shopping_test/train_set_landmarks_all.pt'
        
        class_names = ['Neutral', 'Happy', 'Sad', 'browsing', 'interesting', 'thinking', 'buying', 'passing']
        num_classes = len(class_names)
        
        if os.path.exists(test_data_dir):
            test_dataset = Affectdataset_8class_2(test_data_dir, test_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
            
        model = pyramid_trans_expr_attention_map(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size)
    else:
        return print('Please configure dataset in the code.')

    # 載入 Checkpoint (必須要有，否則測試無意義)
    if args.checkpoint:
        print(f"Loading pretrained weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            
        # ★ 修復核心：手動剝離 module. 前綴，確保權重能完美對應！
        new_state_dict = {}
        matched_keys = 0
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 將剝離後的權重載入模型
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # 顯示載入診斷結果，確認不是在瞎猜
        print(f"✅ 成功對應了 {len(new_state_dict) - len(unexpected_keys)} 個權重層！")
        if len(missing_keys) > 0:
            print(f"⚠️ 警告：有 {len(missing_keys)} 個層沒有載入到權重 (如果只有一點點是正常的，如果很多代表架構有錯)")
            for missing_key in missing_keys: print(missing_key)
    else:
        print("⚠️ 警告：沒有指定 --checkpoint，將使用隨機權重進行測試！")
        
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # ==========================================
    # ★ 攔截：如果只指定執行 Phase 3 測試
    # ==========================================
    if args.phase3_test:
        if not os.path.exists(args.importance_order):
            raise FileNotFoundError(f"找不到重要性排序表：{args.importance_order}，請確認路徑！")
            
        global_importance_order = torch.load(args.importance_order).cuda()
        
        # 進入測試與繪圖流程
        evaluate_stepped_inference(model, test_loader, global_importance_order, args.save_dir)
        return # 執行完畢直接結束程式
    
    # ==========================================
    # (保留：原本的訓練迴圈，如果你需要的話)
    # ==========================================
    print("Normal training phase code is skipped in this snippet for brevity.")
    # 如果你要訓練，請把前一次對話 P2 版本的訓練迴圈貼補在這裡

if __name__ == "__main__":
    run_training()