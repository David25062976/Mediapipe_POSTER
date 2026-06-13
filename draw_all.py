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
from draw_single2 import draw_single
from draw_window2 import draw_window

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Window-based Emotion Recognition')
    parser.add_argument('--image_folder', type=str, default='/home/Dataset/customer_ori2/out/English Vocabulary at a CLOTHING STORE - Real Engl/English Vocabulary at a CLOTHING STORE - Real Engl-2',  help='Path to the folder containing video frames')
    parser.add_argument('--checkpoint_single', type=str, default='./checkpoint/20260128-100001/epoch25_acc0.8408.pth', help='Path to the trained model .pth file')
    parser.add_argument('--checkpoint_window', type=str,  default='/home/lab702/POSTER/checkpoint/window_20260128-095759/epoch41_acc0.9304.pth', help='Path to the trained model .pth file')
    parser.add_argument('--yolo_path', type=str, default='/home/lab702/POSTER/data_preprocessing/head_detect_medium.pt', help='Path to yolov8 face/nano model')
    
    # 模型結構參數 (需與訓練一致)
    parser.add_argument('--modeltype', type=str, default='large', help='Model type: small, base, large')
    parser.add_argument('--temporal_type', type=str, default='MLP', help='Temporal type: MLP / CNN / Transformer')
    parser.add_argument('--face_crop', action='store_true', default=True, help='Apply face crop (Recommended if trained with it)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    
    return parser.parse_args()

def draw_all(args):
    # 建立畫布
    plt.figure(figsize=(18, 8)) 

    # ==========================================
    # 1. 取得資料 (先全部算完再畫，邏輯比較清楚)
    # ==========================================
    print("Running Single Frame Inference...")
    frames_single, preds_single = draw_single(args)
    print(f"-> Single Frame data loaded: {len(frames_single)} points, Max Class: {max(preds_single) if preds_single else 'None'}")

    print("Running Window Inference (Size=8, Stride=8)...")
    frames_window1, preds_window1, all_probs1 = draw_window(args, window_size=8, stride=8)
    print(f"-> Window (Stride=8) data loaded: {len(frames_window1)} points")

    print("Running Window Inference (Size=8, Stride=1)...")
    frames_window2, preds_window2, all_probs2 = draw_window(args, window_size=8, stride=1)
    print(f"-> Window (Stride=1) data loaded: {len(frames_window2)} points")

    # ==========================================
    # 2. 開始繪圖
    # ==========================================
    
    # --- Line 1: Single Frame (底層背景) ---
    for i, item in enumerate(preds_single):
        if item == 7:
            preds_single[i] = 0
        elif item == 5:
            preds_single[i] = 1
        elif item == 4:
            preds_single[i] = 2
        else:
            preds_single[i] = -1

    if frames_single:
        plt.plot(frames_single, preds_single, 
                 label='Single Frame (Base)', 
                 color='black', 
                 linewidth=2, 
                 linestyle='-', 
                 alpha=0.8,
                 zorder=1)

    # --- Line 2: Window Stride=8 (中層) ---
    if frames_window1:
        plt.plot(frames_window1, preds_window1, 
                 label='Window (Stride=8)', 
                 color='dodgerblue',  # 改用亮一點的藍色比較明顯
                 linewidth=2.5, 
                 linestyle='--', 
                 alpha=0.9,
                 zorder=2)

    # --- Line 3: Window Stride=1 (最上層) ---
    if frames_window2:
        plt.plot(frames_window2, preds_window2, 
                 label='Window (Stride=1, Dense)', 
                 color='crimson', 
                 linewidth=2, 
                 linestyle='-', 
                 alpha=1.0,
                 zorder=3)

    # ==========================================
    # 3. 圖表美化設定 (修正重點)
    # ==========================================
    plt.title("Comparison of Emotion Recognition Strategies", fontsize=16, fontweight='bold')
    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("Class ID", fontsize=12)
    
    # 【修正 1】設定 Y 軸刻度：確保顯示 0~7
    max_classes = 3  # 假設你的最大分類是 8 類 (0~7)
    plt.yticks(range(0, max_classes)) 
    
    # 【修正 2】設定 Y 軸範圍：這裡必須包涵所有可能的類別
    # 之前寫 (-0.5, 2.5) 導致大於 2 的分類全部看不見
    plt.ylim(-0.5, max_classes - 0.5) 
    
    # 格線
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # 圖例
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    
    # 存檔路徑處理
    # save_dir = os.path.join('draw_result', os.path.basename(args.image_folder))
    save_dir = args.image_folder + '_draw'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'comparison_result.png')
    plt.savefig(save_path, dpi=300)
    print(f"Done. Image saved to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    draw_all(args)
