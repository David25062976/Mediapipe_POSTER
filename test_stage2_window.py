import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import defaultdict
import torch.utils.data as data
from torchvision import transforms
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# 引入你的專案模組 (請確保路徑正確)
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class_ori import Affectdataset_8class
from data_preprocessing.dataset_customer import AffectDatasetWindow_facecrop
from models.emotion_hyp import pyramid_trans_expr
from models.emotion_hyp_window import pyramid_trans_expr_window
# from utils import * # 假設 load_pretrained_weights 在這裡

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='consumer', help='dataset: rafdb, affectnet, affectnet8class, consumer')
    # Checkpoint 現在是必須的，或者你可以在 default 填入預設路徑
    parser.add_argument('-c', '--checkpoint', type=str, default='/home/lab702/POSTER/checkpoint/window_20260119-152218_Transformer/epoch37_acc0.9082.pth', help='Path to the Pytorch checkpoint file') 
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    
    # Model configuration
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--window_size', type=int, default=8, help='Window size of sliding window')
    parser.add_argument('--stride', type=int, default=8, help='Stride of sliding window')
    parser.add_argument('--face_crop', action='store_true', default=False, help='Crop face or not')
    parser.add_argument('--FL_freeze', action='store_true', default=True, help='Freeze face_landback layer')
    parser.add_argument('--IR_freeze', action='store_true', default=True, help='Freeze ir_back layer')
    parser.add_argument('--PF_freeze', type=str, default=None, help='Freeze pyramid_fuse layer path (structure only)')
    parser.add_argument('--temporal_type', type=str, default='Transformer', help='Temporal type: MLP / CNN / Transformer')
    
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=int, default=123, help='torch.manual_seed')
    parser.add_argument('--save_dir', type=str, default=time.strftime("test_result_%Y%m%d-%H%M%S"), help='result save directory')

    return parser.parse_args()

def save_test_result(gt_labels, pre_labels, save_dir):
    """
    只繪製 Confusion Matrix 並儲存
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # --- 繪製 Confusion Matrix ---
    class_names = ['Dislike', 'Hesitate', 'Like']
    cm = confusion_matrix(gt_labels, pre_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Test Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(save_dir, 'test_confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion Matrix saved to {save_dir}")

def load_pretrained_weights(model, checkpoint):
    import collections
    # 處理 checkpoint 結構
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers = []
    
    # 檢查目前的 model 是否有 'module.' 前綴 (代表已經被 DataParallel 包裹)
    model_has_module = list(model_dict.keys())[0].startswith('module.')
    
    for k, v in state_dict.items():
        # 1. 先標準化 checkpoint 的 key (去掉 module.)
        pure_key = k[7:] if k.startswith('module.') else k
        
        # 2. 根據目前 model 的狀態，決定要匹配的 key
        if model_has_module:
            target_key = 'module.' + pure_key
        else:
            target_key = pure_key

        # 3. 進行匹配與形狀檢查
        if target_key in model_dict:
            if model_dict[target_key].size() == v.size():
                new_state_dict[target_key] = v
                matched_layers.append(target_key)
            else:
                print(f"Skipping {target_key}: size mismatch. Checkpoint: {v.size()}, Model: {model_dict[target_key].size()}")
        
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f'load_weight success: {len(matched_layers)} layers matched.')
    return model

def run_testing():
    args = parse_args()
    print("=" * 15 + " Testing Arguments " + "=" * 15)
    print(args)
    print("=" * 45)
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.save_dir = os.path.join('./test_results', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    # 紀錄 Log
    log_name = os.path.join(args.save_dir, 'test_log.txt')
    with open(log_name, 'w') as f:
        f.write(str(args) + '\n')
        f.write("------------------------------------------\n")

    # 定義 Transform (Testing 只需要 Resize/Normalize)
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 載入資料集
    num_classes = 7
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    
    elif args.dataset == "consumer":
        datapath = '/home/Dataset/face_croped_customer_ori2'
        test_anno = "all_test_list.txt"
        num_classes = 3
        test_dataset = AffectDatasetWindow_facecrop(datapath, anno=test_anno, window_size=args.window_size, stride=args.stride, transform=data_transforms_val, face_crop=args.face_crop, crop_method='yolo')
        
        print(f"Test file = {test_anno}\tsize = {str(test_dataset.__len__())}")
        
        # 即使是測試，模型結構初始化仍需這些參數，但不會用來訓練
        freeze_list = [args.FL_freeze, args.IR_freeze, args.PF_freeze]
        model = pyramid_trans_expr_window(img_size=224, num_classes=num_classes, window_size=args.window_size, type=args.modeltype, freeze_list=freeze_list, temporal_type=args.temporal_type)
    else:
        raise ValueError('Dataset name is not correct')

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # 載入權重
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    # if args.checkpoint:
    #     print(f"Loading pretrained weights from {args.checkpoint}...")
    #     if os.path.isfile(args.checkpoint):
    #         checkpoint = torch.load(args.checkpoint)
    #         # 處理 checkpoint 字典結構，有些存成 {'model_state_dict': ...}，有些直接是 state_dict
    #         if 'model_state_dict' in checkpoint:
    #             state_dict = checkpoint['model_state_dict']
    #         else:
    #             state_dict = checkpoint
            
    #         # 使用你的 utils 中的 load_pretrained_weights，或者直接 load_state_dict
    #         try:
    #             model = load_pretrained_weights(model, state_dict)
    #         except NameError:
    #              # 如果 utils 沒有 load_pretrained_weights，退回使用原生方法
    #             model.load_state_dict(state_dict, strict=False)
    #         print("Weight loaded successfully.")
    #     else:
    #         raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    # else:
    #     raise ValueError("Testing requires a checkpoint file! Use -c or --checkpoint.")

    # 開始測試
    model.eval()
    
    misclassified_counter = defaultdict(int)
    pre_labels = []
    gt_labels = []
    val_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        test_bar = tqdm(val_loader, desc="Testing", ncols=100)
        for img_frames, targets, frame_paths in test_bar:
            img_frames = img_frames.cuda()
            targets = targets.cuda()

            # Forward
            outputs, features = model(img_frames)
            
            probs = F.softmax(outputs, dim=1)
            _, predicts = torch.max(outputs, 1)

            # 收集結果
            pre_labels.extend(predicts.cpu().tolist())
            gt_labels.extend(targets.cpu().tolist())
            val_probs.extend(probs.cpu().tolist())

            # ---------------------------------------------------------
            # 紀錄錯誤樣本邏輯
            # ---------------------------------------------------------
            # frame_paths 是 tuple of tuples: ((path_b1_w1, path_b2_w1), (path_b1_w2, path_b2_w2)...)
            # 需要轉置成 (Batch, Window)
            batch_paths = list(zip(*frame_paths))
            
            batch_preds = predicts.cpu().tolist()
            batch_targets = targets.cpu().tolist()

            for j in range(len(batch_targets)):
                if batch_preds[j] != batch_targets[j]:
                    # 取得該 Window 第一張圖片的路徑作為識別
                    first_frame_path = batch_paths[j][0]
                    misclassified_counter[first_frame_path] += 1
            # ---------------------------------------------------------

    # 計算指標
    acc = accuracy_score(gt_labels, pre_labels)
    f1 = f1_score(gt_labels, pre_labels, average='macro')
    try:
        auc = roc_auc_score(gt_labels, val_probs, multi_class='ovr')
    except ValueError:
        auc = 0.0 # 避免類別不全導致報錯
        print("Warning: AUC calculation failed (likely missing classes in batch).")

    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*20 + " Test Results " + "="*20)
    print(f"Time Elapsed: {elapsed:.2f} min")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("="*54)

    # 寫入 Log
    with open(log_name, 'a') as f:
        f.write(f"\nTest Finished.\nAccuracy: {acc:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}\n")
        f.write(classification_report(gt_labels, pre_labels, digits=4))

    # 儲存錯誤報告
    df_error = pd.DataFrame(list(misclassified_counter.items()), columns=['file_path', 'error_count'])
    df_error = df_error.sort_values(by='error_count', ascending=False)
    csv_path = os.path.join(args.save_dir, 'misclassified_samples.csv')
    df_error.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Misclassification report saved to {csv_path}")

    # 儲存圖片
    save_test_result(gt_labels, pre_labels, args.save_dir)

if __name__ == "__main__":
    run_testing()
