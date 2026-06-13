import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import os
import torch
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# 引用你的模組
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class_2
from models.emotion_hyp_affect import pyramid_trans_expr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet8class', help='dataset: rafdb, affectnet, affectnet8class')
    # Checkpoint 改為必要參數
    parser.add_argument('-c', '--checkpoint', type=str, default='/home/lab702/POSTER/checkpoint/20260115-160210/epoch52_acc0.82425.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers')
    
    # Model config
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze pyramid_fuse layer or not')
    
    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
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
    # class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Like', 'Hesitate', 'Anger', 'Dislike']
    class_names = ['Like', 'Hesitate', 'Dislike', 'back_head']
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
    args.save_dir = os.path.join('./test_results', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])
    
    # 紀錄 Log
    log_name = os.path.join(args.save_dir, 'test_log.txt')
    with open(log_name, 'w') as f:
        f.write(str(args) + '\n')

    # Testing Transform
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
        datapath = './data/AffectNet_customer_2class/'
        num_classes = 4
        test_dataset = Affectdataset_8class_2(datapath, train=False, transform=data_transforms_val, face_crop=True, crop_method='mtcnn')
        # 注意: 雖然是測試，但為了保持結構一致，freeze 參數仍需傳入，儘管 eval 模式下 dropout/bn 行為不同
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze)

    else:
        raise ValueError('Dataset name is not correct')

    print('Test set size:', test_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # -----------------------------------------------------------
    # 修正重點：先載入權重，再上 DataParallel
    # -----------------------------------------------------------
    model = model.cuda()

    if args.checkpoint:
        print(f"Loading pretrained weights from {args.checkpoint}...")
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            
            # 處理 checkpoint 字典結構
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 載入權重 (此時 model 尚未被 DataParallel 包裹，key 不會有 module. 前綴問題)
            try:
                model = load_pretrained_weights(model, state_dict)
            except NameError:
                model.load_state_dict(state_dict, strict=False)
            
            print("Weight loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    else:
        raise ValueError("Checkpoint file is required for testing.")

    # 權重載入完成後，再啟用 DataParallel
    model = torch.nn.DataParallel(model)

    # 開始測試
    pre_labels = []
    gt_labels = []
    val_probs = []
    
    model.eval()
    start_time = time.time()
    
    print("Start Inference...")
    with torch.no_grad():
        test_bar = tqdm(val_loader, desc="Testing", ncols=100)
        for imgs, targets in test_bar:
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs, features = model(imgs)
            
            probs = F.softmax(outputs, dim=1)
            _, predicts = torch.max(outputs, 1)

            pre_labels.extend(predicts.cpu().tolist())
            gt_labels.extend(targets.cpu().tolist())
            val_probs.extend(probs.cpu().tolist())

    # 計算指標
    acc = accuracy_score(gt_labels, pre_labels)
    f1 = f1_score(gt_labels, pre_labels, average='macro')
    try:
        auc = roc_auc_score(gt_labels, val_probs, multi_class='ovr')
    except ValueError:
        auc = 0.0
        print("Warning: AUC calculation failed.")

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
        f.write("\n=== Classification Report ===\n")
        f.write(classification_report(gt_labels, pre_labels, digits=4))

    # 儲存結果圖表
    save_test_result(gt_labels, pre_labels, args.save_dir)

if __name__ == "__main__":
    run_testing()
