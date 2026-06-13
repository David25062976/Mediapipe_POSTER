# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0

# export __GLX_VENDOR_LIBRARY_NAME=nvidia

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
import torch.utils.data as data
from torchvision import transforms
import os
# os.environ['DISPLAY'] = ':0'
import torch
import argparse
from tqdm import tqdm
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class_pt import Affectdataset_8class_2
from data_preprocessing.dataset_window_pt import AffectDatasetWindow_pt
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_preprocessing.sam import SAM
from torchsampler import ImbalancedDatasetSampler
from models.emotion_hyp_affect_mediapipe_pt import pyramid_trans_expr
from models.emotion_hyp_window_pt import pyramid_trans_expr_window_pt, load_single_frame_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet8class_shopping_window', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0000015, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=999, help='Tolerating no progress for n epochs.')

    parser.add_argument('--window_size', type=int, default=8, help='Window size of sliding window')
    parser.add_argument('--stride', type=int, default=8, help='Stride of sliding window')
    parser.add_argument('--face_crop', action='store_true', default=False, help='Crop face or not')
    parser.add_argument('--single_frame_weight', type=str, default='./checkpoint/20260501-060739_Shopping_196_14/best.pth', help='Single frame model weight with face_landback, ir_back, pyramid_fuse')
    parser.add_argument('--FL_freeze', action='store_true', default=True, help='Freeze face_landback layer or not')
    parser.add_argument('--IR_freeze', action='store_true', default=True, help='Freeze ir_back layer or not')
    parser.add_argument('--PF_freeze', action='store_true', default=True, help='Freeze pyramid_fuse layer or not')
    
    parser.add_argument('--temporal_type', type=str, default='MLP', help='Temporal type: MLP / CNN / Transformer')

    parser.add_argument('--mediapipe_points', type=int, default=196, help='49 or 196')
    parser.add_argument('--mediapipe_patch_size', type=int, default=14, help='14 or 24')

    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=str, default=123, help='torch.manual_seed({seed})')
    parser.add_argument('--save_dir', type=str, default=time.strftime("window_%Y%m%d-%H%M%S"), help='model save directory')
    parser.add_argument('--remark', type=str, default="remark", help='remark')

    return parser.parse_args()

def save_result(train_results, val_results, gt_labels, pre_labels, save_dir, epoch, class_names):
    """
    train_results/val_results: 包含 'acc', 'f1', 'auc' 列表的字典
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # --- 1. 繪製訓練趨勢圖 (Acc, F1, AUC) ---
    epochs = range(1, len(train_results['acc']) + 1)
    metrics = ['acc', 'f1', 'auc']
    plt.figure(figsize=(18, 5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        plt.plot(epochs, train_results[metric], label=f'Train {metric.upper()}')
        plt.plot(epochs, val_results[metric], label=f'Val {metric.upper()}')
        plt.title(f'Training and Validation {metric.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.upper())
        plt.ylim(0.4, 1.0)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if epoch == -1:
        plt.savefig(os.path.join(save_dir, f'final_metrics.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'metrics_epoch_{epoch}.png'))
    plt.close()

    # --- 2. 繪製 Confusion Matrix ---
    # class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Like', 'Hesitate', 'Anger', 'Dislike']
    cm = confusion_matrix(gt_labels, pre_labels)
    
    # 1. 計算每個格子的數量與百分比字串
    counts = [f"{value:d}" for value in cm.flatten()]

    # 計算每一列 (Actual) 的總和，並處理除以 0 的情況
    row_sums = np.sum(cm, axis=1, keepdims=True)
    # 將每個格子除以它所在列的總和
    row_percentages = cm / np.where(row_sums == 0, 1, row_sums) 
    percentages = [f"{value:.1%}" for value in row_percentages.flatten()]
    
    # 2. 將數量與百分比合併，並加上換行符號 (\n)
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(cm.shape)

    plt.figure(figsize=(10, 8))
    
    # 3. 將 labels 傳給 annot，並將 fmt 設為空字串 '' (因為我們已經自訂好字串格式了)
    sns.heatmap(row_percentages, annot=labels, fmt='', cmap='Blues',
                vmin=0.0, vmax=1.0,
                xticklabels=class_names, 
                yticklabels=class_names)
                
    if epoch == -1:
        plt.title(f'Final Confusion Matrix')
    else:
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if epoch == -1:
        plt.savefig(os.path.join(save_dir, f'final_cm.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch}.png'))
    
    plt.close()
    
    print(f"Figures saved to {save_dir}")

def save_misclassified_report(counter_dict, save_path):
    if not counter_dict:
        return
    error_list = []
    for path, data in counter_dict.items():
        row = {'file_path': path}
        row.update(data)
        error_list.append(row)
        
    df_error = pd.DataFrame(error_list)
    df_error = df_error.sort_values(by='total_errors', ascending=False)
    df_error.to_csv(save_path, index=False, encoding='utf-8-sig')

def run_training():
    args = parse_args()
    print("=" * 15 + " args " + "=" * 15)
    print(args)
    print("=" * 36)
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join('./checkpoint', args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        
    log_file_path = os.path.join(args.save_dir, 'log.txt')
    with open(log_file_path, 'w') as f:
        f.write("=" * 15 + " Training Parameters " + "=" * 15 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("=" * 51 + "\n\n")
        f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1  | Notes\n")
        f.write("-" * 75 + "\n")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Like', 'Hesitate', 'Anger', 'Dislike']
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        train_dataset = RafDataSet(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        train_dataset = Affectdataset(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class_mediapipe":
        train_data_dir = './data/AffectNet_customer_2class/train_set_mediapipe'
        train_landmarks_db = './data/AffectNet_customer_2class/train_set_mediapipe_landmarks_all.pt'
        valid_data_dir = './data/AffectNet_customer_2class/valid_set_mediapipe'
        valid_landmarks_db = './data/AffectNet_customer_2class/valid_set_mediapipe_landmarks_all.pt'
        # num_classes = 3
        # class_names = ['interested', 'thinking', 'pass']
        class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interesting', 'thinking', 'Anger', 'passing']
        num_classes = len(class_names)
        train_dataset = Affectdataset_8class_2(train_data_dir, train_landmarks_db, train=True, transform=data_transforms, basic_aug=True, point=args.mediapipe_points)
        val_dataset = Affectdataset_8class_2(valid_data_dir, valid_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size)

    elif args.dataset == "affectnet8class_shopping":
        train_data_dir = './data/AffectNet8c_Shopping/train_set'
        train_landmarks_db = './data/AffectNet8c_Shopping/train_set_landmarks_all.pt'
        valid_data_dir = './data/AffectNet8c_Shopping/valid_set'
        valid_landmarks_db = './data/AffectNet8c_Shopping/valid_set_landmarks_all.pt'
        # num_classes = 3
        # class_names = ['interested', 'thinking', 'pass']
        class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interesting', 'thinking', 'buying', 'passing']
        num_classes = len(class_names)
        train_dataset = Affectdataset_8class_2(train_data_dir, train_landmarks_db, train=True, transform=data_transforms, basic_aug=True, point=args.mediapipe_points)
        val_dataset = Affectdataset_8class_2(valid_data_dir, valid_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size)
    
    elif args.dataset == "affectnet8class_ori":
        train_data_dir = './data/AffectNet_customer/train_set/train_set_AffectNet'
        train_landmarks_db = './data/AffectNet_customer/train_set/train_set_AffectNet_landmarks_all.pt'
        valid_data_dir = './data/AffectNet_customer/valid_set/valid_set_AffectNet'
        valid_landmarks_db = './data/AffectNet_customer/valid_set/valid_set_AffectNet_landmarks_all.pt'
        # num_classes = 3
        # class_names = ['interested', 'thinking', 'pass']
        class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
        num_classes = len(class_names)
        train_dataset = Affectdataset_8class_2(train_data_dir, train_landmarks_db, train=True, transform=data_transforms, basic_aug=True, point=args.mediapipe_points)
        val_dataset = Affectdataset_8class_2(valid_data_dir, valid_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size)
    
    elif args.dataset == "affectnet8class_shopping_window":
        datapath = './data/5_classes'

        train_anno = "train_list.txt"
        train_landmarks_db = './data/5_classes/train_landmarks.pt'

        valid_anno = "valid_list.txt"
        valid_landmarks_db = './data/5_classes/valid_landmarks.pt'

        class_names = ['browsing', 'interesting', 'thinking', 'buying', 'passing']
        num_classes = len(class_names)
        train_dataset = AffectDatasetWindow_pt(datapath, anno=train_anno, landmarks_db=train_landmarks_db, window_size=args.window_size, stride=args.stride, transform=data_transforms)
        val_dataset = AffectDatasetWindow_pt(datapath, anno=valid_anno, landmarks_db=valid_landmarks_db, window_size=args.window_size, stride=args.stride, transform=data_transforms_val)
        
        with open(log_file_path, 'a') as f:
            f.write(f"Train file = {train_anno}\tsize = {str(train_dataset.__len__())}\n")
            f.write(f"Test file = {valid_anno}\tsize = {str(val_dataset.__len__())}\n")

        freeze_list = [args.FL_freeze, args.IR_freeze, args.PF_freeze]
        model = pyramid_trans_expr_window_pt(img_size=224, num_classes=num_classes, window_size=args.window_size, type=args.modeltype, freeze_list=freeze_list, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size, temporal_type=args.temporal_type)

    else:
        return print('dataset name is not correct')

    val_num = val_dataset.__len__()
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)
    
    if args.single_frame_weight:
        model = load_single_frame_weights(model, args.single_frame_weight)
    
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    params = model.parameters()
    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    # 找出需要訓練的參數
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # 只把這些參數交給 SAM
    optimizer = SAM(trainable_params, base_optimizer, lr=args.lr, rho=0.05, adaptive=False)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    train_acc_result, train_f1_result, train_auc_result = [], [], []
    val_acc_result, val_f1_result, val_auc_result = [], [], []

    best_acc = 0
    best_epoch = 0
    epoch_times = []
    patience = args.early_stop_patience  # 容忍 early_stop_patience 個 epoch 沒有進步
    counter = 0

    train_misclassified_counter = {}
    valid_misclassified_counter = {}

    for i in range(1, args.epochs + 1):
        train_pre_labels, train_gt_labels, train_probs = [], [], []
        train_loss, correct_sum, iter_cnt = 0.0, 0, 0
        best_train_misclassified_counter = {}
        best_valid_misclassified_counter = {}
        start_time = time.time()
        model.train()

        train_loader_tqdm = tqdm(train_loader, desc=f"[Epoch {i}/{args.epochs}] Train", unit="batch")

        # for batch_i, (imgs, targets) in enumerate(train_loader):
        for batch_i, (imgs, targets, coords, frame_paths) in enumerate(train_loader_tqdm):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()
            coords = coords.cuda()
            # SAM Optimizer first
            outputs, features = model(imgs, coords)

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM Optimizer second
            outputs, features = model(imgs, coords)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            loss.backward()
            optimizer.second_step(zero_grad=True)

            # 收集數據用於計算指標
            probs = F.softmax(outputs, dim=1)
            _, predicts = torch.max(outputs, 1)
            
            train_pre_labels.extend(predicts.cpu().tolist())
            train_gt_labels.extend(targets.cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            correct_sum += torch.eq(predicts, targets).sum()
            train_loss += loss.item()

            # ---------------------------------------------------------
            # 詳細紀錄 Train 誤判樣本
            # ---------------------------------------------------------
            batch_paths = list(zip(*frame_paths))
            batch_preds = predicts.cpu().tolist()
            batch_targets = targets.cpu().tolist()

            for j in range(len(batch_targets)):
                if batch_preds[j] != batch_targets[j]:
                    first_frame_path = batch_paths[j][0] 
                    true_class = class_names[batch_targets[j]]
                    pred_class = class_names[batch_preds[j]]
                    
                    if first_frame_path not in train_misclassified_counter:
                        # 若尚未紀錄過此路徑，初始化結構
                        train_misclassified_counter[first_frame_path] = {
                            'true_label': true_class, 
                            'total_errors': 0
                        }
                        for c in class_names:
                            train_misclassified_counter[first_frame_path][f'pred_as_{c}'] = 0
                            
                    # 更新錯誤次數
                    train_misclassified_counter[first_frame_path]['total_errors'] += 1
                    train_misclassified_counter[first_frame_path][f'pred_as_{pred_class}'] += 1

                    if first_frame_path not in best_train_misclassified_counter:
                        # 若尚未紀錄過此路徑，初始化結構
                        best_train_misclassified_counter[first_frame_path] = {
                            'true_label': true_class, 
                            'total_errors': 0
                        }
                        for c in class_names:
                            best_train_misclassified_counter[first_frame_path][f'pred_as_{c}'] = 0
                            
                    # 更新錯誤次數
                    best_train_misclassified_counter[first_frame_path]['total_errors'] += 1
                    best_train_misclassified_counter[first_frame_path][f'pred_as_{pred_class}'] += 1
            # ---------------------------------------------------------

            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        # 計算 Epoch 指標
        t_acc = correct_sum.float() / len(train_dataset)
        t_f1 = f1_score(train_gt_labels, train_pre_labels, average='macro')
        # Multi-class AUC 需要 specify multi_class='ovr'
        t_auc = roc_auc_score(train_gt_labels, train_probs, multi_class='ovr')
        
        train_acc_result.append(t_acc.item())
        train_f1_result.append(t_f1)
        train_auc_result.append(t_auc)

        train_loss = train_loss / iter_cnt
        elapsed = (time.time() - start_time) / 60
        epoch_times.append(elapsed)

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Training F1:%.4f. Training AUC:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, t_acc, t_f1, t_auc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        # ==========================================
        # 強制釋放最後一個訓練 Batch 的殘留顯存
        # ==========================================
        try:
            del imgs, targets, coords, outputs, features, loss, CE_loss, lsce_loss
        except NameError:
            pass
        torch.cuda.empty_cache() 
        # ==========================================

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            val_pre_labels, val_gt_labels, val_probs = [], [], []
            start_time = time.time()
            model.eval()

            val_loader_tqdm = tqdm(val_loader, desc=f"[Epoch {i}/{args.epochs}] Val", unit="batch")

            # for batch_i, (imgs, targets) in enumerate(val_loader):
            for batch_i, (imgs, targets, coords, frame_paths) in enumerate(val_loader_tqdm):
                imgs = imgs.cuda()
                targets = targets.cuda()
                coords = coords.cuda()

                outputs, features = model(imgs, coords)

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss.item()
                iter_cnt += 1
                probs = F.softmax(outputs, dim=1)
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

                val_pre_labels.extend(predicts.cpu().tolist())
                val_gt_labels.extend(targets.cpu().tolist())
                val_probs.extend(probs.cpu().tolist())

                # ---------------------------------------------------------
                # 詳細紀錄 Valid 誤判樣本
                # ---------------------------------------------------------
                batch_paths = list(zip(*frame_paths))
                batch_preds = predicts.cpu().tolist()
                batch_targets = targets.cpu().tolist()

                for j in range(len(batch_targets)):
                    if batch_preds[j] != batch_targets[j]:
                        first_frame_path = batch_paths[j][0] 
                        true_class = class_names[batch_targets[j]]
                        pred_class = class_names[batch_preds[j]]
                        
                        if first_frame_path not in valid_misclassified_counter:
                            valid_misclassified_counter[first_frame_path] = {
                                'true_label': true_class, 
                                'total_errors': 0
                            }
                            for c in class_names:
                                valid_misclassified_counter[first_frame_path][f'pred_as_{c}'] = 0
                                
                        valid_misclassified_counter[first_frame_path]['total_errors'] += 1
                        valid_misclassified_counter[first_frame_path][f'pred_as_{pred_class}'] += 1

                        if first_frame_path not in best_valid_misclassified_counter:
                            best_valid_misclassified_counter[first_frame_path] = {
                                'true_label': true_class, 
                                'total_errors': 0
                            }
                            for c in class_names:
                                best_valid_misclassified_counter[first_frame_path][f'pred_as_{c}'] = 0
                                
                        best_valid_misclassified_counter[first_frame_path]['total_errors'] += 1
                        best_valid_misclassified_counter[first_frame_path][f'pred_as_{pred_class}'] += 1
                # ---------------------------------------------------------

                val_loader_tqdm.set_postfix(v_loss=f"{loss.item():.4f}")

            val_loss = val_loss / iter_cnt
            v_acc = accuracy_score(val_gt_labels, val_pre_labels) # 或使用原本的 bingo_cnt 邏輯
            v_f1 = f1_score(val_gt_labels, val_pre_labels, average='macro')
            v_auc = roc_auc_score(val_gt_labels, val_probs, multi_class='ovr')
            total_socre = 0.67 * v_f1 + 0.33 * v_acc

            val_acc_result.append(v_acc)
            val_f1_result.append(v_f1)
            val_auc_result.append(v_auc)

            elapsed = (time.time() - start_time) / 60
            print("[Epoch %d] Validation time:%.2f, Validation accuracy:%.4f. Validation F1:%.4f. Validation AUC:%.4f. Loss:%.3f score %4f" % (
            i, elapsed, v_acc, v_f1, v_auc, val_loss, total_socre))

        is_best = False
        if v_acc > best_acc:
            best_acc = v_acc
            best_epoch = i
            is_best = True
            counter = 0
            print(f"New best_acc: {best_acc:.4f} at epoch {i}")
            
            # 固定儲存為 best.pth
            torch.save({'iter': i,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                        os.path.join(args.save_dir, "best.pth"))
            print('Model saved as best.pth.')

            # 更新最佳狀態的圖表
            train_dict = {'acc': train_acc_result, 'f1': train_f1_result, 'auc': train_auc_result}
            val_dict = {'acc': val_acc_result, 'f1': val_f1_result, 'auc': val_auc_result}
            save_result(
                train_results=train_dict,
                val_results=val_dict,
                gt_labels=val_gt_labels,
                pre_labels=val_pre_labels,
                save_dir=args.save_dir,
                epoch=i,
                class_names=class_names
            )

            save_misclassified_report(best_train_misclassified_counter, os.path.join(args.save_dir, 'best_train_misclassified_samples.csv'))
            save_misclassified_report(best_valid_misclassified_counter, os.path.join(args.save_dir, 'best_valid_misclassified_samples.csv'))
            print("Best model updated. Misclassification report saved.")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

        # 寫入單行 Epoch 結果到 log.txt
        log_str = (f"{i:03d}   | {train_loss:.4f}     | {t_acc:.4f}    | {val_loss:.4f}   | {v_acc:.4f}   | {v_f1:.4f} |")
        if is_best:
            log_str += " <--- [BEST]"
            
        with open(log_file_path, 'a') as f:
            f.write(log_str + "\n")
        
        save_misclassified_report(train_misclassified_counter, os.path.join(args.save_dir, 'train_misclassified_samples.csv'))
        save_misclassified_report(valid_misclassified_counter, os.path.join(args.save_dir, 'valid_misclassified_samples.csv'))
        print("Train/Valid misclassification report saved.")

        # Early stopping 觸發
        if counter >= patience:
            print("Early stopping triggered!")

            train_dict = {'acc': train_acc_result, 'f1': train_f1_result, 'auc': train_auc_result}
            val_dict = {'acc': val_acc_result, 'f1': val_f1_result, 'auc': val_auc_result}
            save_result(
                train_results=train_dict,
                val_results=val_dict,
                gt_labels=val_gt_labels,
                pre_labels=val_pre_labels,
                save_dir=args.save_dir,
                epoch=-1,
                class_names=class_names
            )
            with open(log_file_path, 'a') as f:
                f.write(f"\nTraining stopped early at epoch {i}. Best Val Acc: {best_acc:.4f} at epoch {best_epoch}.\n")
            break

    if counter < patience:
        
        train_dict = {'acc': train_acc_result, 'f1': train_f1_result, 'auc': train_auc_result}
        val_dict = {'acc': val_acc_result, 'f1': val_f1_result, 'auc': val_auc_result}
        save_result(
            train_results=train_dict,
            val_results=val_dict,
            gt_labels=val_gt_labels,
            pre_labels=val_pre_labels,
            save_dir=args.save_dir,
            epoch=-1,
            class_names=class_names
        )
        print(f'Final plots saved.')
        
        # 新增：紀錄訓練完成
        with open(log_file_path, 'a') as f:
            f.write(f"\nTraining finished. Best Val Acc: {best_acc:.4f} at epoch {best_epoch}.\n")

if __name__ == "__main__":
    run_training()
