# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
# export __GLX_VENDOR_LIBRARY_NAME=nvidia

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_preprocessing.sam import SAM
from torchsampler import ImbalancedDatasetSampler

# Models
from models.emotion_hyp_affect_mediapipe_pt_1_1 import pyramid_trans_expr as pyramid_trans_expr_1_1
from models.emotion_hyp_affect_mediapipe_pt_1_2 import pyramid_trans_expr as pyramid_trans_expr_1_2
from models.emotion_hyp_affect_mediapipe_pt_1_3 import pyramid_trans_expr as pyramid_trans_expr_1_3
from models.emotion_hyp_affect_mediapipe_pt_1_4 import pyramid_trans_expr as pyramid_trans_expr_1_4
from models.emotion_hyp_affect_mediapipe_pt_1_5 import pyramid_trans_expr as pyramid_trans_expr_1_5
from models.emotion_hyp_affect_mediapipe_pt_attention_map_p2 import pyramid_trans_expr as pyramid_trans_expr_attention_map
from models.emotion_hyp_affect_mediapipe_pt_attention_map_p2 import load_pretrained_weights as load_pretrained_weights_print
from models.emotion_hyp_affect_mediapipe_pt import pyramid_trans_expr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet8class_shopping', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/20260605-012234/best.pth', help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')

    parser.add_argument('--exp_model', type=str, default='1-6', help='Experiment model choose')

    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=999, help='Tolerating no progress for n epochs.')

    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze ir layer or not')

    parser.add_argument('--mediapipe_points', type=int, default=478, help='49 or 196')
    parser.add_argument('--mediapipe_patch_size', type=int, default=14, help='14 or 24')

    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=str, default=123, help='torch.manual_seed({seed})')
    parser.add_argument('--save_dir', type=str, default=time.strftime("%Y%m%d-%H%M%S"), help='model save directory')
    parser.add_argument('--remark', type=str, default="remark", help='attention map Phase 2')
    
    # 新增提取權重的參數
    parser.add_argument('--extract_weights', action='store_true', default=False, help='Extract and save global attention weights after training/loading checkpoint.')
    parser.add_argument('--dynamic_droptoken', action='store_true', default=True, help='Enable dynamic drop token training (Phase 2)')
    parser.add_argument('--importance_order', type=str, default='./checkpoint/20260605-012234/symmetrized_global_points_importance_order.pt', help='Path to the importance order pt file')

    return parser.parse_args()

def save_result(train_results, val_results, gt_labels, pre_labels, save_dir, epoch, class_names):
    # (此部分保持不變)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

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
        plt.savefig(os.path.join(save_dir, 'final_metrics.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'metrics_epoch_{epoch}.png'))
    plt.close()

    cm = confusion_matrix(gt_labels, pre_labels)
    counts = [f"{value:d}" for value in cm.flatten()]

    row_sums = np.sum(cm, axis=1, keepdims=True)
    row_percentages = cm / np.where(row_sums == 0, 1, row_sums) 
    percentages = [f"{value:.1%}" for value in row_percentages.flatten()]
    
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(cm.shape)

    plt.figure(figsize=(10, 8))
    sns.heatmap(row_percentages, annot=labels, fmt='', cmap='Blues',
                vmin=0.0, vmax=1.0,
                xticklabels=class_names, 
                yticklabels=class_names)
                
    if epoch == -1:
        plt.title('Final Confusion Matrix')
    else:
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if epoch == -1:
        plt.savefig(os.path.join(save_dir, 'final_cm.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch}.png'))
    plt.close()

# 新增：提取並排序注意力權重的函數
def extract_and_sort_points(model, val_loader, device, save_dir):
    model.eval()
    all_attention_weights = []

    print("\n" + "="*40)
    print("開始提取注意力權重 (Attention Maps)...")
    print("="*40)

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Extracting Weights", unit="batch")
        for batch_i, (imgs, targets, coords) in enumerate(val_loader_tqdm):
            imgs = imgs.to(device)
            coords = coords.to(device)
            
            # 使用 return_attention=True 獲取 attention maps
            outputs, features, attn_maps = model(imgs, coords, return_attention=True)
            
            batch_layer_weights = []
            
            for layer_idx, block_attn in enumerate(attn_maps):
                # 解包 tuple: (attn_img_map, attn_lm_map)
                if isinstance(block_attn, tuple) or isinstance(block_attn, list):
                    attn_img = block_attn[0]
                else:
                    attn_img = block_attn
                
                # CLS token 對其他點位的注意力 (Index 0 對 Index 1:)
                # shape: [B, Heads, Points]
                cls_attention = attn_img[:, :, 0, 1:] 
                
                # 將各個 Head 的注意力取平均
                cls_attention_mean_heads = cls_attention.mean(dim=1) # shape: [B, Points]
                batch_layer_weights.append(cls_attention_mean_heads)
            
            # 將所有 Layer 的權重取平均
            batch_layer_weights = torch.stack(batch_layer_weights, dim=0) # [Layers, B, Points]
            avg_attention_across_layers = batch_layer_weights.mean(dim=0) # [B, Points]
            
            all_attention_weights.append(avg_attention_across_layers.cpu())

    # 彙總整個 Validation Set 並對所有 batch 取平均
    global_attention = torch.cat(all_attention_weights, dim=0) # [Total_B, Points]
    global_point_importance = global_attention.mean(dim=0)     # [Points] 
    
    # 根據重要性進行排序 (由大到小)
    sorted_importance, sorted_indices = torch.sort(global_point_importance, descending=True)
    
    print("\n提取完成！")
    print(f"最重要前 49 個點的 Index:\n{sorted_indices[:49].tolist()}")
    
    save_path = os.path.join(save_dir, "global_points_importance_order.pt")
    torch.save(sorted_indices, save_path)
    print(f"注意力排序清單已成功儲存至: {save_path}")
    
    return sorted_indices

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # (Dataset 設定省略...與你原本一樣，不更動)
    test_dataset = None

    if args.dataset == "affectnet8class_shopping":
        train_data_dir = './data/AffectNet8c_Shopping_test/train_set'
        train_landmarks_db = './data/AffectNet8c_Shopping_test/train_set_landmarks_all.pt'
        valid_data_dir = './data/AffectNet8c_Shopping_test/valid_set'
        valid_landmarks_db = './data/AffectNet8c_Shopping_test/valid_set_landmarks_all.pt'
        test_data_dir = './data/AffectNet8c_Shopping_test/test_set'
        test_landmarks_db = './data/AffectNet8c_Shopping_test/test_set_landmarks_all.pt'
        
        class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interesting', 'thinking', 'buying', 'passing']
        num_classes = len(class_names)
        
        train_dataset = Affectdataset_8class_2(train_data_dir, train_landmarks_db, train=True, transform=data_transforms, basic_aug=True, point=args.mediapipe_points)
        val_dataset = Affectdataset_8class_2(valid_data_dir, valid_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
        if os.path.exists(test_data_dir):
            test_dataset = Affectdataset_8class_2(test_data_dir, test_landmarks_db, train=False, transform=data_transforms_val, point=args.mediapipe_points)
            
        model = pyramid_trans_expr_attention_map(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_points=args.mediapipe_points, mediapipe_patch_size=args.mediapipe_patch_size)
    else:
        return print('Please configure dataset in the code.')

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
                                             
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.val_batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=False,
                                                  pin_memory=True)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights_print(model, checkpoint)
        print("checkpoint weight loaded")
        
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

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
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
    patience = args.early_stop_patience  
    counter = 0

    # ==========================================
    # 準備 Phase 2：載入全域重要性排序表
    # ==========================================
    global_importance_order = None
    if args.dynamic_droptoken:
        print(f"Loading importance order from {args.importance_order}...")
        global_importance_order = torch.load(args.importance_order).cuda()
        print("Dynamic DropToken Training Enabled! N will vary between 49 and 478.")

    # =============== 訓練迴圈 ===============
    for i in range(1, args.epochs + 1):
        train_pre_labels, train_gt_labels, train_probs = [], [], []
        train_loss, correct_sum, iter_cnt = 0.0, 0, 0
        start_time = time.time()
        model.train()

        train_loader_tqdm = tqdm(train_loader, desc=f"[Epoch {i}/{args.epochs}] Train", unit="batch")

        for batch_i, (imgs, targets, coords) in enumerate(train_loader_tqdm):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()
            coords = coords.cuda()

            # ★ 動態決定這個 Batch 要保留的點位數
            active_indices = None
            if args.dynamic_droptoken and global_importance_order is not None:
                import random
                # 隨機選擇 N 介於 49 到 478 之間
                N = random.randint(49, 478)
                
                # 取出前 N 個最重要的點
                top_n_indices = global_importance_order[:N]
                
                # 【關鍵】：重新由小到大排序，保留臉部空間的拓樸相對順序
                active_indices, _ = torch.sort(top_n_indices)

            # SAM Optimizer first 
            outputs, features = model(imgs, coords, active_indices=active_indices)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM Optimizer second
            outputs, features = model(imgs, coords, active_indices=active_indices)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            loss.backward()
            optimizer.second_step(zero_grad=True)

            probs = F.softmax(outputs, dim=1)
            _, predicts = torch.max(outputs, 1)
            
            train_pre_labels.extend(predicts.cpu().tolist())
            train_gt_labels.extend(targets.cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            correct_sum += torch.eq(predicts, targets).sum()
            train_loss += loss.item()

            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

        t_acc = correct_sum.float() / len(train_dataset)
        t_f1 = f1_score(train_gt_labels, train_pre_labels, average='macro')
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

        try:
            del imgs, targets, coords, outputs, features, loss, CE_loss, lsce_loss
        except NameError:
            pass
        torch.cuda.empty_cache() 

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

            for batch_i, (imgs, targets, coords) in enumerate(val_loader_tqdm):
                imgs = imgs.cuda()
                targets = targets.cuda()
                coords = coords.cuda()

                outputs, features = model(imgs, coords, active_indices=None)

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

                val_loader_tqdm.set_postfix(v_loss=f"{loss.item():.4f}")

            val_loss = val_loss / iter_cnt
            v_acc = accuracy_score(val_gt_labels, val_pre_labels)
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
            
            torch.save({'iter': i,
                        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join(args.save_dir, "best.pth"))
            print('Model saved as best.pth.')

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
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")

        log_str = (f"{i:03d}   | {train_loss:.4f}     | {t_acc:.4f}    | {val_loss:.4f}   | {v_acc:.4f}   | {v_f1:.4f} |")
        if is_best:
            log_str += " <--- [BEST]"
            
        with open(log_file_path, 'a') as f:
            f.write(log_str + "\n")

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

    # 確保正常訓練完也有 final plot 存檔
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
        
        with open(log_file_path, 'a') as f:
            f.write(f"\nTraining finished. Best Val Acc: {best_acc:.4f} at epoch {best_epoch}.\n")

    # ==========================================
    # 最終測試階段 (Testing Phase)
    # ==========================================
    if test_dataset:
        print("\n" + "="*40)
        print("開始進行 Test 資料集測試 (使用最佳權重 best.pth)")
        print("="*40)

        best_model_path = os.path.join(args.save_dir, "best.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("成功載入 best.pth 權重。")
        else:
            print("找不到 best.pth，使用最後一個 epoch 的權重進行測試。")

        model.eval()
        test_loss = 0.0
        test_iter_cnt = 0
        test_pre_labels, test_gt_labels, test_probs = [], [], []

        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader, desc="Testing", unit="batch")
            for batch_i, (imgs, targets, coords) in enumerate(test_loader_tqdm):
                imgs = imgs.cuda()
                targets = targets.cuda()
                coords = coords.cuda()

                outputs, features = model(imgs, coords)
                loss = CE_criterion(outputs, targets)

                test_loss += loss.item()
                test_iter_cnt += 1
                probs = F.softmax(outputs, dim=1)
                _, predicts = torch.max(outputs, 1)

                test_pre_labels.extend(predicts.cpu().tolist())
                test_gt_labels.extend(targets.cpu().tolist())
                test_probs.extend(probs.cpu().tolist())

        test_loss = test_loss / test_iter_cnt
        test_acc = accuracy_score(test_gt_labels, test_pre_labels)
        test_f1 = f1_score(test_gt_labels, test_pre_labels, average='macro')
        test_auc = roc_auc_score(test_gt_labels, test_probs, multi_class='ovr')

        print(f"\n[Test Result] Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Loss: {test_loss:.4f}")

        # 寫入 log
        with open(log_file_path, 'a') as f:
            f.write("\n" + "="*40 + "\n")
            f.write(f"Test Result using best.pth:\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Acc:  {test_acc:.4f}\n")
            f.write(f"Test F1:   {test_f1:.4f}\n")
            f.write(f"Test AUC:  {test_auc:.4f}\n")
            f.write("="*40 + "\n")

        # 繪製專屬的 Test Confusion Matrix
        test_cm = confusion_matrix(test_gt_labels, test_pre_labels)
        counts = [f"{value:d}" for value in test_cm.flatten()]
        row_sums = np.sum(test_cm, axis=1, keepdims=True)
        row_percentages = test_cm / np.where(row_sums == 0, 1, row_sums)
        percentages = [f"{value:.1%}" for value in row_percentages.flatten()]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(counts, percentages)]
        labels = np.asarray(labels).reshape(test_cm.shape)

        plt.figure(figsize=(10, 8))
        sns.heatmap(row_percentages, annot=labels, fmt='', cmap='Blues',
                    vmin=0.0, vmax=1.0,
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Test Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(args.save_dir, 'test_cm.png'))
        plt.close()
        print(f"Test Confusion Matrix saved to {os.path.join(args.save_dir, 'test_cm.png')}")
            
    # =============== 權重提取階段 (Phase 1) ===============
    if args.extract_weights:
        # 確保載入的是收斂後最好的模型
        best_model_path = os.path.join(args.save_dir, "best.pth")
        if os.path.exists(best_model_path):
            print("\n載入 best.pth 以進行權重提取...")
            checkpoint = torch.load(best_model_path)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        
        # 執行權重提取
        extract_and_sort_points(model, val_loader, device='cuda', save_dir=args.save_dir)

if __name__ == "__main__":
    run_training()