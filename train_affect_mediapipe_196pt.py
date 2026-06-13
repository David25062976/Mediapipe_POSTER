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
from models.emotion_hyp_affect_mediapipe_196pt import pyramid_trans_expr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet8class', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000002, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='Tolerating no progress for n epochs.')
    parser.add_argument('--freeze', action='store_true', default=False, help='Freeze pyramid_fuse layer or not')
    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=str, default=123, help='torch.manual_seed({seed})')
    parser.add_argument('--save_dir', type=str, default=time.strftime("%Y%m%d-%H%M%S"), help='model save directory')
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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
        transforms.RandomErasing(scale=(0.02, 0.1)),
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

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet_customer_2class'
        # num_classes = 3
        # class_names = ['interested', 'thinking', 'pass']
        class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'interested', 'thinking', 'Anger', 'pass']
        num_classes = len(class_names)
        train_dataset = Affectdataset_8class_2(datapath, train=True, transform=data_transforms, basic_aug=True)
        val_dataset = Affectdataset_8class_2(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype, freeze=args.freeze, mediapipe_patch_size=14)

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

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    params = model.parameters()
    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False)
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
    for i in range(1, args.epochs + 1):
        train_pre_labels, train_gt_labels, train_probs = [], [], []
        train_loss, correct_sum, iter_cnt = 0.0, 0, 0
        start_time = time.time()
        model.train()

        train_loader_tqdm = tqdm(train_loader, desc=f"[Epoch {i}/{args.epochs}] Train", unit="batch")

        # for batch_i, (imgs, targets) in enumerate(train_loader):
        for batch_i, (imgs, targets, coords, flip_flags) in enumerate(train_loader_tqdm):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()
            coords = coords.cuda()
            flip_flags = flip_flags.cuda()

            # SAM Optimizer first
            outputs, features = model(imgs, coords, flip_flags)

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM Optimizer second
            outputs, features = model(imgs, coords, flip_flags)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            loss.backward()
            optimizer.second_step(zero_grad=True)


            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")

            # 收集數據用於計算指標
            probs = F.softmax(outputs, dim=1)
            _, predicts = torch.max(outputs, 1)
            
            train_pre_labels.extend(predicts.cpu().tolist())
            train_gt_labels.extend(targets.cpu().tolist())
            train_probs.extend(probs.detach().cpu().tolist())
            correct_sum += torch.eq(predicts, targets).sum()
            train_loss += loss.item()

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
            for batch_i, (imgs, targets) in enumerate(val_loader_tqdm):
                outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
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
                            'model_state_dict': model.state_dict(),
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
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter} out of {patience}")

            # 寫入單行 Epoch 結果到 log.txt
            log_str = (f"{i:03d}   | {train_loss:.4f}     | {t_acc:.4f}    | {val_loss:.4f}   | {v_acc:.4f}   | {v_f1:.4f} |")
            if is_best:
                log_str += " <--- [BEST]"
                
            with open(log_file_path, 'a') as f:
                f.write(log_str + "\n")

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
