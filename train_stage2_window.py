import warnings

warnings.filterwarnings("ignore")
# from apex import amp
import numpy as np
import pandas as pd
from collections import defaultdict
import torch.utils.data as data
from torchvision import transforms
import os
from tqdm import tqdm

import torch
import argparse
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class_ori import Affectdataset_8class
from data_preprocessing.dataset_customer import AffectDatasetWindow, AffectDatasetWindow_facecrop

from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_preprocessing.sam import SAM
from models.emotion_hyp import pyramid_trans_expr
from models.emotion_hyp_window import pyramid_trans_expr_window

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='consumer', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')    # './checkpoint/rafdb_best.pth'
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.00004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='Tolerating no progress for n epochs.')
    parser.add_argument('--window_size', type=int, default=8, help='Window size of sliding window')
    parser.add_argument('--stride', type=int, default=8, help='Stride of sliding window')
    parser.add_argument('--face_crop', action='store_true', default=False, help='Crop face or not')
    parser.add_argument('--FL_freeze', action='store_true', default=False, help='Freeze face_landback layer or not')
    parser.add_argument('--IR_freeze', action='store_true', default=False, help='Freeze ir_back layer or not')
    parser.add_argument('--PF_freeze', type=str, default=None, help='Freeze pyramid_fuse layer or not')    # './checkpoint/20260115-160251/epoch32_acc0.84025.pth'
    parser.add_argument('--temporal_type', type=str, default='MLP', help='Temporal type: MLP / CNN / Transformer')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('--seed', type=str, default=123, help='torch.manual_seed({seed})')
    parser.add_argument('--save_dir', type=str, default=time.strftime("window_%Y%m%d-%H%M%S"), help='model save directory')

    return parser.parse_args()

def save_result(train_results, val_results, gt_labels, pre_labels, save_dir, epoch):
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
    class_names = ['Dislike', 'Hesitate', 'Like']
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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    log_name = 'log.txt'
    log_name = os.path.join(args.save_dir, log_name)

    with open(log_name, 'w') as f:
        f.write(f"Checkpoint = {str(args.checkpoint)}\n")
        f.write(f"Batch Size = {str(args.batch_size)}, Val Batch Size = {str(args.val_batch_size)}\n")
        f.write(f"Model type = {str(args.modeltype)}\n")
        f.write(f"Optimizer = {str(args.optimizer)}\n")
        f.write(f"Learning Rate = {str(args.lr)}\n")
        f.write(f"Momentum = {str(args.momentum)}\n")
        f.write(f"Workers = {str(args.workers)}\n")
        f.write(f"Epochs = {str(args.epochs)}\n")
        f.write(f"Early stop patience = {str(args.early_stop_patience)}\n")
        f.write(f"Window Size = {str(args.window_size)}\n")
        f.write(f"Stride = {str(args.stride)}\n")
        f.write(f"Facecrop = {str(args.face_crop)}\n")
        f.write(f"FL_freeze = {str(args.FL_freeze)}\n")
        f.write(f"IR_freeze = {str(args.IR_freeze)}\n")
        f.write(f"PF_freeze = {str(args.PF_freeze)}\n")
        f.write(f"temporal_type = {str(args.temporal_type)}\n")
        f.write(f"gpu = {str(args.gpu)}\n")
        f.write(f"seed = {str(args.seed)}\n")

        f.write("------------------------------------------\n")
        f.write("------------------------------------------\n")
    f.close()

    f = open(log_name, 'a')

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
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
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        train_dataset = RafDataSet(datapath, train=True, transform=data_transforms, basic_aug=True)
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        train_dataset = Affectdataset(datapath, train=True, transform=data_transforms, basic_aug=True)
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        train_dataset = Affectdataset_8class(datapath, train=True, transform=data_transforms, basic_aug=True)
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_val)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    
    elif args.dataset == "consumer":
        datapath = '/home/Dataset/face_croped_customer_ori2'
        train_anno = "all_train_list.txt"
        test_anno = "all_test_list.txt"
        num_classes = 3
        train_dataset = AffectDatasetWindow_facecrop(datapath, anno=train_anno, window_size=args.window_size, stride=args.stride, transform=data_transforms, face_crop=args.face_crop)
        test_dataset = AffectDatasetWindow_facecrop(datapath, anno=test_anno, window_size=args.window_size, stride=args.stride, transform=data_transforms_val, face_crop=args.face_crop)

        f.write(f"Train file = {train_anno}\tsize = {str(train_dataset.__len__())}\n")
        f.write(f"Test file = {test_anno}\tsize = {str(test_dataset.__len__())}\n")
        freeze_list = [args.FL_freeze, args.IR_freeze, args.PF_freeze]
        model = pyramid_trans_expr_window(img_size=224, num_classes=num_classes, window_size=args.window_size, type=args.modeltype, freeze_list=freeze_list, temporal_type=args.temporal_type)

    else:
        return print('dataset name is not correct')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # model = Networks.ResNet18_ARM___RAF()

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    params = model.parameters()
    if args.optimizer == 'adamw':
        # base_optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        # base_optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        # base_optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")
    # print(optimizer)
    # optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False,)    # ori
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.1, adaptive=False, weight_decay=1e-4)    # L2 Regularization

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    f.write('Total Parameters: %.3fM \n' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    train_acc_result, train_f1_result, train_auc_result = [], [], []
    val_acc_result, val_f1_result, val_auc_result = [], [], []

    l1_lambda = 1e-5
    best_acc = 0
    patience = args.early_stop_patience  # 容忍 early_stop_patience 個 epoch 沒有進步
    counter = 0
    misclassified_counter = defaultdict(int)    # 儲存 validation 判斷錯誤
    for i in range(1, args.epochs + 1):
        train_pre_labels, train_gt_labels, train_probs = [], [], []
        train_loss, correct_sum, iter_cnt = 0.0, 0, 0
        start_time = time.time()
        model.train()

        train_bar = tqdm(train_loader, desc="Training", ncols=100)
        for img_frames, targets, frame_paths in train_bar:
            # img_frames:   [v1[f1, f2, f3, ..., window_size], v2[], v3[], ..., batch_size]
            # targets:      [t1, t2, t3, t4, ..., batch_size]
            iter_cnt += 1
            optimizer.zero_grad()
            img_frames, targets = img_frames.cuda(), targets.cuda()

            outputs, features = model(img_frames)

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            # L1 Regularization (first step)
            l1_reg = torch.tensor(0.).cuda()
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, features = model(img_frames)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            loss = 2 * lsce_loss + CE_loss

            # L1 Regularization (second step)
            l1_reg = torch.tensor(0.).cuda()
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss = loss + l1_lambda * l1_reg

            loss.backward() # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

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

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Training F1:%.4f. Training AUC:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, t_acc, t_f1, t_auc, train_loss, optimizer.param_groups[0]["lr"]))
        f.write('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Training F1:%.4f. Training AUC:%.4f. Loss: %.3f LR:%.6f \n' %
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

            test_bar = tqdm(val_loader, desc="Validating", ncols=100)
            for img_frames, targets, frame_paths in test_bar:
                outputs, features = model(img_frames.cuda())
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

                # ---------------------------------------------------------
                # 2. 【新增】紀錄錯誤樣本邏輯
                # ---------------------------------------------------------
                # 把被轉置的路徑轉回來: (Window, Batch) -> (Batch, Window)
                batch_paths = list(zip(*frame_paths))
                
                # 轉換 tensor 到 CPU list 以便與路徑對應
                batch_preds = predicts.cpu().tolist()
                batch_targets = targets.cpu().tolist()

                # 遍歷這個 Batch 中的每一個樣本
                for j in range(len(batch_targets)):
                    # 如果預測錯誤
                    if batch_preds[j] != batch_targets[j]:
                        # 取得該樣本 Window 中的第一張圖片路徑
                        # batch_paths[j] 是一個 tuple，包含該樣本所有 frame 的路徑
                        first_frame_path = batch_paths[j][0] 
                        
                        # 字典計數加一
                        misclassified_counter[first_frame_path] += 1
                # ---------------------------------------------------------

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
            f.write("[Epoch %d] Validation time:%.2f, Validation accuracy:%.4f. Validation F1:%.4f. Validation AUC:%.4f. Loss:%.3f score %4f \n" % (
            i, elapsed, v_acc, v_f1, v_auc, val_loss, total_socre))


            if v_acc > 0.65 and v_acc > best_acc:
                best_acc = v_acc
                counter = 0
                print("best_acc:" + str(best_acc))
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                            os.path.join(args.save_dir, "epoch" + str(i) + "_acc" + str(round(v_acc, 4)) + ".pth"))
                print('Model saved.')

                train_dict = {'acc': train_acc_result, 'f1': train_f1_result, 'auc': train_auc_result}
                val_dict = {'acc': val_acc_result, 'f1': val_f1_result, 'auc': val_auc_result}
                save_result(
                    train_results=train_dict,
                    val_results=val_dict,
                    gt_labels=val_gt_labels,
                    pre_labels=val_pre_labels,
                    save_dir=args.save_dir, # 你指定的目錄
                    epoch=i
                )
                print(f'Plots saved at epoch {i}.')

            elif v_acc > best_acc:
                best_acc = v_acc
                counter = 0
                print("best_acc:" + str(best_acc))
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter} out of {patience}")
                if counter >= patience:
                    print("Early stopping triggered!")
                    break # 跳出訓練迴圈

    df_error = pd.DataFrame(list(misclassified_counter.items()), columns=['file_path', 'error_count'])
    df_error = df_error.sort_values(by='error_count', ascending=False)
    df_error.to_csv(os.path.join(args.save_dir, 'misclassified_samples.csv'), index=False, encoding='utf-8-sig')
    print("Misclassification report saved.")

    train_dict = {'acc': train_acc_result, 'f1': train_f1_result, 'auc': train_auc_result}
    val_dict = {'acc': val_acc_result, 'f1': val_f1_result, 'auc': val_auc_result}
    save_result(
        train_results=train_dict,
        val_results=val_dict,
        gt_labels=val_gt_labels,
        pre_labels=val_pre_labels,
        save_dir=args.save_dir, # 你指定的目錄
        epoch=-1
    )
    print(f'Final plots saved.')

    f.close()

if __name__ == "__main__":
    run_training()
