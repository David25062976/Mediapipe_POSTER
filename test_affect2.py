import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
import matplotlib.pyplot as plt

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class_ori import Affectdataset_8class
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class_2

from utils import *
from models.emotion_hyp_affect import pyramid_trans_expr
from sklearn.metrics import confusion_matrix


# === 內建混淆矩陣繪圖函式 ===
def plot_confusion_matrix_inline(cm, labels_name, title, acc, normalize=True):
    """
    cm: ndarray 混淆矩陣
    labels_name: list 標籤名稱
    title: 圖表標題
    acc: 測試準確率
    normalize: 是否將數值轉為百分比
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # 處理除零 NaN
        fmt = '.2%'  # 顯示百分比
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{title} - Acc: {acc:.4f}")
    plt.colorbar()

    tick_marks = np.arange(len(labels_name))
    plt.xticks(tick_marks, labels_name, rotation=45)
    plt.yticks(tick_marks, labels_name)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text_val = f"{cm[i, j]*100:.2f}"
            else:
                text_val = str(cm[i, j])
            plt.text(j, i, text_val,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Plot confusion matrix.")
    return parser.parse_args()


def test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_classes = 7
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class_2":
        datapath = './data/AffectNet_customer_2class/'
        num_classes = 8
        test_dataset = Affectdataset_8class_2(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)
    else:
        return print('dataset name is not correct')

    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    test_size = len(test_dataset)
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              shuffle=False,
                                              pin_memory=True)

    model = model.cuda()

    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for imgs, targets in test_loader:
            outputs, _ = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()

    acc = bingo_cnt.float() / float(test_size)
    acc = np.around(acc.numpy(), 4)
    print(f"Test accuracy: {acc:.4f}")

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]
        elif args.dataset == "affectnet":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]
        elif args.dataset == "affectnet8class":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]
        elif args.dataset == "affectnet8class_2":
            labels_name = ['NE', 'HA', 'SA', 'SU', '(LI)', 'DI', "AN", "(HES)"]

        plot_confusion_matrix_inline(cm, labels_name, args.dataset, acc, normalize=True)


if __name__ == "__main__":
    test()
