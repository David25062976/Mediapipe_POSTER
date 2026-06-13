#!/bin/bash
chmod a+x train_affect_mediapipe.sh
python train_shopping_window_pt.py --gpu 0 --temporal_type MLP --epochs 30 --window_size 8 --stride 8 --remark noFreeze
python train_shopping_window_pt.py --gpu 0 --temporal_type MLP --epochs 30 --window_size 8 --stride 4 --remark noFreeze
python train_shopping_window_pt.py --gpu 0 --temporal_type Transformer --epochs 30 --window_size 8 --stride 8 --remark nhead=4,num_layers=2,noFreeze
python train_shopping_window_pt.py --gpu 0 --temporal_type Transformer --epochs 30 --window_size 8 --stride 4 --remark nhead=4,num_layers=2,noFreeze