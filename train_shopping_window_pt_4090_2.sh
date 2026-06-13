#!/bin/bash
chmod a+x train_affect_mediapipe.sh
python train_shopping_window_pt.py --gpu 0,1 --temporal_type MLP --window_size 16 --stride 16
python train_shopping_window_pt.py --gpu 0,1 --temporal_type Transformer --window_size 16 --stride 16 --remark nhead=4,num_layers=2
python train_shopping_window_pt.py --gpu 0,1 --temporal_type Transformer --window_size 16 --stride 8 --remark nhead=4,num_layers=2