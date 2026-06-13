#!/bin/bash
chmod a+x train_affect_mediapipe.sh
python train_shopping_window_pt.py --gpu 0,1 --temporal_type MLP --window_size 32 --stride 32
python train_shopping_window_pt.py --gpu 0,1 --temporal_type MLP --window_size 32 --stride 16
python train_shopping_window_pt.py --gpu 0,1 --temporal_type MLP --window_size 32 --stride 8
python train_shopping_window_pt.py --gpu 0,1 --temporal_type MLP --window_size 16 --stride 16