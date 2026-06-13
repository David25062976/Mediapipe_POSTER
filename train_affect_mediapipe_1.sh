#!/bin/bash
chmod a+x train_affect_mediapipe.sh
export LD_LIBRARY_PATH=/home/custexp/anaconda3/envs/poster_env_v2/lib:$LD_LIBRARY_PATH
unset DISPLAY
python train_affect2.py --gpu 1 --epochs 50 --early_stop_patience 50 --lr 0.0000015 --dataset affectnet8class_shopping --batch_size 128 --remark base,Shopping
