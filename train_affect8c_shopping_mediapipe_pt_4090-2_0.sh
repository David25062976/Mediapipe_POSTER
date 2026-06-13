#!/bin/bash
chmod a+x train_affect_mediapipe.sh
export LD_LIBRARY_PATH=/home/custexp/anaconda3/envs/poster_env_v2/lib:$LD_LIBRARY_PATH
unset DISPLAY
python train_affect8c_shopping_mediapipe_pt.py --gpu 0 --dataset affectnet8class_PD --batch_size 64 --exp_model 1-6 --remark "PD, exp 6"
python train_affect8c_shopping_mediapipe_pt.py --gpu 0 --dataset affectnet8class_PD --batch_size 64 --exp_model 1-5 --remark "PD, exp 5"