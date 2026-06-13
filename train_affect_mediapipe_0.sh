#!/bin/bash
chmod a+x train_affect_mediapipe.sh
export LD_LIBRARY_PATH=/home/custexp/anaconda3/envs/poster_env_v2/lib:$LD_LIBRARY_PATH
unset DISPLAY
python train_affect_mediapipe_pt.py --gpu 0 --dataset affectnet8class_shopping --batch_size 48 --mediapipe_points 196 --mediapipe_patch_size 24 --remark points=196,patch_size=24,Shopping
python train_affect_mediapipe_pt.py --gpu 0 --dataset affectnet8class_shopping --batch_size 48 --mediapipe_points 196 --mediapipe_patch_size 14 --remark points=196,patch_size=14,Shopping