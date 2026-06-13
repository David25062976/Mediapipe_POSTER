#!/bin/bash
chmod a+x train_affect_mediapipe.sh
export LD_LIBRARY_PATH=/home/custexp/anaconda3/envs/poster_env_v2/lib:$LD_LIBRARY_PATH
unset DISPLAY
python test_all_confbox_mediapipe_pt.py --name_notes _result_shopping_196_14 --image_folder "/home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-1" --mediapipe_points 196 --mediapipe_patch_size 14
python test_all_confbox_mediapipe_pt.py --name_notes _result_shopping_196_14 --image_folder "/home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-2" --mediapipe_points 196 --mediapipe_patch_size 14
python test_all_confbox_mediapipe_pt.py --name_notes _result_shopping_196_14 --image_folder "/home/Dataset/customer_ori2/out/A Simple Shopping Haul Turned Into Online Drama/Haul-4" --mediapipe_points 196 --mediapipe_patch_size 14