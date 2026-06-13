#!/bin/bash
chmod a+x train_affect_mediapipe.sh
export LD_LIBRARY_PATH=/home/custexp/anaconda3/envs/poster_env_v2/lib:$LD_LIBRARY_PATH
unset DISPLAY
python test_single_confbox_mediapipe_pt.py --name_notes _result_AffectNet_49_14 --checkpoint ./checkpoint/20260423-121025_49_14_ori/best.pth --mediapipe_points 49 --mediapipe_patch_size 14
python test_single_confbox_mediapipe_pt.py --name_notes _result_AffectNet_49_24 --checkpoint ./checkpoint/20260424-134314_49_24_ori/best.pth --mediapipe_points 49 --mediapipe_patch_size 24
python test_single_confbox_mediapipe_pt.py --name_notes _result_AffectNet_196_14 --checkpoint ./checkpoint/20260426-161602_196_14_ori/best.pth --mediapipe_points 196 --mediapipe_patch_size 14
python test_single_confbox_mediapipe_pt.py --name_notes _result_AffectNet_196_24 --checkpoint ./checkpoint/20260425-132208_196_24_ori/best.pth --mediapipe_points 196 --mediapipe_patch_size 24
python test_single_confbox_shopping.py

# python test_single_confbox_mediapipe_pt.py --name_notes _result_49_14 --checkpoint ./checkpoint/20260430-220136_Shopping_49_14/best.pth --mediapipe_points 49 --mediapipe_patch_size 14
# python test_single_confbox_mediapipe_pt.py --name_notes _result_49_24 --checkpoint ./checkpoint/20260429-184718_Shopping_49_24/best.pth --mediapipe_points 49 --mediapipe_patch_size 24
# python test_single_confbox_mediapipe_pt.py --name_notes _result_196_14 --checkpoint ./checkpoint/20260501-060739_Shopping_196_14/best.pth --mediapipe_points 196 --mediapipe_patch_size 14
# python test_single_confbox_mediapipe_pt.py --name_notes _result_196_24 --checkpoint ./checkpoint/20260429-184618_Shopping_196_24/best.pth --mediapipe_points 196 --mediapipe_patch_size 24