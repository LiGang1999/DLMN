#!/bin/bash

python3 mst_eval.py --model="deeplab" \
--load_model="./checkpoints/dl_gta5_adapted_wo_cpae.pth" --test_scales='1.2, 1.4'

# python3 mst_eval.py --model="deeplab" \
# --load_model="./checkpoints/dl_gta5_adapted_wo_cpae.pth"
