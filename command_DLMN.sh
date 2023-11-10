#!/bin/bash

python3 train_DLMN.py --batch_size 4 --start_iter 0 --end_iter 50000 --load_saved True \
--model deeplab --runs dl_gta5_DLMN --save_every 100 --source gta5 --entw 5e-3 --loss ce
