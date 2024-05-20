#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
            --nproc_per_node=1  \
            --master_port=$((RANDOM%1000+12000))  \
            main.py \
            --cfg_file './cfgs/test.yaml' \
            --mode train