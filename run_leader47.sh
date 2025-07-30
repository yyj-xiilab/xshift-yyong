#!/bin/bash

# 🚀 리더님 방식: 파일명 기반 47개 클래스 Domain Adaptation 실행
# main_leader47.py 파일 실행

CUDA_VISIBLE_DEVICES=3 python3 main_leader47.py \
    --name leader47_filename_da \
    --dataset leader_filename_47class \
    --model_type ViT-B_16 \
    --pretrained_dir checkpoint/imagenet21k_ViT-B_16.npz \
    --num_steps 3000 \
    --img_size 256 \
    --learning_rate 0.05 \
    --weight_decay 0.01 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_every 100 \
    --beta 0.1 \
    --gamma 0.2 \
    --theta 0.1 \
    --use_im \
    --use_cp \
    --optimal 1 \
    --warmup_steps 300 \
    --gpu_id 3 