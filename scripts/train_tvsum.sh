#!/bin/bash

# SummDiff on TVSum

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train True \
    --model SummDiff \
    --dataset tvsum \
    --batch_size 40 \
    --epochs 200 \
    --tag summdiff_tvsum \
    --l2_reg 5e-3 \
    --lr 3e-4 \
    --denoiser latentmlp \
    --sigmoid_temp 2 \
    --eps 1e-3 \
    --n_input_proj 3 \
    --individual False \
    --scores_embed learned \
    --train_val False \
    --p_uncond 0.2 \
    --clamp True \
    --dec_layers 3
