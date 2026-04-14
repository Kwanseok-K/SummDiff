#!/bin/bash

# SummDiff on SumMe

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train True \
    --model SummDiff \
    --dataset summe \
    --batch_size 20 \
    --epochs 45 \
    --tag summdiff_summe_dit \
    --l2_reg 5e-4 \
    --lr 3e-5 \
    --denoiser DiT \
    --sigmoid_temp 2 \
    --eps 1e-3 \
    --n_input_proj 3 \
    --individual True \
    --scores_embed learned \
    --train_val False \
    --p_uncond 0.2 \
    --clamp True \
    --w 0.2
