#!/bin/bash

# SummDiff on MrHiSum

CUDA_VISIBLE_DEVICES=0 python main.py \
    --train True \
    --model SummDiff \
    --dataset mrhisum \
    --batch_size 256 \
    --epochs 40 \
    --tag summdiff_mrhisum \
    --l2_reg 5e-4 \
    --lr 5e-5 \
    --denoiser DiT \
    --ema True \
    --sigmoid_temp 2 \
    --eps 1e-3 \
    --n_input_proj 3 \
    --p_uncond 0.2 \
    --K 100 \
    --scores_embed learned
