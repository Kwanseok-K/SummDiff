# SummDiff: Diffusion-based Video Summarization

This repository contains the official implementation of **SummDiff**, a diffusion-based approach to video summarization. SummDiff formulates importance score prediction as a denoising process, leveraging a diffusion model conditioned on video features to generate frame-level saliency scores.

## Overview

SummDiff introduces a novel diffusion-based framework for video summarization that:
- Models importance score prediction as an iterative denoising process
- Supports multiple denoiser architectures: **DiT** (Diffusion Transformer) and **LatentMLP**
- Employs classifier-free guidance with unconditional training for improved generation
- Uses Exponential Moving Average (EMA) for stable training
- Evaluates with comprehensive metrics: Kendall's Tau, Spearman's Rho

## Results

### MrHiSum

| Method | KTau | SRho |
|--------|------|------|
| SummDiff | **0.175** | **0.238** |

### TVSum 

| Method | KTau | SRho |
|--------|------|------|
| SummDiff | **0.195** | **0.255** |

### SumMe 

| Method | KTau | SRho |
|--------|------|------|
| SummDiff | **0.256** | **0.285** |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA (recommended)

```bash
# Create conda environment
conda create -n summdiff python=3.8
conda activate summdiff

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### TVSum

1. Download the preprocessed h5 file from the [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) repository (`data/` folder) or use the version provided in this repo:
   ```
   dataset/tvsum/eccv16_dataset_tvsum_google_pool5.h5
   ```
   The h5 file contains pre-extracted GoogLeNet Pool5 features, change points, ground truth scores, and user summaries. The features were originally extracted by [Ke Zhang](https://github.com/kezhang-cs) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce).

2. Download the TVSum annotation file `ydata-tvsum50.tsv` from the [TVSum dataset page](https://github.com/yalesong/tvsum):
   ```bash
   # Place as:
   dataset/ydata-anno.tsv
   ```
   This file contains per-user frame-level importance scores used for Kendall's Tau and Spearman's Rho evaluation.

3. The train/test splits are already included:
   ```
   dataset/tvsum_splits.txt    # 5-fold cross-validation splits
   ```

### SumMe

1. Download the preprocessed h5 file from the [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) repository (`data/` folder) or use the version provided in this repo:
   ```
   dataset/summe/eccv16_dataset_summe_google_pool5.h5
   ```

2. The train/test splits are already included:
   ```
   dataset/summe_splits.txt    # 5-fold cross-validation splits
   ```

### MrHiSum

1. Download the MrHiSum dataset h5 file (contains pre-extracted features, ground truth scores, change points, and ground truth summaries).

2. Specify the path via `--data_path`:
   ```bash
   python main.py --data_path /path/to/mrsum_with_features_gtsummary_modified.h5 ...
   ```

3. The train/val/test split is already included:
   ```
   dataset/mrsum_split.json
   ```

### Expected Directory Structure

```
dataset/
├── tvsum/
│   └── eccv16_dataset_tvsum_google_pool5.h5
├── summe/
│   └── eccv16_dataset_summe_google_pool5.h5
├── tvsum_splits.txt
├── summe_splits.txt
├── mrsum_split.json
├── null_video.npy
└── ydata-anno.tsv
```

## Training

Training scripts are provided in `scripts/`:
```bash
bash scripts/train_mrhisum.sh
bash scripts/train_tvsum.sh
bash scripts/train_summe.sh
```

You can specify the GPU device by setting `CUDA_VISIBLE_DEVICES` in the script.

## Evaluation

To evaluate a trained model:

```bash
# MrHiSum (with EMA)
python main.py \
    --train False \
    --model SummDiff \
    --dataset mrhisum \
    --data_path /path/to/mrsum_with_features_gtsummary_modified.h5 \
    --denoiser DiT \
    --ema True \
    --sigmoid_temp 2 \
    --eps 1e-3 \
    --n_input_proj 3 \
    --scores_embed learned \
    --K 100 \
    --ckpt_path Summaries/SummDiff/<tag>/best_f1score_model/best_f1.pkl
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (`SummDiff`, `MLP`) | `SummDiff` |
| `--dataset` | Dataset (`mrhisum`, `tvsum`, `summe`) | `mrhisum` |
| `--denoiser` | Denoiser type (`DiT`, `latentmlp`, `Transformer_dec`) | `DiT` |
| `--K` | Number of diffusion steps | `200` |
| `--ema` | Use Exponential Moving Average | `False` |
| `--p_uncond` | Probability of unconditional training | `0.0` |
| `--w` | Weight for classifier-free guidance | `0.1` |
| `--sigmoid_temp` | Temperature for sigmoid scheduling | `1.0` |
| `--eps` | Epsilon for logit transformation | `1e-3` |
| `--clamp` | Clamp ground truth scores to [0.05, 0.95] | `False` |
| `--individual` | Use individual annotator labels | `False` |
| `--scores_embed` | Score embedding type (`learned`, `sinusoidal`) | `learned` |

## Project Structure

```
SummDiff/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── scripts/                         # Training scripts
│   ├── train_mrhisum.sh
│   ├── train_tvsum.sh
│   └── train_summe.sh
├── model/
│   ├── configs.py                   # Configuration class
│   ├── solver.py                    # Training and evaluation logic
│   ├── mrsum_dataset.py             # Dataset loaders
│   └── utils/
│       ├── evaluation_metrics.py    # F-score, Kendall Tau, Spearman Rho
│       ├── generate_summary.py      # Knapsack-based summary generation
│       ├── evaluate_map.py          # mAP evaluation
│       └── knapsack_implementation.py
├── networks/
│   ├── mlp.py                       # Simple MLP baseline
│   └── summ_diff/                   # SummDiff diffusion model
│       ├── summ_diff.py             # Main model and loss
│       ├── transformer.py           # Diffusion transformer (DiT/decoder)
│       ├── attention.py             # Multi-head attention
│       ├── latentmlp.py             # LatentMLP denoiser
│       ├── position_encoding.py     # Positional encodings
│       └── utils.py                 # Diffusion utilities
└── dataset/                         # Data splits and annotations
    ├── mrsum_split.json
    ├── tvsum_splits.txt
    ├── summe_splits.txt
    ├── null_video.npy
    ├── ydata-anno.tsv
    ├── tvsum/                       # TVSum h5 features
    └── summe/                       # SumMe h5 features
```

## Acknowledgements

This codebase builds upon:
- [MomentDiff](https://github.com/HDVideo/MomentDiff) for the diffusion-based framework
- [DETR](https://github.com/facebookresearch/detr) for the transformer and matcher modules
- [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR) for the multi-head attention implementation
- [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM) for the preprocessed dataset h5 files
- [CSTA](https://github.com/thswodnjs3/CSTA) for the evaluation framework

## Citation

```bibtex
@article{summdiff,
  title={SummDiff: Generative Modeling of Video Summarization with Diffusion},
  author={Kwanseok Kim*, Jaehoon Hahm*, Sumin Kim, Jinwhan Sul, Byunghak Kim, Joonseok Lee},
  journal={ICCV},
  year={2025}
}
```
