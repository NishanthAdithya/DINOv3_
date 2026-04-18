# Finetuning DINOv2 with LoRA for Image Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NishanthAdithya/DINOv2_/blob/main/Explanation.ipynb)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

This repository explores finetuning **DINOv2** (Oquab et al., 2024) encoder weights using **Low-Rank Adaptation** (LoRA, Hu et al., 2021) and a simple 1×1 convolution decoder. LoRA makes it possible to finetune to new tasks without adjusting the original encoder weights, by adding a small set of weights between each encoder block. The DINOv2 encoder weights are learned by self-supervised learning and accurately capture the natural image domain. For example, by applying PCA to the outputs of the encoders, we can get a coarse segmentation of the objects in the image and see semantically similar objects colored in the same color.

Check out the [`Explanation.ipynb`](Explanation.ipynb) notebook for a more detailed walkthrough of the code and ideas behind it. For high-resolution PCA embeddings, see [`Embedding_visualization.ipynb`](Embedding_visualization.ipynb).

---

## Pipeline

![Pipeline](assets/pipeline.png)

---

## Results after 2 Epochs (CPU, DINOv2-Base)

| Metric | Value |
|---|---|
| **Mean IoU** | **74.6%** |
| Pixel Accuracy | 94.1% |
| Trainable params | 599K / 87M (0.69%) |
| Train loss | 0.378 |
| Val loss | 0.211 |

### End-to-End: Input → PCA Features → Prediction → Overlay

![End-to-end grid](assets/grid.png)

### PCA Feature Maps (frozen encoder — no finetuning needed)

![PCA features](assets/eval_pca.png)

### Predictions vs Ground Truth

![Predictions](assets/eval_predictions.png)

### Per-class IoU

| Class | IoU | Class | IoU |
|---|---|---|---|
| background | 0.931 | cow | 0.828 |
| aeroplane | 0.765 | diningtable | 0.673 |
| bicycle | 0.402 | dog | 0.847 |
| bird | 0.820 | horse | 0.787 |
| boat | 0.755 | motorbike | 0.791 |
| bottle | 0.775 | person | 0.808 |
| bus | 0.881 | pottedplant | 0.386 |
| car | 0.830 | sheep | 0.801 |
| cat | 0.870 | sofa | 0.686 |
| chair | 0.450 | train | 0.870 |
| — | — | tvmonitor | 0.703 |

---

## Architecture

```
Input image  (B, 3, H, W)
      │
      ▼
DINOv2 ViT Encoder  ── frozen weights ──────┐
      │                    LoRA adapters ───┘   rank=4, α=1.0
      │  patch tokens  (B, N, C)
      │  reshape  →  (B, C, h, w)   where h = H / patch_size
      ▼
1×1 Conv decoder head  →  (B, num_classes, h, w)
      │
      ▼
Bilinear upsample  →  (B, num_classes, H, W)
```

Only **LoRA adapter weights** and the **decoder head** are trained — everything else stays frozen.

---

## Installation

```bash
git clone https://github.com/NishanthAdithya/DINOv2_.git
cd DINOv2_

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

---

## Quick Start

### 1 — Download Pascal VOC 2012

```python
from torchvision.datasets import VOCSegmentation
VOCSegmentation('./data', year='2012', download=True)
```

### 2 — Train

```bash
python train.py                            # default config (5 epochs, bs=4)
python train.py --epochs 20 --lr 2e-4     # override settings
python train.py --resume checkpoints/best.pth   # resume
```

### 3 — Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pth
python evaluate.py --checkpoint checkpoints/best.pth \
    --save_viz viz_results/preds.png --save_pca viz_results/pca.png
```

### 4 — End-to-End Visualize

```bash
# 8 VOC val samples in one grid
python visualize.py --dataset val --n_samples 8 \
    --checkpoint checkpoints/best.pth --grid

# Single image
python visualize.py --image photo.jpg \
    --checkpoint checkpoints/best.pth --show_pca

# Whole folder
python visualize.py --image folder/ \
    --checkpoint checkpoints/best.pth --out_dir results/
```

---

## Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml):

| Key | Default | Description |
|---|---|---|
| `encoder.name` | `facebook/dinov2-base` | HuggingFace model ID |
| `lora.rank` | `4` | LoRA rank |
| `lora.alpha` | `1.0` | LoRA scaling factor (`α/r`) |
| `decoder.num_classes` | `21` | Output classes (21 for VOC) |
| `data.image_size` | `224` | Training resolution |
| `training.epochs` | `5` | Training epochs |
| `training.lr` | `1e-4` | Learning rate |
| `training.amp` | `false` | Mixed-precision (set `true` on CUDA) |

---

## Projects in this Repository

| Project | Task | Dataset | Key metric |
|---|---|---|---|
| **[Segmentation](.)** (root) | Semantic segmentation — 21 classes | Pascal VOC 2012 | mIoU 74.6% |
| **[Depth Estimation](depth/)** | Monocular depth from a single image | NYU Depth v2 (indoor) | AbsRel, RMSE, δ<1.25 |

Both projects share the same `src/lora.py` LoRA implementation and frozen DINOv2-Base backbone — only the task head and loss differ.

---

## Project Structure

```
├── src/
│   ├── lora.py                # LoRALinear + apply_lora()  (shared by both projects)
│   ├── model.py               # DINOSegmenter (encoder + LoRA + decoder)
│   ├── dataset.py             # Pascal VOC wrapper + paired augmentations
│   └── utils.py               # PCA viz, RunningMIoU, plotting helpers
├── depth/                     # Monocular depth estimation sub-project
│   ├── src/                   # DepthEstimator, NYUDepthDataset, losses, metrics
│   ├── configs/default.yaml
│   ├── train.py / evaluate.py / predict.py
│   └── README.md
├── configs/
│   └── default.yaml           # All hyperparameters
├── assets/                    # README images & pipeline diagram
├── train.py                   # Training loop (tqdm + AMP + checkpointing)
├── evaluate.py                # Full evaluation + per-class IoU table
├── predict.py                 # Single-image / folder inference
├── visualize.py               # End-to-end visualization pipeline
├── Explanation.ipynb          # Step-by-step walkthrough notebook
├── Embedding_visualization.ipynb  # High-resolution PCA embedding viewer
├── requirements.txt
└── LICENSE
```

---

## Tips for Better Results

| Change | Expected gain |
|---|---|
| More epochs (10–50) | +5–15% mIoU |
| GPU training | Same accuracy, 10× faster |
| `image_size: 448` | +2–5% mIoU (richer features) |
| `lora.rank: 8` or `16` | +1–3% mIoU |

---

## References

- Oquab et al. (2024) — *DINOv2: Learning Robust Visual Features without Supervision*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
