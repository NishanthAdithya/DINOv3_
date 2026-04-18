"""
Evaluate a trained depth model on the NYU Depth v2 validation split.

Usage:
    python evaluate.py --checkpoint checkpoints/depth_best.pt
    python evaluate.py --checkpoint checkpoints/depth_best.pt --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model   import DepthEstimator
from src.dataset import NYUDepthDataset
from src.utils   import DepthMetrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def bar(val: float, width: int = 20) -> str:
    filled = int(round(val * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    enc = cfg["encoder"]
    lor = cfg["lora"]
    model = DepthEstimator(
        encoder_name  = enc["name"],
        patch_size    = enc["patch_size"],
        lora_rank     = lor["rank"],
        lora_alpha    = lor["alpha"],
        lora_keywords = tuple(lor["target_keywords"]),
        max_depth     = cfg["depth"]["max_depth"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # Data
    data_cfg = cfg["data"]
    val_ds   = NYUDepthDataset(
        h5_path      = data_cfg["h5_path"],
        split        = "val",
        image_size   = data_cfg["image_size"],
        val_fraction = data_cfg["val_fraction"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = data_cfg["num_workers"],
        pin_memory  = torch.cuda.is_available(),
    )

    metrics   = DepthMetrics(max_depth=cfg["depth"]["max_depth"])
    depth_bins = {
        "0-1m"  : DepthMetrics(max_depth=1.0),
        "1-3m"  : DepthMetrics(max_depth=3.0),
        "3-10m" : DepthMetrics(max_depth=10.0),
    }

    with torch.no_grad():
        for imgs, depths in tqdm(val_loader, desc="Evaluating", dynamic_ncols=True):
            imgs   = imgs.to(device)
            depths = depths.to(device)
            pred   = model(imgs)
            metrics.update(pred, depths)

            # Per-range metrics
            for label, (lo, hi) in [("0-1m", (0, 1)), ("1-3m", (1, 3)), ("3-10m", (3, 10))]:
                mask = (depths >= lo) & (depths < hi)
                if mask.any():
                    depth_bins[label].update(pred * mask, depths * mask)

    m = metrics.compute()

    # ── Results table ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  NYU Depth v2 — Validation Results")
    print("=" * 60)
    print(f"  AbsRel  : {m['abs_rel']:.4f}  {bar(1 - m['abs_rel'])}")
    print(f"  RMSE    : {m['rmse']:.4f}  {bar(max(0, 1 - m['rmse']/5))}")
    print(f"  delta<1.25   : {m['delta1']:.4f}  {bar(m['delta1'])}")
    print(f"  delta<1.25^2 : {m['delta2']:.4f}  {bar(m['delta2'])}")
    print(f"  delta<1.25^3 : {m['delta3']:.4f}  {bar(m['delta3'])}")
    print()

    # Per depth-range breakdown
    print("  Per depth-range breakdown:")
    print(f"  {'Range':<10} {'AbsRel':>8} {'RMSE':>8} {'d<1.25':>8}")
    print("  " + "-" * 40)
    for label, dm in depth_bins.items():
        try:
            bm = dm.compute()
            print(f"  {label:<10} {bm['abs_rel']:>8.4f} {bm['rmse']:>8.4f} {bm['delta1']:>8.4f}")
        except Exception:
            print(f"  {label:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
