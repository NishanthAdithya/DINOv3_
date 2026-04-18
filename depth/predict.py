"""
Single-image depth inference with plasma colormap visualization.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/folder/ --output results/
    python predict.py --image img.jpg --checkpoint checkpoints/depth_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model   import DepthEstimator
from src.dataset import DepthTransform

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True,  help="image file or folder")
    p.add_argument("--checkpoint", default="checkpoints/depth_best.pt")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--output",     default="predictions/")
    p.add_argument("--max_depth",  type=float, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def colorize(depth: np.ndarray, vmin=0.0, vmax=10.0) -> np.ndarray:
    norm  = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    color = (cm.get_cmap("plasma")(norm)[:, :, :3] * 255).astype(np.uint8)
    return color


def predict_single(model, image_path: Path, transform, device: str,
                   max_depth: float, out_dir: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    img_t, _ = transform(img, np.zeros((img.height, img.width), dtype=np.float32))
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        depth_pred = model(img_t)[0, 0].cpu().numpy()

    depth_colored = colorize(depth_pred, vmax=max_depth)

    # Side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    axes[0].imshow(np.array(img))
    axes[0].set_title("Input Image", color="white", fontsize=13)

    im = axes[1].imshow(depth_colored)
    axes[1].set_title("Predicted Depth", color="white", fontsize=13)

    # Colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="plasma",
                              norm=plt.Normalize(vmin=0, vmax=max_depth)),
        ax=axes[1], fraction=0.046, pad=0.04,
    )
    cbar.set_label("Depth (m)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    out_path = out_dir / (image_path.stem + "_depth.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # Also save raw depth as npy
    np.save(out_dir / (image_path.stem + "_depth.npy"), depth_pred)
    print(f"  Saved: {out_path}")


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    max_d    = args.max_depth or cfg["depth"]["max_depth"]

    # Model
    enc = cfg["encoder"]
    lor = cfg["lora"]
    model = DepthEstimator(
        encoder_name  = enc["name"],
        patch_size    = enc["patch_size"],
        lora_rank     = lor["rank"],
        lora_alpha    = lor["alpha"],
        lora_keywords = tuple(lor["target_keywords"]),
        max_depth     = max_d,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    transform = DepthTransform(image_size=cfg["data"]["image_size"], is_train=False)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    src = Path(args.image)
    if src.is_dir():
        images = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    else:
        images = [src]

    print(f"Predicting on {len(images)} image(s) -> {out_dir}")
    for img_path in images:
        predict_single(model, img_path, transform, device, max_d, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
