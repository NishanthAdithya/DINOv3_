"""
End-to-end visualization pipeline.

Shows the complete flow for every image:
  Input → PCA features (frozen encoder) → Segmentation prediction → Overlay

Usage
-----
python visualize.py --image path/to/image.jpg
python visualize.py --image path/to/image.jpg --checkpoint checkpoints/best.pth
python visualize.py --image folder/ --out_dir viz_results/ --checkpoint checkpoints/best.pth
python visualize.py --dataset val --n_samples 8 --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage
from tqdm import tqdm

from src.model import DINOSegmenter
from src.dataset import (
    VOC_CLASSES, VOC_COLORMAP,
    VOCSegmentationDataset, SegmentationTransform,
    denormalize, decode_segmap,
)
from src.utils import visualize_pca_features
from train import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Single-image end-to-end figure
# ---------------------------------------------------------------------------

def make_e2e_figure(
    orig_rgb: np.ndarray,
    pca_before: np.ndarray,
    pca_after: np.ndarray | None,
    seg_rgb: np.ndarray,
    overlay: np.ndarray,
    gt_rgb: np.ndarray | None = None,
    title: str = "",
) -> plt.Figure:
    """
    Build the end-to-end visualization panel:

    Row 1 (always):
      [Input] [PCA (no finetuning)] [PCA (finetuned)] [Segmentation] [Overlay]

    If gt_rgb is provided, a ground-truth column is added between Segmentation
    and Overlay.
    """
    has_gt     = gt_rgb is not None
    has_after  = pca_after is not None
    panels = ["Input image", "PCA (frozen)", "PCA (finetuned)" if has_after else None,
              "Ground truth" if has_gt else None, "Prediction", "Overlay"]
    panels = [p for p in panels if p is not None]
    n_cols = len(panels)

    fig = plt.figure(figsize=(4 * n_cols, 5), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, n_cols, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]

    images_to_plot = [orig_rgb, pca_before]
    if has_after:
        images_to_plot.append(pca_after)
    if has_gt:
        images_to_plot.append(gt_rgb)
    images_to_plot += [seg_rgb, overlay]

    for ax, img, label in zip(axes, images_to_plot, panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# Per-image helper
# ---------------------------------------------------------------------------

def process_image(
    model: DINOSegmenter,
    tensor: torch.Tensor,           # (1, C, H, W) on device
    device: torch.device,
    gt_mask: torch.Tensor | None = None,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits   = model(tensor)
        features = model.get_patch_features(tensor)

    orig_rgb = denormalize(tensor[0])
    pred_idx = logits.argmax(dim=1)[0].cpu().numpy()
    seg_rgb  = decode_segmap(pred_idx, VOC_COLORMAP)

    # Semi-transparent overlay
    overlay = (orig_rgb.astype(float) * 0.5 + seg_rgb.astype(float) * 0.5).astype(np.uint8)

    pca_rgb = visualize_pca_features(features, n_components=3)

    gt_rgb = None
    if gt_mask is not None:
        m = gt_mask.cpu().numpy().copy()
        m[m == 255] = 0
        gt_rgb = decode_segmap(m, VOC_COLORMAP)

    return {
        "orig": orig_rgb,
        "pca":  pca_rgb,
        "seg":  seg_rgb,
        "overlay": overlay,
        "gt":   gt_rgb,
        "features": features,
    }


def load_image_tensor(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    transform = SegmentationTransform(image_size=image_size, is_train=False)
    pil = PILImage.open(image_path).convert("RGB")
    dummy = PILImage.fromarray(np.zeros((pil.height, pil.width), dtype=np.uint8))
    t, _ = transform(pil, dummy)
    return t.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Dataset-mode: multi-image grid
# ---------------------------------------------------------------------------

def make_dataset_grid(results: list[dict], out_path: str) -> None:
    """Save a compact grid: one row per sample, columns = panels."""
    n = len(results)
    cols = ["Input", "PCA features", "Prediction", "Overlay"]
    has_gt = results[0]["gt"] is not None
    if has_gt:
        cols.insert(2, "Ground truth")

    fig, axes = plt.subplots(n, len(cols), figsize=(4 * len(cols), 4 * n),
                             constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        panels = [r["orig"], r["pca"]]
        if has_gt:
            panels.append(r["gt"])
        panels += [r["seg"], r["overlay"]]

        for ax, img, col in zip(axes[row], panels, cols):
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(col, fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end DINOv2/v3 segmentation visualizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   help="Single image file or folder of images")
    group.add_argument("--dataset", choices=["train", "val"], help="Visualize samples from VOC dataset")

    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint. If omitted, uses the frozen (untrained) model.")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--out_dir",    default="viz_results")
    parser.add_argument("--n_samples",  type=int, default=6,
                        help="Number of dataset samples to visualize (--dataset mode)")
    parser.add_argument("--grid",       action="store_true",
                        help="Save all samples in one combined grid image")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    enc_cfg = cfg["encoder"]
    model   = DINOSegmenter(
        encoder_name=enc_cfg["name"],
        num_classes=cfg["decoder"]["num_classes"],
        patch_size=enc_cfg["patch_size"],
        lora_rank=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_keywords=tuple(cfg["lora"]["target_keywords"]),
        use_dinov3=enc_cfg.get("use_dinov3", False),
        dinov3_hub_repo=enc_cfg.get("dinov3_hub_repo", ""),
    ).to(device)

    trained = args.checkpoint is not None
    if trained:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    else:
        print("No checkpoint — visualizing frozen (untrained) encoder features.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg["data"]["image_size"]

    # ── image / folder mode ──────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        files    = sorted(img_path.iterdir()) if img_path.is_dir() else [img_path]
        files    = [f for f in files if f.suffix.lower() in IMG_EXTS]

        all_results = []
        for fpath in tqdm(files, desc="Processing images"):
            tensor = load_image_tensor(str(fpath), image_size, device)
            r      = process_image(model, tensor, device)
            all_results.append(r)

            fig = make_e2e_figure(
                r["orig"], r["pca"],
                pca_after=r["pca"] if trained else None,
                seg_rgb=r["seg"],
                overlay=r["overlay"],
                title=fpath.name,
            )
            out_file = out_dir / (fpath.stem + "_e2e.png")
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out_file}")

        if args.grid and len(all_results) > 1:
            make_dataset_grid(all_results, str(out_dir / "grid.png"))

    # ── dataset mode ─────────────────────────────────────────────────────
    else:
        from torch.utils.data import DataLoader
        dataset = VOCSegmentationDataset(
            cfg["data"]["root"], split=args.dataset,
            image_size=image_size,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        all_results = []
        for idx, (images, masks) in enumerate(tqdm(loader, total=args.n_samples, desc="Visualizing")):
            if idx >= args.n_samples:
                break
            tensor = images.to(device)
            r = process_image(model, tensor, device, gt_mask=masks[0])
            all_results.append(r)

            fig = make_e2e_figure(
                r["orig"], r["pca"],
                pca_after=r["pca"] if trained else None,
                seg_rgb=r["seg"],
                overlay=r["overlay"],
                gt_rgb=r["gt"],
                title=f"VOC {args.dataset} sample {idx}",
            )
            out_file = out_dir / f"sample_{idx:04d}_e2e.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

        if args.grid:
            make_dataset_grid(all_results, str(out_dir / "grid.png"))
        print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
