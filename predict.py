"""
Run segmentation inference on a single image (or folder of images).

Usage
-----
python predict.py --image path/to/image.jpg --checkpoint checkpoints/best.pth
python predict.py --image path/to/folder/  --checkpoint checkpoints/best.pth --out_dir results/
python predict.py --image img.jpg --checkpoint ckpt.pth --show_pca
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import DINOSegmenter
from src.dataset import (
    VOC_CLASSES, VOC_COLORMAP,
    SegmentationTransform, denormalize, decode_segmap,
)
from src.utils import visualize_pca_features
from train import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def predict_image(
    model: DINOSegmenter,
    image_path: str,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Return (original_rgb, segmentation_rgb, patch_features)."""
    transform = SegmentationTransform(image_size=image_size, is_train=False)

    pil_img = Image.open(image_path).convert("RGB")
    dummy_mask = Image.fromarray(np.zeros((pil_img.height, pil_img.width), dtype=np.uint8))
    tensor, _ = transform(pil_img, dummy_mask)
    tensor = tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits   = model(tensor)
        features = model.get_patch_features(tensor)

    pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
    seg_rgb   = decode_segmap(pred_mask, VOC_COLORMAP)
    orig_rgb  = denormalize(tensor[0])

    return orig_rgb, seg_rgb, features


def save_figure(orig: np.ndarray, seg: np.ndarray, features: torch.Tensor, out_path: str, show_pca: bool):
    cols = 3 if show_pca else 2
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))

    axes[0].imshow(orig);  axes[0].set_title("Input");         axes[0].axis("off")
    axes[1].imshow(seg);   axes[1].set_title("Segmentation");  axes[1].axis("off")

    if show_pca:
        pca_rgb = visualize_pca_features(features)
        axes[2].imshow(pca_rgb); axes[2].set_title("PCA features"); axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Image file or folder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--out_dir",    default="predictions")
    parser.add_argument("--show_pca",   action="store_true")
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded {args.checkpoint}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = [p for p in image_path.iterdir() if p.suffix.lower() in IMG_EXTS]
    else:
        image_files = [image_path]

    print(f"Running inference on {len(image_files)} image(s)...")
    for img_path in image_files:
        orig, seg, features = predict_image(
            model, str(img_path), cfg["data"]["image_size"], device
        )
        out_file = out_dir / (img_path.stem + "_pred.png")
        save_figure(orig, seg, features, str(out_file), args.show_pca)

    print("Done.")


if __name__ == "__main__":
    main()
