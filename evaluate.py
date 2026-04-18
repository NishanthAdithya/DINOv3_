"""
Evaluate a trained DINOv2/v3 + LoRA segmentation model on a validation set.

Usage
-----
python evaluate.py --checkpoint checkpoints/best.pth
python evaluate.py --checkpoint checkpoints/best.pth --config configs/default.yaml --split val
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import DINOSegmenter
from src.dataset import VOCSegmentationDataset, VOC_CLASSES, VOC_COLORMAP
from src.utils import RunningMIoU, plot_predictions, visualize_pca_grid
from train import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--split",      default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_viz",   type=str, default=None, help="path to save prediction figure")
    parser.add_argument("--save_pca",   type=str, default=None, help="path to save PCA feature figure")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- dataset --------------------------------------------------------
    data_cfg = cfg["data"]
    dataset  = VOCSegmentationDataset(
        data_cfg["root"], split=args.split, image_size=data_cfg["image_size"]
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=data_cfg["num_workers"], pin_memory=True,
    )

    # ---- model ----------------------------------------------------------
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
    print(f"Loaded checkpoint from {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # ---- evaluation -----------------------------------------------------
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["training"]["ignore_index"])
    num_classes = cfg["decoder"]["num_classes"]

    metric     = RunningMIoU(num_classes)
    total_loss = 0.0
    model.eval()

    viz_images = viz_masks = viz_logits = viz_pca_img = None
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Evaluating", unit="batch")):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            metric.update(logits, masks)

            if batch_idx == 0:
                viz_images  = images.cpu()
                viz_masks   = masks.cpu()
                viz_logits  = logits.cpu()
                viz_pca_img = images[:4].to(device)

    results = metric.compute()
    avg_loss = total_loss / len(loader)

    print(f"\n{'='*50}")
    print(f"Split        : {args.split}")
    print(f"Val loss     : {avg_loss:.4f}")
    print(f"Mean IoU     : {results['miou']:.4f}")
    print(f"Pixel Acc    : {results['pixel_accuracy']:.4f}")
    print(f"\nPer-class IoU:")
    for cls_name, iou in zip(VOC_CLASSES, results["iou_per_class"]):
        filled = int(iou * 20)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"  {cls_name:<15s} [{bar}] {iou:.3f}")

    # ---- optional visualizations ----------------------------------------
    if args.save_viz:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_predictions(viz_images, viz_masks, viz_logits, VOC_COLORMAP)
        fig.savefig(args.save_viz, dpi=150, bbox_inches="tight")
        print(f"Prediction figure saved to {args.save_viz}")

    if args.save_pca:
        import matplotlib
        matplotlib.use("Agg")
        fig = visualize_pca_grid(model, viz_pca_img)
        fig.savefig(args.save_pca, dpi=150, bbox_inches="tight")
        print(f"PCA feature figure saved to {args.save_pca}")


if __name__ == "__main__":
    main()
