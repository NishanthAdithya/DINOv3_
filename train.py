"""
Training script for DINOv2/v3 + LoRA semantic segmentation.

Usage
-----
python train.py                          # use configs/default.yaml
python train.py --config configs/default.yaml --epochs 30 --lr 2e-4
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.model import DINOSegmenter
from src.dataset import VOCSegmentationDataset
from src.utils import RunningMIoU, save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: DINOSegmenter, lr: float, wd: float) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    sched_name = cfg["training"].get("scheduler", "cosine")
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"].get("warmup_epochs", 0)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=warmup * steps_per_epoch,
    )
    if sched_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - warmup) * steps_per_epoch
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, main_scheduler],
        milestones=[warmup * steps_per_epoch],
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DINOSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
) -> float:
    model.train()
    amp_enabled = cfg["training"].get("amp", False) and device.type == "cuda"
    clip = cfg["training"].get("gradient_clip", None)

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False, unit="batch")

    for step, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)

        if amp_enabled:
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}", lr=f"{lr:.2e}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: DINOSegmenter,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    metric = RunningMIoU(num_classes)
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="        [val]  ", leave=False, unit="batch"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        metric.update(logits, masks)

    results = metric.compute()
    results["loss"] = total_loss / len(loader)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--resume",     type=str,   default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:     cfg["training"]["epochs"]     = args.epochs
    if args.lr:         cfg["training"]["lr"]          = args.lr
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- datasets -------------------------------------------------------
    data_cfg = cfg["data"]
    train_ds = VOCSegmentationDataset(data_cfg["root"], split="train",  image_size=data_cfg["image_size"])
    val_ds   = VOCSegmentationDataset(data_cfg["root"], split="val",    image_size=data_cfg["image_size"])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=data_cfg["num_workers"], pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"], pin_memory=pin,
    )

    # ---- model ----------------------------------------------------------
    enc_cfg = cfg["encoder"]
    model = DINOSegmenter(
        encoder_name=enc_cfg["name"],
        num_classes=cfg["decoder"]["num_classes"],
        patch_size=enc_cfg["patch_size"],
        lora_rank=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_keywords=tuple(cfg["lora"]["target_keywords"]),
        use_dinov3=enc_cfg.get("use_dinov3", False),
        dinov3_hub_repo=enc_cfg.get("dinov3_hub_repo", ""),
    ).to(device)

    model.print_summary()

    # ---- training setup -------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["training"]["ignore_index"])
    optimizer = build_optimizer(model, cfg["training"]["lr"], cfg["training"]["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler("cuda", enabled=cfg["training"].get("amp", False) and device.type == "cuda")

    start_epoch = 0
    best_miou   = 0.0

    if args.resume:
        from src.utils import load_checkpoint
        start_epoch, best_miou = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, mIoU {best_miou:.4f})")

    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop --------------------------------------------------
    epoch_pbar = tqdm(range(start_epoch, cfg["training"]["epochs"]), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, cfg, epoch + 1,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, cfg["decoder"]["num_classes"])

        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            mIoU=f"{val_metrics['miou']:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch+1:3d} | train {train_loss:.4f} "
            f"| val {val_metrics['loss']:.4f} "
            f"| mIoU {val_metrics['miou']:.4f} "
            f"| px_acc {val_metrics['pixel_accuracy']:.4f}"
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(model, optimizer, epoch + 1, best_miou, save_dir / "best.pth")
            tqdm.write(f"  -> New best mIoU: {best_miou:.4f}  (saved)")

        if (epoch + 1) % cfg["checkpoint"].get("save_every", 5) == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_metrics["miou"], save_dir / f"epoch_{epoch+1}.pth")

    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
