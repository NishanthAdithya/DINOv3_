"""
Training script for DINOv2 + LoRA monocular depth estimation on NYU Depth v2.

Usage:
    python train.py                          # default config
    python train.py --config configs/default.yaml
    python train.py --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# allow importing from repo root (for src/lora.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model   import DepthEstimator
from src.dataset import NYUDepthDataset
from src.utils   import DepthLoss, DepthMetrics, save_checkpoint

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/default.yaml")
    p.add_argument("--epochs",    type=int,   default=None)
    p.add_argument("--batch_size",type=int,   default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--resume",    default=None, help="path to checkpoint to resume from")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override(cfg: dict, args) -> dict:
    if args.epochs     is not None: cfg["training"]["epochs"]     = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.lr         is not None: cfg["training"]["lr"]          = args.lr
    return cfg


# ---------------------------------------------------------------------------

def build_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_ds = NYUDepthDataset(
        h5_path      = data_cfg["h5_path"],
        split        = "train",
        image_size   = data_cfg["image_size"],
        val_fraction = data_cfg["val_fraction"],
    )
    val_ds = NYUDepthDataset(
        h5_path      = data_cfg["h5_path"],
        split        = "val",
        image_size   = data_cfg["image_size"],
        val_fraction = data_cfg["val_fraction"],
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = data_cfg["num_workers"],
        pin_memory  = pin,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = data_cfg["num_workers"],
        pin_memory  = pin,
    )
    return train_loader, val_loader


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t_cfg  = cfg["training"]
    name   = t_cfg.get("scheduler", "cosine")
    warmup = t_cfg.get("warmup_epochs", 2)
    epochs = t_cfg["epochs"]
    total  = epochs * steps_per_epoch

    if name == "cosine":
        warmup_steps   = warmup * steps_per_epoch
        cosine_steps   = max(total - warmup_steps, 1)
        warmup_sched   = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers   = [warmup_sched, cosine_sched],
            milestones   = [warmup_steps],
        )
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=total, gamma=1.0)


# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, scaler, amp_enabled: bool, log_every: int,
                    epoch: int) -> float:
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch:>3} [train]", leave=False, dynamic_ncols=True)

    for step, (imgs, depths) in enumerate(bar, 1):
        imgs   = imgs.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            pred = model(imgs)
            loss = criterion(pred, depths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled: bool,
             max_depth: float) -> tuple[float, dict]:
    model.eval()
    metrics  = DepthMetrics(max_depth=max_depth)
    total    = 0.0
    bar      = tqdm(loader, desc="             [val]  ", leave=False, dynamic_ncols=True)

    for imgs, depths in bar:
        imgs   = imgs.to(device)
        depths = depths.to(device)
        with autocast("cuda", enabled=amp_enabled):
            pred = model(imgs)
            loss = criterion(pred, depths)
        total += loss.item()
        metrics.update(pred, depths)

    return total / len(loader), metrics.compute()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = override(load_config(args.config), args)

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = cfg["training"].get("amp", False) and device == "cuda"
    print(f"Device : {device}  |  AMP : {amp_enabled}")

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
    model.print_summary()

    # Optimiser + loss
    t_cfg     = cfg["training"]
    criterion = DepthLoss(silog_w=t_cfg["silog_weight"], grad_w=t_cfg["grad_weight"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"],
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # Data
    train_loader, val_loader = build_loaders(cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Resume
    start_epoch = 1
    if args.resume:
        from src.utils import load_checkpoint
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # Checkpoint dir
    ckpt_dir  = Path(cfg["checkpoint"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = cfg["checkpoint"]["save_every"]
    log_every  = cfg["logging"]["log_every"]
    max_depth  = cfg["depth"]["max_depth"]

    best_abs_rel = float("inf")

    # ── Training loop ───────────────────────────────────────────────────────
    epoch_bar = tqdm(range(start_epoch, t_cfg["epochs"] + 1),
                     desc="Training", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_bar:
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, device, scaler, amp_enabled,
                                     log_every, epoch)
        val_loss, m = validate(model, val_loader, criterion, device,
                               amp_enabled, max_depth)

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            abs_rel=f"{m['abs_rel']:.4f}",
            d1=f"{m['delta1']:.3f}",
        )
        tqdm.write(
            f"Epoch {epoch:>3}/{t_cfg['epochs']} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"AbsRel {m['abs_rel']:.4f} | RMSE {m['rmse']:.4f} | "
            f"d1 {m['delta1']:.3f} | d2 {m['delta2']:.3f} | d3 {m['delta3']:.3f} | "
            f"{time.time()-t0:.0f}s"
        )

        # Save checkpoint
        if epoch % save_every == 0:
            path = ckpt_dir / f"depth_ep{epoch:03d}.pt"
            save_checkpoint(model, optimizer, epoch, m, str(path))
            tqdm.write(f"  Saved {path}")

        # Best model
        if m["abs_rel"] < best_abs_rel:
            best_abs_rel = m["abs_rel"]
            save_checkpoint(model, optimizer, epoch, m, str(ckpt_dir / "depth_best.pt"))
            tqdm.write(f"  New best AbsRel = {best_abs_rel:.4f}")

    print("\nTraining complete.")
    print(f"Best AbsRel : {best_abs_rel:.4f}")
    print(f"Checkpoints : {ckpt_dir}")


if __name__ == "__main__":
    main()
