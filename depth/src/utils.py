"""
Depth estimation utilities: SiLog loss, metrics, visualisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SiLogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic loss (Eigen et al., 2014).

    L = sqrt( mean(d_i^2) - lambda * mean(d_i)^2 )
    where d_i = log(pred_i) - log(gt_i)

    lambda=0.85 balances scale-invariance vs absolute accuracy.
    """

    def __init__(self, lambda_: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.lambda_ = lambda_
        self.eps     = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        valid = (gt > self.eps) & (pred > self.eps)
        d = torch.log(pred[valid] + self.eps) - torch.log(gt[valid] + self.eps)
        return torch.sqrt(d.pow(2).mean() - self.lambda_ * d.mean().pow(2))


class GradientLoss(nn.Module):
    """Edge-aware gradient loss to sharpen depth boundaries."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        valid = (gt > 1e-6).float()
        diff  = torch.log(pred.clamp(1e-6)) - torch.log(gt.clamp(1e-6))

        dx = torch.abs(diff[:, :, :, :-1] - diff[:, :, :, 1:]) * valid[:, :, :, :-1]
        dy = torch.abs(diff[:, :, :-1, :] - diff[:, :, 1:, :]) * valid[:, :, :-1, :]
        return dx.mean() + dy.mean()


class DepthLoss(nn.Module):
    def __init__(self, silog_w: float = 1.0, grad_w: float = 0.5):
        super().__init__()
        self.silog   = SiLogLoss()
        self.grad    = GradientLoss()
        self.silog_w = silog_w
        self.grad_w  = grad_w

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return self.silog_w * self.silog(pred, gt) + self.grad_w * self.grad(pred, gt)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class DepthMetrics:
    """Accumulates depth metrics across batches."""

    def __init__(self, max_depth: float = 10.0):
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self._abs_rel = []
        self._rmse    = []
        self._delta1  = []
        self._delta2  = []
        self._delta3  = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = pred.clamp(1e-3, self.max_depth)
        gt   = gt.clamp(1e-3, self.max_depth)
        valid = gt > 1e-3

        p = pred[valid]
        g = gt[valid]

        thresh = torch.maximum(p / g, g / p)
        self._abs_rel.append((torch.abs(p - g) / g).mean().item())
        self._rmse.append(torch.sqrt(((p - g) ** 2).mean()).item())
        self._delta1.append((thresh < 1.25   ).float().mean().item())
        self._delta2.append((thresh < 1.25**2).float().mean().item())
        self._delta3.append((thresh < 1.25**3).float().mean().item())

    def compute(self) -> dict[str, float]:
        return {
            "abs_rel": float(np.mean(self._abs_rel)),
            "rmse":    float(np.mean(self._rmse)),
            "delta1":  float(np.mean(self._delta1)),
            "delta2":  float(np.mean(self._delta2)),
            "delta3":  float(np.mean(self._delta3)),
        }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def colorize_depth(depth: np.ndarray, vmin: float = 0.0, vmax: float = 10.0,
                   cmap: str = "plasma") -> np.ndarray:
    """Convert a (H, W) depth array to an (H, W, 3) uint8 RGB image."""
    norm  = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    color = (cm.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)
    return color


def plot_depth_predictions(
    images: torch.Tensor,
    depths_gt: torch.Tensor,
    depths_pred: torch.Tensor,
    max_cols: int = 4,
) -> plt.Figure:
    """Plot (image | ground-truth depth | predicted depth | error) panels."""
    from .dataset import denormalize
    B = min(images.shape[0], max_cols)
    fig, axes = plt.subplots(4, B, figsize=(4 * B, 14))
    if B == 1:
        axes = axes[:, np.newaxis]

    for i in range(B):
        img_np  = denormalize(images[i])
        gt_np   = depths_gt[i, 0].cpu().numpy()
        pr_np   = depths_pred[i, 0].cpu().numpy()
        err_np  = np.abs(pr_np - gt_np)
        vmax    = gt_np.max()

        for ax, arr, title, kw in zip(
            axes[:, i],
            [img_np, gt_np, pr_np, err_np],
            ["Image", "Ground Truth", "Prediction", "|Error| (m)"],
            [{}, {"cmap":"plasma","vmin":0,"vmax":vmax},
               {"cmap":"plasma","vmin":0,"vmax":vmax},
               {"cmap":"hot",   "vmin":0,"vmax":1.0}],
        ):
            ax.imshow(arr, **kw)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    return fig


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    torch.save({
        "epoch": epoch,
        "metrics": metrics,
        "model_state": model.lora_state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt.get("metrics", {})
