"""
Evaluation metrics, PCA feature visualization, and plotting helpers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA feature visualization  (the colorful figures in the README)
# ---------------------------------------------------------------------------

def visualize_pca_features(
    features: torch.Tensor,
    n_components: int = 3,
    bg_threshold: float | None = 0.3,
) -> np.ndarray:
    """
    Project patch features to 3 PCA components and return an RGB heatmap.

    Parameters
    ----------
    features : (B, C, h, w) or (C, h, w) tensor of patch features
    n_components : number of PCA components (3 → RGB)
    bg_threshold : if set, pixels whose first component is below this
                   value are treated as background and painted black.

    Returns
    -------
    rgb : (h, w, 3) float32 array in [0, 1]
    """
    if features.dim() == 4:
        features = features[0]           # take first batch item

    C, h, w = features.shape
    flat = features.permute(1, 2, 0).reshape(-1, C).cpu().float().numpy()

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(flat)

    # min-max normalise each component independently
    lo = components.min(axis=0)
    hi = components.max(axis=0)
    components = (components - lo) / (hi - lo + 1e-8)

    rgb = components.reshape(h, w, n_components).astype(np.float32)

    if bg_threshold is not None:
        bg_mask = rgb[..., 0] < bg_threshold
        rgb[bg_mask] = 0.0

    return rgb


def visualize_pca_grid(
    model,
    images: torch.Tensor,
    titles: list[str] | None = None,
    n_components: int = 3,
) -> plt.Figure:
    """Plot PCA feature maps for a batch of images side-by-side."""
    model.eval()
    with torch.no_grad():
        features = model.get_patch_features(images)

    n = images.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        from .dataset import denormalize
        img_np = denormalize(images[i])
        pca_np = visualize_pca_features(features[i : i + 1], n_components=n_components)

        axes[0, i].imshow(img_np)
        axes[0, i].axis("off")
        axes[0, i].set_title(titles[i] if titles else f"Image {i}")

        axes[1, i].imshow(pca_np)
        axes[1, i].axis("off")
        axes[1, i].set_title("PCA features")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """
    Compute mean Intersection-over-Union.

    Parameters
    ----------
    pred   : (B, num_classes, H, W) raw logits or (B, H, W) class indices
    target : (B, H, W) ground-truth class indices
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    ious: list[float] = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        valid = target != ignore_index

        intersection = (p & t & valid).sum().item()
        union = ((p | t) & valid).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


class RunningMIoU:
    """Accumulates confusion matrix entries across batches for exact mIoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        pred = pred.cpu().numpy().ravel()
        gt   = target.cpu().numpy().ravel()
        valid = gt != self.ignore_index
        pred, gt = pred[valid], gt[valid]
        np.add.at(self.confusion, (gt, pred), 1)

    def compute(self) -> dict[str, float]:
        diag = np.diag(self.confusion)
        row_sum = self.confusion.sum(axis=1)
        col_sum = self.confusion.sum(axis=0)
        union = row_sum + col_sum - diag
        iou_per_class = np.where(union > 0, diag / union, np.nan)
        miou = float(np.nanmean(iou_per_class))
        pixel_acc = float(diag.sum() / self.confusion.sum())
        return {"miou": miou, "pixel_accuracy": pixel_acc, "iou_per_class": iou_per_class.tolist()}

    def reset(self) -> None:
        self.confusion[:] = 0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_predictions(
    images: torch.Tensor,
    masks_gt: torch.Tensor,
    logits: torch.Tensor,
    colormap: np.ndarray,
    max_cols: int = 4,
) -> plt.Figure:
    """Plot (image | ground truth | prediction) triplets."""
    from .dataset import denormalize, decode_segmap

    B = min(images.shape[0], max_cols)
    fig, axes = plt.subplots(3, B, figsize=(4 * B, 12))
    if B == 1:
        axes = axes[:, np.newaxis]

    preds = logits.argmax(dim=1).cpu().numpy()

    for i in range(B):
        img_np = denormalize(images[i])
        gt_np  = decode_segmap(masks_gt[i].cpu().numpy(), colormap)
        pr_np  = decode_segmap(preds[i], colormap)

        for ax, arr, title in zip(
            axes[:, i], [img_np, gt_np, pr_np], ["Image", "Ground truth", "Prediction"]
        ):
            ax.imshow(arr)
            ax.axis("off")
            ax.set_title(title)

    plt.tight_layout()
    return fig


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    miou: float,
    path: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "miou": miou,
            "model_state": model.lora_state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path: str, device: str = "cpu") -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"], ckpt["miou"]
