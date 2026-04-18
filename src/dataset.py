"""
Dataset utilities for semantic segmentation.

Supports Pascal VOC 2012 out of the box (21 classes including background).
The SegmentationTransform class applies consistent spatial augmentations to
both image and mask, ensuring pixel correspondence is preserved.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Pascal VOC class metadata
# ---------------------------------------------------------------------------

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

VOC_COLORMAP = np.array([
    [0,   0,   0], [128, 0,   0], [0,   128, 0], [128, 128, 0],
    [0,   0,   128], [128, 0,   128], [0,   128, 128], [128, 128, 128],
    [64,  0,   0], [192, 0,   0], [64,  128, 0], [192, 128, 0],
    [64,  0,   128], [192, 0,   128], [64,  128, 128], [192, 128, 128],
    [0,   64,  0], [128, 64,  0], [0,   192, 0], [128, 192, 0],
    [0,   64,  128],
], dtype=np.uint8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Joint image+mask transform
# ---------------------------------------------------------------------------

class SegmentationTransform:
    """Applies paired spatial augmentations to an (image, mask) pair."""

    def __init__(self, image_size: int = 448, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        # Resize both with appropriate interpolation
        image = TF.resize(image, (self.image_size, self.image_size), T.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.image_size, self.image_size), T.InterpolationMode.NEAREST)

        if self.is_train:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if torch.rand(1).item() > 0.3:
                scale = torch.empty(1).uniform_(1.0, 2.0).item()
                new_h = int(self.image_size * scale)
                new_w = int(self.image_size * scale)
                image = TF.resize(image, (new_h, new_w), T.InterpolationMode.BILINEAR)
                mask  = TF.resize(mask,  (new_h, new_w), T.InterpolationMode.NEAREST)
                i, j, h, w = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
                image = TF.crop(image, i, j, h, w)
                mask  = TF.crop(mask,  i, j, h, w)

            image = self.color_jitter(image)

        image = self.normalize(TF.to_tensor(image))
        # VOC masks are palette PNGs; convert border pixels (255) to ignore_index
        mask_arr = np.array(mask, dtype=np.int64)
        mask_arr[mask_arr == 255] = 255          # keep 255 as ignore_index for loss
        mask_tensor = torch.from_numpy(mask_arr)
        return image, mask_tensor


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class VOCSegmentationDataset(Dataset):
    """Pascal VOC 2012 semantic segmentation dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",           # "train" | "val"
        image_size: int = 448,
        download: bool = False,
    ):
        self.transform = SegmentationTransform(image_size=image_size, is_train=(split == "train"))
        self.voc = VOCSegmentation(
            root=root,
            year="2012",
            image_set=split,
            download=download,
            transforms=None,
        )

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.voc[idx]
        return self.transform(image, mask)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def decode_segmap(mask: np.ndarray, colormap: np.ndarray = VOC_COLORMAP) -> np.ndarray:
    """Convert class-index mask (H, W) to an RGB image (H, W, 3)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colormap):
        rgb[mask == cls_idx] = color
    return rgb


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return a (H, W, 3) uint8 array."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
