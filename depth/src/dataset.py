"""
NYU Depth v2 dataset loader.

Download the preprocessed HDF5 file (~2.8 GB) from:
  https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTbd  (Bhat et al.)

or the official zip from:
  https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

Place it at  data/nyu_depth_v2_labeled.mat  (MATLAB format)
or          data/nyu_depth_v2.h5            (HDF5 format — preferred)

The HDF5 file contains:
  /images : (1449, 3, 480, 640) uint8
  /depths : (1449, 480, 640)    float32  (metres, max ~10 m)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MAX_DEPTH     = 10.0   # metres


class DepthTransform:
    """Paired spatial augmentations for (RGB image, depth map)."""

    def __init__(self, image_size: int = 224, is_train: bool = True):
        self.image_size = image_size
        self.is_train   = is_train
        self.normalize  = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                                           saturation=0.3, hue=0.05)

    def __call__(
        self, image: Image.Image, depth: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:

        depth_pil = Image.fromarray(depth.astype(np.float32), mode="F")

        image     = TF.resize(image,     (self.image_size, self.image_size),
                               T.InterpolationMode.BILINEAR)
        depth_pil = TF.resize(depth_pil, (self.image_size, self.image_size),
                               T.InterpolationMode.NEAREST)

        if self.is_train:
            if torch.rand(1) > 0.5:
                image     = TF.hflip(image)
                depth_pil = TF.hflip(depth_pil)

            if torch.rand(1) > 0.5:
                image = self.color_jitter(image)

            # Random crop from a scaled-up version
            if torch.rand(1) > 0.4:
                scale = torch.empty(1).uniform_(1.0, 1.5).item()
                nh = int(self.image_size * scale)
                nw = int(self.image_size * scale)
                image     = TF.resize(image,     (nh, nw), T.InterpolationMode.BILINEAR)
                depth_pil = TF.resize(depth_pil, (nh, nw), T.InterpolationMode.NEAREST)
                i, j, h, w = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
                image     = TF.crop(image,     i, j, h, w)
                depth_pil = TF.crop(depth_pil, i, j, h, w)

        img_t   = self.normalize(TF.to_tensor(image))
        depth_t = torch.from_numpy(np.array(depth_pil, dtype=np.float32)).unsqueeze(0)
        depth_t = depth_t.clamp(1e-3, MAX_DEPTH)
        return img_t, depth_t


class NYUDepthDataset(Dataset):
    """NYU Depth v2 — loads from the preprocessed HDF5 file."""

    def __init__(
        self,
        h5_path: str = "data/nyu_depth_v2.h5",
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        import h5py
        self.transform = DepthTransform(image_size=image_size, is_train=(split == "train"))

        with h5py.File(h5_path, "r") as f:
            self.images = np.array(f["images"])   # (N, 3, H, W) uint8
            self.depths = np.array(f["depths"])   # (N, H, W)   float32

        N = len(self.images)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        val_n = int(N * val_fraction)
        if split == "val":
            self.idx = idx[:val_n]
        else:
            self.idx = idx[val_n:]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.idx[i]
        img   = Image.fromarray(self.images[n].transpose(1, 2, 0))  # HWC
        depth = self.depths[n]                                        # HW float32
        return self.transform(img, depth)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
