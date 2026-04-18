"""
Monocular depth estimation using a frozen DINOv2 encoder + LoRA + lightweight decoder.

The decoder progressively upsamples patch tokens (16×16 @ 768-dim for a 224-px input)
back to the full image resolution, predicting a single-channel depth map.

Reuses src/lora.py from the parent project — run from the repo root or add .. to path.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from src.lora import apply_lora, lora_state_dict, count_parameters


class DepthDecoder(nn.Module):
    """
    Progressive upsampling decoder: patch grid → full-resolution depth map.

    Stages (for 224-px input, patch_size=14, h=w=16):
      16×16 → 32×32 → 64×64 → 112×112 → 224×224
    """

    def __init__(self, in_channels: int = 768, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth

        def up_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.up1 = up_block(in_channels, 256)   # 16→32
        self.up2 = up_block(256, 128)            # 32→64
        self.up3 = up_block(128, 64)             # 64→112
        self.up4 = up_block(64, 32)              # 112→224
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, h, w)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)
        depth = torch.sigmoid(self.head(x)) * self.max_depth   # (B, 1, H, W)
        return depth


class DepthEstimator(nn.Module):
    """
    DINOv2 + LoRA monocular depth estimator.

    Frozen DINOv2-Base backbone with LoRA adapters in every Q/K/V/dense layer,
    followed by a progressive-upsampling depth decoder.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov2-base",
        patch_size: int = 14,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_keywords: tuple[str, ...] = ("query", "key", "value", "dense"),
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_depth  = max_depth

        self.encoder = Dinov2Model.from_pretrained(encoder_name)
        hidden_size  = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        apply_lora(self.encoder, rank=lora_rank, alpha=lora_alpha,
                   target_keywords=lora_keywords)

        self.decoder = DepthDecoder(in_channels=hidden_size, max_depth=max_depth)

    # ------------------------------------------------------------------
    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        out = self.encoder(x)
        tokens = out.last_hidden_state[:, 1:]        # drop CLS
        C = tokens.shape[-1]
        return tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B,C,h,w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._patch_tokens(x)               # (B, C, h, w)
        depth  = self.decoder(tokens)                # (B, 1, H, W)
        H, W   = x.shape[-2:]
        return F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._patch_tokens(x)

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v for k, v in self.state_dict().items()
            if "lora_" in k or k.startswith("decoder.")
        }

    def print_summary(self) -> None:
        stats = count_parameters(self)
        print(f"Total parameters    : {stats['total']:,}")
        print(f"Trainable (LoRA+dec): {stats['trainable']:,}")
        print(f"Frozen (encoder)    : {stats['frozen']:,}")
        print(f"Trainable ratio     : {100*stats['trainable']/stats['total']:.2f}%")
