"""
DINOv2 / DINOv3 encoder + LoRA + 1x1 conv decoder for semantic segmentation.

Encoder backbone is frozen; only LoRA adapter weights and the decoder head
are trained.  At inference the decoder output is bilinearly upsampled back to
the input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import apply_lora, lora_state_dict, count_parameters


# ---------------------------------------------------------------------------
# Helper: load encoder
# ---------------------------------------------------------------------------

def _load_dinov2(model_name: str) -> tuple[nn.Module, int]:
    """Load DINOv2 via HuggingFace Transformers. Returns (model, hidden_dim)."""
    from transformers import Dinov2Model
    encoder = Dinov2Model.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size
    return encoder, hidden_size


def _load_dinov3(hub_repo: str, model_name: str) -> tuple[nn.Module, int]:
    """
    Load DINOv3 (Siméoni et al., 2025) via torch.hub.

    Replace hub_repo with the official repository path once published, e.g.:
        'siméoni/dinov3'
    The model must expose `.embed_dim` and return patch tokens from forward().
    """
    encoder = torch.hub.load(hub_repo, model_name, pretrained=True)
    hidden_size = encoder.embed_dim
    return encoder, hidden_size


# ---------------------------------------------------------------------------
# Segmentation model
# ---------------------------------------------------------------------------

class DINOSegmenter(nn.Module):
    """
    Semantic segmentation head on top of a frozen DINOv2/v3 ViT encoder.

    Architecture
    ============
    1. Frozen ViT encoder (DINOv2 or DINOv3)
    2. LoRA adapters injected into every QKV / projection linear layer
    3. Single 1×1 convolution decoder: hidden_dim → num_classes
    4. Bilinear upsampling to input resolution
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov2-base",
        num_classes: int = 21,
        patch_size: int = 14,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_keywords: tuple[str, ...] = ("query", "key", "value", "dense"),
        use_dinov3: bool = False,
        dinov3_hub_repo: str = "author/dinov3",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_dinov3 = use_dinov3

        # ---- load encoder ------------------------------------------------
        if use_dinov3:
            self.encoder, hidden_size = _load_dinov3(dinov3_hub_repo, encoder_name)
        else:
            self.encoder, hidden_size = _load_dinov2(encoder_name)

        # ---- freeze encoder ----------------------------------------------
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- inject LoRA -------------------------------------------------
        apply_lora(self.encoder, rank=lora_rank, alpha=lora_alpha, target_keywords=lora_keywords)

        # ---- decoder head ------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 2),
            nn.GELU(),
            nn.Conv2d(hidden_size // 2, num_classes, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder, return spatial patch tokens (B, C, h, w)."""
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        if self.use_dinov3:
            # DINOv3 hub models typically return (B, 1+num_patches, C)
            tokens = self.encoder(x)
            if isinstance(tokens, torch.Tensor):
                patch_tokens = tokens[:, 1:]          # drop CLS
            else:
                patch_tokens = tokens.last_hidden_state[:, 1:]
        else:
            out = self.encoder(x)
            patch_tokens = out.last_hidden_state[:, 1:]  # drop CLS

        C = patch_tokens.shape[-1]
        return patch_tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B,C,h,w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        patch_tokens = self._extract_patch_tokens(x)      # (B, C, h, w)
        logits = self.decoder(patch_tokens)               # (B, num_classes, h, w)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw patch features before the decoder (for PCA viz)."""
        with torch.no_grad():
            return self._extract_patch_tokens(x)

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v for k, v in self.state_dict().items()
            if "lora_" in k or k.startswith("decoder.")
        }

    def parameter_summary(self) -> dict[str, int]:
        return count_parameters(self)

    def print_summary(self) -> None:
        stats = self.parameter_summary()
        print(f"Total parameters  : {stats['total']:,}")
        print(f"Trainable (LoRA+dec): {stats['trainable']:,}")
        print(f"Frozen (encoder)  : {stats['frozen']:,}")
        ratio = 100 * stats["trainable"] / stats["total"]
        print(f"Trainable ratio   : {ratio:.2f}%")
