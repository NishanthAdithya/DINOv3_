import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank residual path."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


def _set_nested_attr(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_keywords: tuple[str, ...] = ("query", "key", "value", "dense", "qkv", "proj"),
) -> nn.Module:
    """Replace matching nn.Linear layers in *model* with LoRALinear wrappers in-place."""
    replacements = {
        name: LoRALinear(module, rank=rank, alpha=alpha)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(kw in name for kw in target_keywords)
    }
    for name, lora_module in replacements.items():
        _set_nested_attr(model, name, lora_module)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA parameters — useful for lightweight checkpointing."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
