"""Models — custom CNN, pretrained backbones, ViT+LoRA, and model utilities.

Combines:
  - CustomCNN  (residual + SE-attention CNN, Model A)
  - Pretrained (timm-based factory + backbone selection, Model B)
  - ViT + LoRA (parameter-efficient fine-tuning, Model C)
  - Model utilities (parameter counts, FLOPs, summary)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Custom CNN (Model A)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    """Two 3×3 conv-BN-ReLU layers + SE attention with an identity shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class CustomCNN(nn.Module):
    """Residual + attention CNN sized for ~5–10M params."""

    def __init__(self, num_classes: int = 21, widths=(64, 128, 256, 512),
                 drop_rate: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks.append(ResidualBlock(in_ch, w, stride=stride))
            blocks.append(ResidualBlock(w, w, stride=1))
            in_ch = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_custom_cnn(cfg) -> CustomCNN:
    return CustomCNN(num_classes=cfg.project.num_classes,
                     drop_rate=cfg.model.drop_rate)


# ============================================================================
# Pretrained backbone (Model B)
# ============================================================================

CANDIDATES: List[str] = [
    "convnext_tiny",
    "efficientnet_b3",
    "resnet50",
    "densenet121",
]


def build_pretrained(name: str, num_classes: int, pretrained: bool = True,
                     drop_rate: float = 0.2) -> nn.Module:
    """Create a timm backbone with a fresh classification head."""
    return timm.create_model(
        name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head = model.get_classifier()
    head_params = set(id(p) for p in head.parameters())
    for p in model.parameters():
        p.requires_grad = trainable or (id(p) in head_params)


def select_best_backbone(splits, cfg, train_one_fn,
                         candidates: List[str] | None = None) -> Tuple[str, dict]:
    """Probe candidate backbones; return (best_name, scores dict)."""
    candidates = candidates or CANDIDATES
    scores = {}
    for name in candidates:
        model = build_pretrained(name, cfg.project.num_classes,
                                 cfg.model.pretrained, cfg.model.drop_rate)
        scores[name] = float(train_one_fn(model, splits, cfg, epochs=2))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(scores, key=scores.get)
    return best, scores


# ===============
# ViT-B/16 + LoRA
# ===============

def build_vit_lora(cfg) -> Tuple[nn.Module, dict]:
    """Build a frozen ViT-B/16 with LoRA adapters on attention layers."""
    from peft import LoraConfig, get_peft_model

    base = timm.create_model(cfg.lora.base_model, pretrained=True,
                             num_classes=cfg.project.num_classes)

    def _count(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    for p in base.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        modules_to_save=["head"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    total, trainable = _count(model)
    return model, {
        "backend": "peft-lora",
        "r": cfg.lora.r, "alpha": cfg.lora.alpha,
        "total": total, "trainable": trainable,
        "trainable_pct": 100 * trainable / total,
    }


# ============================================================================
# Model utilities
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def estimate_flops(model: nn.Module, image_size: int = 224,
                   device: str = "cpu") -> float:
    """Estimate forward-pass GFLOPs; requires fvcore or thop."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis(model, dummy).total() / 1e9
    except Exception:
        pass
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return -1.0


def model_summary(model: nn.Module, image_size: int = 224) -> str:
    """Return a compact text summary (params + per-top-module breakdown)."""
    counts = count_parameters(model)
    lines = [
        f"Total params:     {counts['total']:,}",
        f"Trainable params: {counts['trainable']:,}",
        f"Frozen params:    {counts['frozen']:,}",
        "-" * 48,
        f"{'Module':<28}{'Params':>18}",
        "-" * 48,
    ]
    for name, module in model.named_children():
        p = sum(x.numel() for x in module.parameters())
        lines.append(f"{name:<28}{p:>18,}")
    return "\n".join(lines)
