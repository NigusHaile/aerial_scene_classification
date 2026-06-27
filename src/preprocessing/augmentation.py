"""Augmentation pipelines (Albumentations).

Two transforms are exposed:

* train_transform: strong, probability-tuned augmentation suited to
  overhead/aerial imagery. Crucially it includes VerticalFlip and
  RandomRotate90 because aerial scenes have **no canonical up direction**,
  so these are label-preserving (unlike natural-photo datasets).
* val_transform: deterministic resize + normalise only, so validation
  and test metrics are stable and leakage-free.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transform(cfg) -> A.Compose:
    """Build the training augmentation pipeline from config probabilities."""
    a = cfg.augment
    size = cfg.data.image_size
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(a.rrc_scale_min, 1.0),
                            ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=a.hflip_p),
        A.VerticalFlip(p=a.vflip_p),                 # valid for aerial imagery
        A.RandomRotate90(p=a.rotate90_p),            # valid for aerial imagery
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=20, border_mode=0,
                           p=a.shift_scale_rotate_p),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                   p=a.brightness_contrast_p),
        A.CLAHE(clip_limit=2.0, p=a.clahe_p),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                      hue=0.05, p=a.color_jitter_p),
        A.GaussNoise(p=a.gauss_noise_p),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=a.grid_distortion_p),
        A.CoarseDropout(p=a.coarse_dropout_p),       # ~ Cutout regularisation
        A.Normalize(mean=cfg.data.mean, std=cfg.data.std),
        ToTensorV2(),
    ])


def val_transform(cfg) -> A.Compose:
    """Deterministic evaluation transform: resize + normalise only."""
    size = cfg.data.image_size
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=cfg.data.mean, std=cfg.data.std),
        ToTensorV2(),
    ])
