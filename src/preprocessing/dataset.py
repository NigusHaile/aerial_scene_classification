""" Dataset, DataLoaders and class-imbalance utilities.

AerialDataset reads an in-memory list of (path, label_idx) rows, applies an
Albumentations transform, and (optionally) caches decoded images in RAM. The
small dataset size (~2100 imgs) makes caching cheap and a big speed win.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class AerialDataset(Dataset):  
    """Image-classification dataset backed by a (path, label) list."""

    def __init__(self, rows: Sequence[Tuple[str, int]],
                 transform: Optional[Callable] = None,
                 cache: bool = False) -> None:
        self.rows = list(rows)
        self.transform = transform
        self.cache = cache
        self._store: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, idx: int) -> np.ndarray:
        if self.cache and idx in self._store:
            return self._store[idx]
        path, _ = self.rows[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.cache:
            self._store[idx] = img
        return img

    def __getitem__(self, idx: int):
        img = self._read(idx)
        label = self.rows[idx][1]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


def compute_class_weights(labels: Sequence[int], num_classes: int):
    """Inverse-frequency class weights, normalised to mean 1.0. For UC Merced this returns ~all-ones (balanced), but the code path is kept
    correct so that quarantining images never silently biases training.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: Sequence[int], num_classes: int):
    """A WeightedRandomSampler that equalises class sampling probability."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    per_class_w = 1.0 / counts
    sample_w = np.array([per_class_w[l] for l in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w), replacement=True,
    )


def export_split_images(splits: Dict, dest: str | Path = "data/split") -> None:
    """Copy images into dest/<train|val|test>/<class_name>/ folders.

    Creates a self-contained directory tree you can hand to any framework
    that expects the standard ImageFolder layout.
    """
    dest = Path(dest)
    inv = {v: k for k, v in splits["label_map"].items()}
    for split_name in ("train", "val", "test"):
        for path, label_idx in splits[split_name]:
            class_name = inv[label_idx]
            out_dir = dest / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_dir / Path(path).name)
    print(f"Split images exported to {dest}/")


def build_loaders(splits: Dict[str, List[Tuple[str, int]]], cfg,
                  train_tf, eval_tf,
                  use_weighted_sampler: bool = False,
                  worker_init_fn=None) -> Dict[str, "DataLoader"]:
    """Construct train/val/test DataLoaders.

    ``use_weighted_sampler`` is mutually exclusive with shuffling and is left
    off by default (the set is balanced)."""
    train_ds = AerialDataset(splits["train"], train_tf, cache=cfg.data.cache_images)
    val_ds = AerialDataset(splits["val"], eval_tf, cache=cfg.data.cache_images)
    test_ds = AerialDataset(splits["test"], eval_tf, cache=cfg.data.cache_images)

    common = dict(num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                  worker_init_fn=worker_init_fn)

    if use_weighted_sampler:
        labels = [l for _, l in splits["train"]]
        sampler = make_weighted_sampler(labels, cfg.project.num_classes)
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                  sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                                  shuffle=True, drop_last=True, **common)

    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                             shuffle=False, **common)
    return {"train": train_loader, "val": val_loader, "test": test_loader}
