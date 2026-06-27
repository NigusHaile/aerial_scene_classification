""" Stratified, reproducible, leakage-free data splitting.

The split is computed once and **frozen to JSON**. Every model (Custom CNN,
ConvNeXt, ViT+LoRA) and every Optuna trial reads the same frozen split, which
guarantees fair comparison and prevents accidental leakage between phases.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def make_label_map(df: pd.DataFrame) -> Dict[str, int]:
    """Deterministic class-name -> index mapping (sorted for stability)."""
    classes = sorted(df["label"].unique())
    return {c: i for i, c in enumerate(classes)}


def stratified_split(df: pd.DataFrame, cfg) -> Dict[str, List[Tuple[str, int]]]:
    """Return train/val/test as lists of (path, label_idx), class-balanced.

    Two-stage stratified split: first carve out the test set, then split the
    remainder into train/val — all stratified by label so every split keeps the
    original class proportions.
    """
    label_map = make_label_map(df)
    df = df.copy()
    df["y"] = df["label"].map(label_map)

    test_size = cfg.data.test_frac
    val_size = cfg.data.val_frac / (1.0 - test_size)  # relative to remainder

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["y"],
        random_state=cfg.project.seed,
    )
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["y"],
        random_state=cfg.project.seed,
    )

    to_rows = lambda d: list(zip(d["path"].tolist(), d["y"].tolist()))
    return {
        "train": to_rows(train),
        "val": to_rows(val),
        "test": to_rows(test),
        "label_map": label_map,  # type: ignore[dict-item]
    }


def save_splits(splits: Dict, path: str | Path,
                image_dest: str | Path | None = "data/split") -> None:
    """Freeze the split to JSON and copy images into an ImageFolder tree.

    ``image_dest`` is the root for <split>/<class>/ folders.
    Pass ``image_dest=None`` to skip image export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle)

    if image_dest is not None:
        dest = Path(image_dest)
        inv = {v: k for k, v in splits["label_map"].items()}
        for split_name in ("train", "val", "test"):
            for img_path, label_idx in splits[split_name]:
                out_dir = dest / split_name / inv[label_idx]
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_dir / Path(img_path).name)
        print(f"Split images exported to {dest}/")


def load_splits(path: str | Path) -> Dict:
    """Load a previously frozen split (lists become tuples for the Dataset)."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    for key in ("train", "val", "test"):
        raw[key] = [tuple(r) for r in raw[key]]
    return raw


def split_distribution(splits: Dict) -> pd.DataFrame:
    """Per-split class counts to verify balance is preserved (no leakage)."""
    inv = {v: k for k, v in splits["label_map"].items()}
    frames = []
    for name in ("train", "val", "test"):
        counts = pd.Series([inv[y] for _, y in splits[name]]).value_counts()
        frames.append(counts.rename(name))
    table = pd.concat(frames, axis=1).fillna(0).astype(int).sort_index()
    table["total"] = table.sum(axis=1)
    return table


def assert_no_leakage(splits: Dict) -> None:
    """Hard guarantee that no image path appears in more than one split."""
    tr = {p for p, _ in splits["train"]}
    va = {p for p, _ in splits["val"]}
    te = {p for p, _ in splits["test"]}
    assert tr.isdisjoint(va), "Leakage: train ∩ val"
    assert tr.isdisjoint(te), "Leakage: train ∩ test"
    assert va.isdisjoint(te), "Leakage: val ∩ test"


def run_split(cfg, clean: pd.DataFrame) -> Dict:
    """Compute and freeze the stratified split; export images and distribution CSV."""
    splits = stratified_split(clean, cfg)
    assert_no_leakage(splits)
    save_splits(splits, cfg.paths.splits)
    dist = split_distribution(splits)
    dist.to_csv(Path(cfg.paths.outputs) / "split_distribution.csv")
    print(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    return splits
