""" Data validation, duplicate detection, outlier & quality analysis.

It:
1. Detects corrupt / unreadable / empty / wrong-format files.
2. Finds exact and near-duplicate images via perceptual hashing (pHash).
3. Flags statistical outliers on simple image descriptors
   (dimensions, brightness, contrast, entropy).
4. (Optionally) finds deep-feature outliers with IsolationForest + LOF on
   pretrained embeddings — only run when torch is available.
5. Runs quality checks: blur (variance of Laplacian), over/under-exposure,
   and grayscale anomalies.

Everything is reported as a pandas DataFrame and saved; rows are only dropped
when the corresponding ``remove_*`` flag is enabled in the config, so the
default behaviour is non-destructive and auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image



# Index building & corruption detection
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def scan_dataset(data_root: str | Path) -> pd.DataFrame:
    """Walk ``data_root/<class>/<image>`` and validate every file.

    Returns a DataFrame with columns: path, label, readable, reason.
    """
    data_root = Path(data_root)
    rows: List[Dict] = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for file in sorted(class_dir.iterdir()):
            readable, reason = _check_readable(file)
            rows.append(
                {"path": str(file), "label": label,
                 "readable": readable, "reason": reason}
            )
    return pd.DataFrame(rows)


def _check_readable(file: Path) -> Tuple[bool, str]:
    """Verify a single image file is a non-empty, decodable image."""
    if file.suffix.lower() not in VALID_EXTS:
        return False, "unsupported_format"
    try:
        if file.stat().st_size == 0:
            return False, "empty_file"
    except OSError:
        return False, "stat_error"
    try:
        with Image.open(file) as im:
            im.verify()  # cheap structural check
        # verify() invalidates the object; reopen to confirm pixels decode.
        with Image.open(file) as im:
            im.convert("RGB").load()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - we want any decode failure
        return False, f"unreadable:{type(exc).__name__}"



# Perceptual hashing for (near-)duplicate detection
def phash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute a DCT-based perceptual hash, returned as a boolean bit array."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude DC term from the median
    return (low > med).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two equal-length boolean hashes."""
    return int(np.count_nonzero(a != b))


def find_duplicates(df: pd.DataFrame, max_distance: int = 4) -> pd.DataFrame:
    """Group near-duplicate images by pairwise pHash Hamming distance.

    Returns a DataFrame of duplicate pairs: (path_a, path_b, distance).
    O(n^2) but trivially fine for ~2k images.
    """
    readable = df[df["readable"]].reset_index(drop=True)
    hashes: List[np.ndarray] = []
    for path in readable["path"]:
        img = _load_rgb(path)
        hashes.append(phash(img) if img is not None else None)

    pairs: List[Dict] = []
    n = len(hashes)
    for i in range(n):
        if hashes[i] is None:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            dist = hamming(hashes[i], hashes[j])
            if dist <= max_distance:
                pairs.append({
                    "path_a": readable.loc[i, "path"],
                    "path_b": readable.loc[j, "path"],
                    "label_a": readable.loc[i, "label"],
                    "label_b": readable.loc[j, "label"],
                    "distance": dist,
                })
    return pd.DataFrame(pairs)


# Statistical descriptors & quality checks
def _load_rgb(path: str | Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:  
        return None


def image_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-image descriptors used for outlier & quality analysis.
    Columns added: width, height, brightness, contrast, entropy,blur_var, is_grayscale."""
    records: List[Dict] = []
    for _, row in df[df["readable"]].iterrows():
        img = _load_rgb(row["path"])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        prob = hist / (hist.sum() + 1e-9)
        entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())
        # Grayscale anomaly: an "RGB" image whose channels are ~identical.
        channel_spread = float(np.mean(np.std(img.astype(np.float32), axis=2)))
        records.append({
            "path": row["path"],
            "label": row["label"],
            "width": w,
            "height": h,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "entropy": entropy,
            "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "is_grayscale": channel_spread < 1.0,
        })
    return pd.DataFrame(records)


def flag_quality_issues(desc: pd.DataFrame, cfg_quality) -> pd.DataFrame:
    """Add boolean quality-flag columns based on configured thresholds."""
    out = desc.copy()
    out["flag_blurry"] = out["blur_var"] < cfg_quality.blur_var_threshold
    out["flag_dark"] = out["brightness"] < cfg_quality.dark_threshold
    out["flag_bright"] = out["brightness"] > cfg_quality.bright_threshold
    out["flag_grayscale"] = out["is_grayscale"]
    out["flag_any"] = out[["flag_blurry", "flag_dark",
                           "flag_bright", "flag_grayscale"]].any(axis=1)
    return out


def statistical_outliers(desc: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Flag rows whose descriptor is a >z-sigma outlier within its class."""
    out = desc.copy()
    feats = ["brightness", "contrast", "entropy", "blur_var"]
    out["stat_outlier"] = False
    for _, idx in out.groupby("label").groups.items():
        sub = out.loc[idx, feats]
        zscores = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
        out.loc[idx, "stat_outlier"] = (zscores.abs() > z).any(axis=1).values
    return out


def deep_feature_outliers(df: pd.DataFrame, cfg) -> Optional[pd.DataFrame]:
    """Extract pretrained embeddings and flag outliers (IsolationForest + LOF)."""
    import timm
    import torch
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).eval().to(device)
    mean = np.array(cfg.data.mean, dtype=np.float32)
    std = np.array(cfg.data.std, dtype=np.float32)

    readable = df[df["readable"]].reset_index(drop=True)
    feats, kept = [], []
    with torch.no_grad():
        for path in readable["path"]:
            img = _load_rgb(path)
            if img is None:
                continue
            x = cv2.resize(img, (cfg.data.image_size, cfg.data.image_size))
            x = (x.astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            feats.append(model(x).cpu().numpy().ravel())
            kept.append(path)
    feats = np.stack(feats)

    iso = IsolationForest(contamination=cfg.quality.iso_contamination,
                          random_state=cfg.project.seed).fit_predict(feats)
    lof = LocalOutlierFactor(n_neighbors=cfg.quality.lof_neighbors).fit_predict(feats)
    return pd.DataFrame({
        "path": kept,
        "iso_outlier": iso == -1,
        "lof_outlier": lof == -1,
    })


def build_clean_index(df: pd.DataFrame, quality: pd.DataFrame,
                      duplicates: pd.DataFrame, cfg) -> pd.DataFrame:
    """Assemble the final clean (path,label) index respecting removal flags."""
    clean = df[df["readable"]].copy()
    drop_paths: set = set()

    if cfg.quality.remove_outliers and not quality.empty:
        drop_paths |= set(quality.loc[quality["flag_any"], "path"])
        if "stat_outlier" in quality:
            drop_paths |= set(quality.loc[quality["stat_outlier"], "path"])

    # For each duplicate pair we keep the first, quarantine the second.
    if cfg.quality.remove_outliers and not duplicates.empty:
        drop_paths |= set(duplicates["path_b"])

    clean = clean[~clean["path"].isin(drop_paths)].reset_index(drop=True)
    return clean[["path", "label"]]


def run_validation(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full validation pipeline: scan → descriptors → quality → clean index.

    Returns ``(clean, desc)`` so callers have both for downstream EDA.
    """
    df = scan_dataset(cfg.paths.data_root)
    print(f"Scanned {len(df)} files | readable {int(df.readable.sum())}")
    desc = image_descriptors(df)
    quality = flag_quality_issues(desc, cfg.quality)
    quality = statistical_outliers(quality)
    duplicates = find_duplicates(df, cfg.quality.phash_distance)
    print(f"Quality flags: {int(quality.flag_any.sum())} | duplicate pairs: {len(duplicates)}")
    clean = build_clean_index(df, quality, duplicates, cfg)
    clean.to_csv(cfg.paths.clean_index, index=False)
    quality.to_csv(Path(cfg.paths.outputs) / "quality_report.csv", index=False)
    cfg.project.class_names = sorted(clean["label"].unique())
    return clean, desc
