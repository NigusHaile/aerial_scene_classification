"""Exploratory Data Analysis.

Produces and saves publication-quality figures: class distribution, sample
montage, RGB channel distributions, resolution scatter, descriptor correlations,
and embedding projections (PCA / t-SNE). Every figure is written to
``results_figures/`"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import save_fig

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def plot_class_distribution(df: pd.DataFrame, fig_dir: str | Path) -> None:
    """Horizontal count bar, imbalance deviation, and cumulative coverage."""
    counts = df["label"].value_counts().sort_values(ascending=True)
    palette = sns.color_palette("viridis", len(counts))

    # Horizontal bar class names readable without rotation
    fig, ax = plt.subplots(figsize=(9, 10))
    bars = ax.barh(counts.index, counts.values,
                   color=palette, edgecolor="none")
    for bar, v in zip(bars, counts.values):
        ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    ax.set_xlabel("Image count")
    ax.set_title("Images per class")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, Path(fig_dir) / "eda_class_distribution_bar.png")

    # 2. Deviation from mean — highlights under/over-represented classes
    mean = counts.mean()
    deviation = counts - mean
    colors_dev = ["#d62728" if d < 0 else "#2ca02c" for d in deviation.values]
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(deviation.index, deviation.values, color=colors_dev, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Deviation from mean ({mean:.0f} images)")
    ax.set_title("Class balance: deviation from mean count")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, Path(fig_dir) / "eda_class_imbalance.png")

    # 3. Cumulative coverage — how many classes cover X% of the data
    sorted_desc = counts.sort_values(ascending=False)
    cumulative = sorted_desc.cumsum() / sorted_desc.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(cumulative) + 1), cumulative.values,
            marker="o", markersize=4, linewidth=1.5, color="#1f77b4")
    ax.axhline(80, color="orange", linestyle="--", linewidth=1, label="80%")
    ax.axhline(95, color="red", linestyle="--", linewidth=1, label="95%")
    ax.set_xlabel("Number of classes (ranked by frequency)")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_title("Cumulative class coverage")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, Path(fig_dir) / "eda_class_cumulative.png")


def plot_sample_grid(df: pd.DataFrame, fig_dir: str | Path,
                     n: int = 25, seed: int = 42) -> None:
    """Show a 5x5 montage of random images with their labels."""
    sample = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    side = int(np.ceil(np.sqrt(len(sample))))
    fig, axes = plt.subplots(side, side, figsize=(14, 14))
    for ax in axes.ravel():
        ax.axis("off")
    for i, row in sample.iterrows():
        img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
        ax = axes.ravel()[i]
        ax.imshow(img)
        ax.set_title(row["label"], fontsize=8)
    fig.suptitle("Random sample images", fontsize=16)
    save_fig(fig, Path(fig_dir) / "eda_sample_grid.png")


def plot_rgb_distribution(df: pd.DataFrame, fig_dir: str | Path,
                          n: int = 300, seed: int = 42) -> None:
    """Average per-channel intensity histogram over a random subset."""
    sample = df.sample(min(n, len(df)), random_state=seed)
    acc = np.zeros((3, 256), dtype=np.float64)
    for path in sample["path"]:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        for c in range(3):
            acc[c] += cv2.calcHist([img], [c], None, [256], [0, 256]).ravel()
    acc /= len(sample)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, color in enumerate(["red", "green", "blue"]):
        ax.plot(acc[c], color=color, label=color.capitalize(), alpha=0.8)
    ax.set_title("Average RGB intensity distribution")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Mean frequency")
    ax.legend()
    save_fig(fig, Path(fig_dir) / "eda_rgb_distribution.png")


def plot_resolution(desc: pd.DataFrame, fig_dir: str | Path) -> None:
    """Scatter of image widths vs heights (reveals off-256 outliers)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    jitter = lambda v: v + np.random.uniform(-1.5, 1.5, len(v))
    ax.scatter(jitter(desc["width"]), jitter(desc["height"]),
               alpha=0.3, s=18, edgecolor="none")
    ax.set_title("Image resolution distribution")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    save_fig(fig, Path(fig_dir) / "eda_resolution.png")


def plot_descriptor_correlation(desc: pd.DataFrame, fig_dir: str | Path) -> None:
    """Correlation heatmap among image descriptors."""
    feats = ["width", "height", "brightness", "contrast", "entropy", "blur_var"]
    corr = desc[feats].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Descriptor correlation")
    save_fig(fig, Path(fig_dir) / "eda_descriptor_correlation.png")


def _extract_embeddings(df: pd.DataFrame, cfg, n_per_class: int = 20):
    """Pretrained ResNet18 embeddings for a class-balanced subset (for EDA)."""
    try:
        import timm
        import torch
    except ImportError:
        return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet18", pretrained=True, num_classes=0).eval().to(device)
    mean = np.array(cfg.data.mean, np.float32)
    std = np.array(cfg.data.std, np.float32)
    # groupby().apply() dropped the groupby column in pandas 3.0; iterate instead
    sub = pd.concat([
        grp.sample(min(n_per_class, len(grp)), random_state=cfg.project.seed)
        for _, grp in df.groupby("label")
    ])
    feats, labels = [], []
    with torch.no_grad():
        for _, row in sub.iterrows():
            img = cv2.cvtColor(cv2.imread(row["path"]), cv2.COLOR_BGR2RGB)
            x = cv2.resize(img, (cfg.data.image_size, cfg.data.image_size))
            x = (x.astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            feats.append(model(x).cpu().numpy().ravel())
            labels.append(row["label"])
    return np.stack(feats), np.array(labels)


def plot_embeddings(df: pd.DataFrame, cfg, fig_dir: str | Path) -> None:
    """PCA, t-SNE, and (if installed) UMAP projections of deep features."""
    feats, labels = _extract_embeddings(df, cfg)
    if feats is None:
        return
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    classes = sorted(set(labels))
    palette = dict(zip(classes, sns.color_palette("husl", len(classes))))
    colors = [palette[l] for l in labels]

    projections = {
        "PCA": PCA(n_components=2, random_state=cfg.project.seed).fit_transform(feats),
        "tSNE": TSNE(n_components=2, random_state=cfg.project.seed,
                     perplexity=30, init="pca").fit_transform(feats),
    }
    for name, emb in projections.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=25, alpha=0.8)
        ax.set_title(f"{name} projection of pretrained features")
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=palette[c], markersize=7, label=c)
                   for c in classes]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, ncol=1)
        save_fig(fig, Path(fig_dir) / f"eda_embedding_{name.lower()}.png")


def class_similarity(df: pd.DataFrame, cfg, fig_dir: str | Path) -> Optional[pd.DataFrame]:
    """Cosine-similarity matrix between mean class embeddings (confusion risk)."""
    feats, labels = _extract_embeddings(df, cfg)
    if feats is None:
        return None
    from sklearn.metrics.pairwise import cosine_similarity
    classes = sorted(set(labels))
    centroids = np.stack([feats[labels == c].mean(0) for c in classes])
    sim = cosine_similarity(centroids)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(sim, xticklabels=classes, yticklabels=classes,
                cmap="magma", ax=ax)
    ax.set_title("Inter-class similarity (cosine of mean embeddings)")
    save_fig(fig, Path(fig_dir) / "eda_class_similarity.png")
    return pd.DataFrame(sim, index=classes, columns=classes)


def run_eda(cfg, clean: pd.DataFrame, desc: pd.DataFrame) -> None:
    """Run the full EDA suite and save all figures to cfg.paths.figures."""
    fig_dir = cfg.paths.figures
    # plot_class_distribution(clean, fig_dir)
    plot_sample_grid(clean, fig_dir)
    # plot_rgb_distribution(clean, fig_dir)
    # plot_resolution(desc, fig_dir)
    plot_descriptor_correlation(desc, fig_dir)
    plot_embeddings(clean, cfg, fig_dir)
    class_similarity(clean, cfg, fig_dir)
