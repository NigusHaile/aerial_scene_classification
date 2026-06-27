"""Evaluation — metrics, visualisations, interpretability, and report generation.

Combines:
  - Metrics (compute_metrics, per_class_f1)
  - Plots (confusion matrix, training curves, per-class F1, most-confused pairs)
  - Interpretability (GradCAM, confidence errors, feature embeddings)
  - Report generation (comparison table, pick_winner, generate_report)
"""
from __future__ import annotations

import sys
import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, top_k_accuracy_score,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import save_fig



# Metrics

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    num_classes: Optional[int] = None) -> Dict[str, float]:
    """Compute the full metric suite (primary: macro-F1)."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        labels = list(range(num_classes or y_prob.shape[1]))
        try:
            metrics["top3_accuracy"] = float(
                top_k_accuracy_score(y_true, y_prob, k=3, labels=labels))
        except ValueError:
            metrics["top3_accuracy"] = float("nan")
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr",
                              average="macro", labels=labels))
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")
    return metrics


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: List[str]) -> Dict[str, float]:
    """Return per-class F1 scores keyed by class name."""
    scores = f1_score(y_true, y_pred, average=None,
                      labels=list(range(len(class_names))), zero_division=0)
    return {name: float(s) for name, s in zip(class_names, scores)}


# ============================================================================
# Plots
# ============================================================================

def plot_training_curves(history, fig_dir: str | Path, run_name: str) -> None:
    """Loss + macro-F1/accuracy curves over epochs."""
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history.train_loss, label="train loss")
    axes[0].plot(epochs, history.val_loss, label="val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history.val_acc, label="val accuracy")
    axes[1].plot(epochs, history.val_macro_f1, label="val macro-F1")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle(f"Training history — {run_name}")
    save_fig(fig, Path(fig_dir) / f"training_curves_{run_name}.png")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                   class_names: List[str], fig_dir: str | Path,
                   run_name: str) -> None:
    """Raw and row-normalised confusion matrices."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion matrix — {run_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save_fig(fig, Path(fig_dir) / f"confusion_{run_name}.png")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Normalised confusion matrix — {run_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    save_fig(fig, Path(fig_dir) / f"confusion_norm_{run_name}.png")


def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: List[str], fig_dir: str | Path,
                      run_name: str) -> pd.DataFrame:
    """Bar chart of per-class F1; returns sorted DataFrame (hardest first)."""
    scores = per_class_f1(y_true, y_pred, class_names)
    df = (pd.DataFrame({"class": list(scores), "f1": list(scores.values())})
          .sort_values("f1"))
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="class", y="f1", hue="class",
                palette="rocket", legend=False, ax=ax)
    ax.set_title(f"Per-class F1 — {run_name}")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 1)
    save_fig(fig, Path(fig_dir) / f"per_class_f1_{run_name}.png")
    return df


def hardest_classes(per_class_df: pd.DataFrame, k: int = 5) -> List[str]:
    """Return the k classes with the lowest F1."""
    return per_class_df.nsmallest(k, "f1")["class"].tolist()


def most_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                        class_names: List[str], k: int = 10
                        ) -> List[Tuple[str, str, int]]:
    """Top-k off-diagonal confusion pairs (true → predicted)."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    return sorted(pairs, key=lambda t: t[2], reverse=True)[:k]


# ============================================================================
# Interpretability (GradCAM + error analysis)
# ============================================================================

class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) via forward/backward hooks."""

    def __init__(self, model, target_layer) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, input_tensor, class_idx: Optional[int] = None) -> np.ndarray:
        import torch
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] CAM heatmap onto an RGB uint8 image."""
    cam_resized = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * image_rgb).astype(np.uint8)


def gradcam_grid(model, target_layer, samples, cfg, device,
                 class_names: List[str], fig_dir: str | Path,
                 run_name: str, eval_tf) -> None:
    """Save a montage of Grad-CAM overlays for a list of (path, true_label)."""
    cam_engine = GradCAM(model, target_layer)
    mean = np.array(cfg.data.mean, np.float32)
    std = np.array(cfg.data.std, np.float32)
    n = len(samples)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (path, true_label) in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        x = eval_tf(image=img)["image"].unsqueeze(0).to(device)
        cam = cam_engine(x)
        pred = int(model(x).argmax(1).item())
        disp = cv2.resize(img, (cfg.data.image_size, cfg.data.image_size))
        axes[i].imshow(overlay_cam(disp, cam))
        ok = "✓" if pred == true_label else "✗"
        axes[i].set_title(f"{ok} T:{class_names[true_label]}\nP:{class_names[pred]}",
                          fontsize=9)
    cam_engine.remove()
    fig.suptitle(f"Grad-CAM — {run_name}", fontsize=15)
    save_fig(fig, Path(fig_dir) / f"gradcam_{run_name}.png")


def confidence_errors(eval_out: dict, splits_test, class_names: List[str],
                      top_k: int = 10) -> pd.DataFrame:
    """Mine the most confident wrong predictions."""
    targets = eval_out["targets"]
    preds = eval_out["preds"]
    probs = eval_out["probs"]
    conf = probs.max(1)
    wrong = preds != targets
    rows = []
    for i in np.where(wrong)[0]:
        rows.append({
            "path": splits_test[i][0],
            "true": class_names[targets[i]],
            "pred": class_names[preds[i]],
            "confidence": float(conf[i]),
        })
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def plot_confidence_histogram(eval_out: dict, fig_dir: str | Path,
                              run_name: str) -> None:
    """Histogram of prediction confidence split by correct/incorrect."""
    correct = eval_out["preds"] == eval_out["targets"]
    conf = eval_out["probs"].max(1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(conf[correct], bins=20, alpha=0.7, label="correct", color="seagreen")
    ax.hist(conf[~correct], bins=20, alpha=0.7, label="incorrect", color="crimson")
    ax.set_title(f"Prediction confidence — {run_name}")
    ax.set_xlabel("max softmax probability"); ax.set_ylabel("count"); ax.legend()
    save_fig(fig, Path(fig_dir) / f"confidence_hist_{run_name}.png")


def plot_feature_embedding(features: np.ndarray, labels: np.ndarray,
                           class_names: List[str], fig_dir: str | Path,
                           run_name: str, seed: int = 42) -> None:
    """t-SNE of penultimate features on the test set."""
    from sklearn.manifold import TSNE
    projections = {"tSNE": TSNE(n_components=2, init="pca", perplexity=30,
                                random_state=seed).fit_transform(features)}
    palette = sns.color_palette("husl", len(class_names))
    for name, emb in projections.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(len(class_names)):
            m = labels == c
            ax.scatter(emb[m, 0], emb[m, 1], s=20, color=palette[c],
                       label=class_names[c], alpha=0.8)
        ax.set_title(f"{name} of test features — {run_name}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        save_fig(fig, Path(fig_dir) / f"test_embedding_{name.lower()}_{run_name}.png")


# ============================================================================
# Report generation
# ============================================================================

def comparison_table(records: List[Dict]) -> pd.DataFrame:
    """Assemble the model comparison table sorted by macro-F1."""
    df = pd.DataFrame(records)
    cols = ["model", "params_m", "accuracy", "macro_f1", "train_time_min", "gflops"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def pick_winner(df: pd.DataFrame) -> str:
    """Winner = highest test macro-F1."""
    return df.iloc[0]["model"]


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;margin:12px 0;">'


def generate_report(cfg, comparison: pd.DataFrame, winner: str,
                    test_metrics: Dict[str, Dict], extras: Dict,
                    figures: List[str], reports_dir: str | Path) -> Dict[str, Path]:
    """Write Markdown + HTML (+ optional PDF) reports; return the paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(cfg.paths.figures)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = []
    md.append(f"# Aerial Scene Classification — Final Report\n")
    md.append(f"*Generated {ts}*\n")
    md.append("## 1. Executive Summary\n")
    md.append(
        f"We classify the 21-class UC Merced Land-Use dataset (2,100 aerial images). "
        f"Three models were compared: a custom residual-attention CNN, a fine-tuned "
        f"**{cfg.model.name}** backbone, and a ViT-B/16 with LoRA adapters. "
        f"The selected model is **{winner}**, chosen on held-out test macro-F1.\n")
    md.append("## 2. Dataset\n")
    md.append(extras.get("dataset_summary", "") + "\n")
    md.append("## 3. Methodology\n")
    md.append(
        "- Stratified 70/15/15 split, frozen to JSON (no leakage).\n"
        "- Albumentations augmentation tuned for orientation-invariant aerial imagery.\n"
        f"- Imbalance strategy: **{cfg.imbalance.technique}** "
        f"(label smoothing = {cfg.imbalance.label_smoothing}).\n"
        "- Primary metric for checkpoint selection: **validation macro-F1**.\n"
        "- AMP mixed precision, gradient clipping, cosine schedule, early stopping.\n")
    md.append(f"\n**Backbone selection rationale:** {extras.get('selection_rationale','')}\n")
    md.append("## 4. Results\n")
    md.append(comparison.to_markdown(index=False) + "\n")
    md.append("### Per-model test metrics\n")
    md.append(pd.DataFrame(test_metrics).T.to_markdown() + "\n")
    md.append("## 5. Error Analysis\n")
    hc = extras.get("hardest_classes", [])
    cp = extras.get("confused_pairs", [])
    md.append(f"- Hardest classes (lowest F1): {', '.join(hc)}\n")
    if cp:
        md.append("- Most confused pairs (true → predicted):\n")
        for a, b, n in cp:
            md.append(f"  - {a} → {b}: {n}\n")
    md.append("## 6. Conclusion\n")
    md.append(extras.get("conclusion",
              f"The {winner} model offers the best accuracy/efficiency trade-off.") + "\n")
    md.append("## 7. Future Work\n")
    md.append(
        "- Test-time augmentation and model ensembling.\n"
        "- Self-supervised pretraining on unlabeled aerial imagery.\n"
        "- Higher-resolution inputs and multi-scale features for fine classes.\n")

    md_text = "\n".join(md)
    md_path = reports_dir / "final_report.md"
    md_path.write_text(md_text, encoding="utf-8")

    figs_html = "".join(
        f"<h3>{Path(f).stem}</h3>{_img_b64(fig_dir / f)}" for f in figures)
    try:
        import markdown as md_lib
        body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + md_text + "</pre>"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Aerial Scene Classification Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;
margin:40px auto;padding:0 20px;color:#1a1a1a;line-height:1.6}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#f4f4f4}} h1,h2,h3{{color:#0b3d62}}
img{{border:1px solid #eee;border-radius:6px}}
</style></head><body>{body}<hr><h2>Figures</h2>{figs_html}</body></html>"""
    html_path = reports_dir / "final_report.html"
    html_path.write_text(html, encoding="utf-8")

    paths = {"markdown": md_path, "html": html_path}
    try:
        from weasyprint import HTML
        pdf_path = reports_dir / "final_report.pdf"
        HTML(string=html).write_pdf(str(pdf_path))
        paths["pdf"] = pdf_path
    except Exception:
        pass

    return paths
