# Aerial Scene Classification — UC Merced Land-Use (21 Classes)

End-to-end, research-grade deep-learning pipeline that classifies 256×256 RGB
aerial scenes into 21 land-use categories. The project compares a **custom
residual-attention CNN**, a **fine-tuned DenseNet-121 backbone**, and a
**ViT-B/16 with LoRA adapters**, then deploys the winner behind a Streamlit
dashboard.

> Dataset: UC Merced Land-Use — 21 classes × 100 images = **2,100** images.

---

## Results

| Model | Params (M) | Test Accuracy | Test Macro-F1 | Train Time (min) |
|---|---|---|---|---|
| **ViT-B/16 + LoRA** | 86.06 | **99.05%** | **0.9905** | 3.6 |
| DenseNet-121 | 6.98 | 96.67% | 0.9665 | 2.2 |
| Custom CNN | 11.27 | 77.14% | 0.7584 | 2.9 |

Primary checkpoint selection metric: **validation macro-F1**.

---

## Highlights

- **Config-driven** (`configs/config.yaml`) — every tunable in one place, CLI-overridable.
- **Strict reproducibility** — global seed 42, deterministic cuDNN, frozen JSON splits.
- **Rigorous data hygiene** — corruption checks, perceptual-hash de-duplication,
  statistical + deep-feature outlier detection, blur/exposure/grayscale quality flags.
- **Leakage-free stratified 70/20/10 split**, verified programmatically.
- **Aerial-aware augmentation** — vertical flips & 90° rotations are label-preserving
  for overhead imagery (unlike natural-photo datasets).
- **Primary metric = macro-F1** for checkpoint selection (robust, class-balanced).
- **Grad-CAM** interpretability (implemented from scratch with hooks — no extra deps).
- **Optuna** hyper-parameter search (TPE sampler + Median pruner) with saved history
  & importance plots.
- **Auto-generated report** in Markdown + HTML (PDF if WeasyPrint is installed).
- **Streamlit** dashboard for single & batch inference, Grad-CAM visualization,
  and per-model performance reports.

---

## Design Decisions

1. **Custom CNN — residual + SE-attention architecture.** A from-scratch CNN with
   a 3×3 conv stem followed by four `ResidualBlock` stages (widths 64→128→256→512),
   each block containing two Conv-BN-ReLU layers plus a Squeeze-and-Excitation
   channel-attention module. Global average pooling and a dropout head follow.
   Trained end-to-end on the UC Merced dataset (~11.3 M parameters, no pretrained
   weights), it serves as the baseline to quantify how much transfer learning helps.
3. **Pretrained backbone — DenseNet-121.** Dense skip connections transfer well to
   fine-grained overhead imagery and the compact architecture (~7 M params) trains
   quickly. The backbone is frozen for the first 2 epochs to warm up the head, then
   fully fine-tuned.
4. **ViT-B/16 + LoRA.** Parameter-efficient fine-tuning via PEFT LoRA adapters on
   `qkv` and `proj` attention modules. Only ~0.55% of parameters are trainable,
   yet the model achieves the best accuracy, demonstrating that large frozen
   pretrained ViTs transfer extremely well even to small aerial datasets.
5. **Grad-CAM** is the interpretability method (`src/evaluate.py`), implemented
   from scratch with forward/backward hooks — no extra dependency needed.
6. **Two imbalance techniques suited to this *balanced* dataset:**
   **label smoothing** (default — regularises confident logits, helps confusable
   pairs like dense/medium residential) and **class-weighted cross-entropy**
   (inverse-frequency; a safety net if any images are quarantined).
   Switch via `imbalance.technique` in the config.

---

## Project Structure

```
aerial_scene_classification/
├── configs/
│   └── config.yaml              # all hyper-parameters & paths
├── data/
│   └── Images/                  # put dataset here: <class>/<image>.tif
├── dashboard/
│   └── app.py                   # Streamlit inference & report dashboard
├── notebooks/
│   └── aerial_scene_classification.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── data_validation.py   # corruption checks, quality flags, deduplication
│   │   ├── eda.py               # EDA plots (class distribution, PCA, t-SNE, UMAP)
│   │   ├── splitting.py         # stratified 70/20/10 split, frozen to JSON
│   │   ├── augmentation.py      # Albumentations train/val transforms
│   │   └── dataset.py           # AerialDataset, build_loaders, class weights
│   ├── model.py                 # CustomCNN, DenseNet-121, ViT+LoRA, utils
│   ├── train.py                 # Trainer, losses, Optuna, run_training
│   ├── evaluate.py              # metrics, plots, GradCAM, report generation
│   └── utils.py                 # DotDict config, seeding, device, save_fig
├── bestmodels/                  # *_best.pt checkpoints (auto-created)
├── results_figures/             # all figures, CSVs, splits.json (auto-created)
├── pyproject.toml
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch matching your CUDA version
#    See https://pytorch.org/get-started for the correct command

# 3. Place the UC Merced dataset so this path exists:
#      data/Images/<class_name>/<image>.tif
#    (edit paths.data_root in configs/config.yaml if your path differs)
```

---

## Usage

The notebook in `notebooks/` walks through every step (EDA → validation →
splitting → training → evaluation → report) with explanations and inline figures,
calling the same `src/` modules so the notebook and the Python scripts stay in sync.

### Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard supports single-image and batch inference, Grad-CAM overlays,
model comparison tables, and EDA figures for all trained checkpoints.

### Config overrides

Any field in `configs/config.yaml` can be overridden from the CLI using dotted
`key=value` syntax passed to `load_config()`:

```python
from src.utils import load_config
cfg = load_config("configs/config.yaml", overrides=["training.epochs=50", "training.lr=1e-4"])
```

---

## Outputs

After a training run you will find:

| Path | Contents |
|---|---|
| `results_figures/` | EDA plots, training curves, confusion matrices (raw + normalised), Grad-CAM grids, per-class F1 bars, confidence histograms, Optuna diagnostics |
| `results_figures/model_comparison.csv` | Side-by-side accuracy / macro-F1 / params / GFLOPs table |
| `results_figures/quality_report.csv` | Per-image quality flags from data validation |
| `results_figures/split_distribution.csv` | Class counts per split |
| `results_figures/splits.json` | Frozen train / val / test split (reused across all runs) |
| `results_figures/config_used.yaml` | Exact config snapshot for the run |
| `bestmodels/` | `<model_name>_best.pt` — checkpoint with best val macro-F1 |

---

## Reproducibility

- **Seed = 42** everywhere — Python `random`, NumPy, PyTorch, CUDA, DataLoader workers.
- **cuDNN deterministic** mode enabled; `CUBLAS_WORKSPACE_CONFIG=:4096:8` set.
- **Frozen split** — `results_figures/splits.json` is written once and reused by every
  model and every Optuna trial, guaranteeing fair comparison and no leakage.
- **Config snapshot** — `results_figures/config_used.yaml` captures the exact
  hyper-parameters for every run.

---

## Optional Dependencies

| Package | Purpose | Fallback |
|---|---|---|
| `umap-learn` | UMAP embedding plots in EDA | skipped gracefully |
| `fvcore` / `thop` | FLOPs counting (`gflops` column) | column left empty |
| `weasyprint` | PDF report export | only MD + HTML produced |

> **GPU note:** Data preprocessing runs on CPU. Training expects a CUDA GPU
> for reasonable speed; the pipeline falls back to CPU automatically.
