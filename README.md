# Aerial Scene Classification — UC Merced Land-Use (21 Classes)

End-to-end, research-grade deep-learning pipeline that classifies 256×256 RGB
aerial scenes into 21 land-use categories. The project compares a **custom
residual-attention CNN**, a **single best-in-class pretrained backbone**
(ConvNeXt-Tiny, selected automatically), and a **ViT-B/16 with LoRA adapters**,
then ships the winner behind a Streamlit app.

> Dataset: UC Merced Land-Use — 21 classes × 100 images = **2,100** images.

---

## Highlights

- **Config-driven** (`configs/config.yaml`) — every tunable in one place, CLI-overridable.
- **Strict reproducibility** — global seeding, deterministic cuDNN, frozen JSON splits.
- **Rigorous data hygiene** — corruption checks, perceptual-hash de-duplication,
  statistical + deep-feature outlier detection, blur/exposure/grayscale quality flags.
- **Leakage-free stratified 70/15/15 split**, verified programmatically.
- **Aerial-aware augmentation** — vertical flips & 90° rotations are label-preserving
  for overhead imagery (unlike natural-photo datasets).
- **Primary metric = macro-F1** for checkpoint selection (robust, class-balanced).
- **Grad-CAM** interpretability, confusion analysis, confidence mining, t-SNE/UMAP.
- **Optuna** hyper-parameter search with saved history & importance plots.
- **Auto-generated report** in Markdown + HTML (PDF if WeasyPrint is installed).
- **Streamlit** deployment app.

## Design decisions (per your instructions)

1. **One pretrained model.** The default and recommended backbone is
   **ConvNeXt-Tiny** — strongest transfer on small fine-grained overhead datasets,
   more efficient than EfficientNet-B3 at 224px. `select_backbone()` still verifies
   this empirically against ResNet50 / DenseNet121 / EfficientNet-B3 via a short probe.
   Disable probing with `--no-select` to train ConvNeXt-Tiny directly.
2. **Grad-CAM** is the interpretability method (`src/evaluation/interpretability.py`),
   implemented from scratch with hooks so it needs no extra dependency.
3. **Two imbalance techniques suited to this *balanced* set:**
   **label smoothing** (default — regularises confident logits, helps confusable
   pairs like dense/medium residential) and **class-weighted cross-entropy**
   (inverse-frequency; a correctness safeguard if any images are quarantined).
   Switch via `imbalance.technique` in the config.

---

## Project structure

```
project/
├── configs/config.yaml          # all hyper-parameters & paths
├── data/                        # put the dataset here (or symlink)
├── notebooks/aerial_scene_classification.ipynb
├── src/
│   ├── preprocessing/           # validation, EDA, augmentation, dataset, splitting
│   ├── models/                  # custom_cnn, pretrained, vit_lora, model_utils
│   ├── training/                # trainer, losses, tuning (Optuna)
│   ├── evaluation/              # metrics, plots, interpretability, report
│   ├── utils/                   # config, reproducibility, logging
│   ├── pipeline.py              # data stages (torch-free)
│   └── model_pipeline.py        # train/eval/compare/report (needs torch)
├── outputs/                     # figures, checkpoints, reports, logs (auto-created)
├── app.py                       # Streamlit deployment app
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create an environment and install deps
pip install -r requirements.txt
# Install the torch build matching your CUDA from https://pytorch.org

# 2. Place the dataset so that this path exists:
#    data/UCMerced_LandUse/Images/<class>/<image>.tif
#    (edit paths.data_root in configs/config.yaml if different)
```

## Usage

```bash
# Data preparation + EDA + split only (no GPU required)
python main.py --stage data

# Full run: data prep, train all three models, compare, generate report
python main.py --stage all

# Train ConvNeXt-Tiny directly (skip the backbone probe), skip LoRA:
python main.py --stage all --no-select --no-lora

# Override any config field from the CLI
python main.py --stage all --set training.epochs=60 training.batch_size=64

# Launch the app on the winning checkpoint
streamlit run app.py -- --checkpoint outputs/checkpoints/convnext_tiny_best.pt

# TensorBoard
tensorboard --logdir outputs/logs/tb
```

The notebook in `notebooks/` walks through every step (1–18) with explanations
and inline figures, calling the same `src/` modules so notebook and CLI stay in sync.

---

## Outputs

After a run you will find:

- `outputs/figures/` — EDA plots, training curves, confusion matrices, Grad-CAM,
  per-class F1, embeddings, Optuna diagnostics.
- `outputs/checkpoints/` — `*_best.pt` per model (selected on val macro-F1).
- `outputs/reports/` — `final_report.md`, `final_report.html` (+ `.pdf` if available).
- `outputs/model_comparison.csv`, `quality_report.csv`, `split_distribution.csv`.
- `outputs/logs/` — run logs, environment metadata, TensorBoard events.

## Reproducibility

Seed = 42 everywhere; cuDNN deterministic; the train/val/test split is frozen to
`outputs/splits.json` and reused by every model and every Optuna trial. The exact
config used is snapshotted to `outputs/config_used.yaml`.

## Notes

- UMAP, FLOPs counters (fvcore/thop), and WeasyPrint (PDF) are optional; the
  pipeline degrades gracefully if they are absent.
- The data stages run on CPU; training expects a CUDA GPU for reasonable speed.
