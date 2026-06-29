"""Training — loss functions, Trainer engine, and Optuna hyperparameter tuning.

Combines:
  - Loss functions (FocalLoss, build_loss)
  - Trainer (AMP, early stopping, checkpointing, TensorBoard)
  - Optuna hyperparameter optimisation (run_optuna, suggest_common)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compute_metrics, comparison_table, pick_winner
from src.utils import save_fig, get_device, seed_worker
from src import evaluate as ev


# ==============
# Loss functions
# ==============

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) — kept for ablation."""

    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(cfg, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Return the criterion selected by ``cfg.imbalance.technique``."""
    technique = cfg.imbalance.technique
    smoothing = cfg.imbalance.label_smoothing
    weight = class_weights if (technique == "class_weighted"
                               and cfg.imbalance.use_class_weights) else None
    if getattr(cfg.training, "loss", "ce") == "focal":
        return FocalLoss(gamma=2.0, weight=weight, label_smoothing=smoothing)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)


# =======
# Trainer
# ========

def _make_grad_scaler(enabled: bool):
    try:
        from torch.amp import GradScaler
        return GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError):
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=enabled)


def _autocast(device: str, enabled: bool):
    try:
        from torch.amp import autocast
        return autocast(device_type="cuda" if device == "cuda" else "cpu",
                        enabled=enabled and device == "cuda")
    except (ImportError, TypeError):
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=enabled and device == "cuda")


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.training.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.training.lr, momentum=0.9,
                               weight_decay=cfg.training.weight_decay, nesterov=True)
    return torch.optim.AdamW(params, lr=cfg.training.lr,
                             weight_decay=cfg.training.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    name = cfg.training.scheduler.lower()
    epochs = cfg.training.epochs
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * steps_per_epoch), "step"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.training.lr,
            steps_per_epoch=steps_per_epoch, epochs=epochs), "step"
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3), "epoch"
    return None, "none"


class Trainer:
    """Reusable trainer; one instance per model."""

    def __init__(self, model: nn.Module, criterion: nn.Module, cfg,
                 device: str, run_name: str,
                 class_names: Optional[List[str]] = None) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.cfg = cfg
        self.device = device
        self.run_name = run_name
        self.class_names = class_names
        self.history = TrainHistory()
        self.best_metric = -np.inf
        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = _make_grad_scaler(enabled=cfg.training.amp and device == "cuda")

    def _run_epoch(self, loader: DataLoader, train: bool,
                   optimizer=None, scheduler=None, sched_mode="none") -> float:
        self.model.train(train)
        total_loss, n = 0.0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                with _autocast(self.device, self.cfg.training.amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                if train:
                    self.scaler.scale(loss).backward()
                    if self.cfg.training.grad_clip:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.cfg.training.grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    if scheduler is not None and sched_mode == "step":
                        scheduler.step()
                total_loss += loss.item() * images.size(0)
                n += images.size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Run inference over a loader; return loss + metrics + arrays."""
        self.model.eval()
        all_logits, all_targets, total_loss, n = [], [], 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with _autocast(self.device, self.cfg.training.amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_logits.append(logits.float().cpu())
            all_targets.append(targets.cpu())
        logits = torch.cat(all_logits).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = probs.argmax(1)
        metrics = compute_metrics(targets, preds, probs,
                                  num_classes=self.cfg.project.num_classes)
        metrics["loss"] = total_loss / max(n, 1)
        return {"metrics": metrics, "targets": targets,
                "preds": preds, "probs": probs}

    def fit(self, loaders: Dict[str, DataLoader], epochs: Optional[int] = None,
            freeze_backbone_epochs: int = 0,
            unfreeze_fn=None) -> TrainHistory:
        """Full training loop with checkpointing & early stopping."""
        epochs = epochs or self.cfg.training.epochs
        optimizer = build_optimizer(self.model, self.cfg)
        scheduler, sched_mode = build_scheduler(
            optimizer, self.cfg, steps_per_epoch=len(loaders["train"]))
        patience = self.cfg.training.early_stop_patience
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            if unfreeze_fn is not None and epoch == freeze_backbone_epochs + 1:
                unfreeze_fn(self.model)
                optimizer = build_optimizer(self.model, self.cfg)
                print(f"[{self.run_name}] Backbone unfrozen at epoch {epoch}")

            t0 = time.time()
            train_loss = self._run_epoch(loaders["train"], True, optimizer,
                                         scheduler, sched_mode)
            val = self.evaluate(loaders["val"])
            vm = val["metrics"]
            if scheduler is not None and sched_mode == "epoch":
                scheduler.step(vm[self.cfg.training.monitor.replace("val_", "")])

            lr_now = optimizer.param_groups[0]["lr"]
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(vm["loss"])
            self.history.val_acc.append(vm["accuracy"])
            self.history.val_macro_f1.append(vm["macro_f1"])
            self.history.lr.append(lr_now)

            print(f"[{self.run_name}] Epoch {epoch:03d}/{epochs} | "
                  f"train_loss {train_loss:.4f} | val_loss {vm['loss']:.4f} | "
                  f"val_acc {vm['accuracy']:.4f} | val_macroF1 {vm['macro_f1']:.4f} | "
                  f"lr {lr_now:.2e} | {time.time() - t0:.1f}s")

            monitor_val = vm["macro_f1"]
            if monitor_val > self.best_metric:
                self.best_metric = monitor_val
                bad_epochs = 0
                self.save_checkpoint(epoch, optimizer, best=True)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[{self.run_name}] Early stopping at epoch {epoch} "
                          f"(best macroF1 {self.best_metric:.4f})")
                    break

        return self.history

    def save_checkpoint(self, epoch: int, optimizer, best: bool = False) -> Path:
        path = self.ckpt_dir / (f"{self.run_name}_best.pt" if best
                                else f"{self.run_name}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.cfg.to_dict(),
            "class_names": self.class_names,
        }, path)
        return path

    def load_checkpoint(self, path: str | Path, optimizer=None) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.best_metric = ckpt.get("best_metric", -np.inf)
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt.get("epoch", 0)


# ============================
# Optuna hyperparameter tuning
# ============================

def run_optuna(objective_builder: Callable, cfg, fig_dir: str | Path,
               study_name: str = "aerial") -> Dict:
    """Execute an Optuna study and persist diagnostic plots.

    ``objective_builder(trial, cfg) -> float`` should return val macro-F1.
    """
    import optuna

    sampler = optuna.samplers.TPESampler(seed=cfg.project.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=pruner, study_name=study_name)
    study.optimize(
        lambda trial: objective_builder(trial, cfg),
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout_minutes * 60,
        show_progress_bar=True,
    )
    print(f"Best trial value (val macroF1): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    try:
        import optuna.visualization.matplotlib as ov
        fig = ov.plot_optimization_history(study).figure
        save_fig(fig, Path(fig_dir) / "optuna_history.png")
        fig = ov.plot_param_importances(study).figure
        save_fig(fig, Path(fig_dir) / "optuna_importance.png")
    except Exception as exc:
        print(f"Optuna plotting skipped: {exc}")

    return {"best_params": study.best_params, "best_value": study.best_value}


def suggest_common(trial, cfg) -> Dict:
    """Suggest shared hyperparameters for CNN objectives."""
    return {
        "lr": trial.suggest_float("lr", cfg.optuna.lr_min, cfg.optuna.lr_max, log=True),
        "weight_decay": trial.suggest_float("weight_decay", cfg.optuna.wd_min,
                                            cfg.optuna.wd_max, log=True),
        "drop_rate": trial.suggest_float("drop_rate", cfg.optuna.dropout_min,
                                         cfg.optuna.dropout_max),
        "batch_size": trial.suggest_categorical("batch_size", cfg.optuna.batch_choices),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "onecycle"]),
    }


# ======================
# Training orchestration
# ======================

def _quick_val_f1(model: nn.Module, splits: Dict, cfg, epochs: int) -> float:
    """Short training run used by backbone selection and Optuna probes."""
    from src.preprocessing.augmentation import train_transform, val_transform
    from src.preprocessing.dataset import build_loaders, compute_class_weights

    device = get_device(cfg)
    loaders = build_loaders(splits, cfg, train_transform(cfg), val_transform(cfg),
                            worker_init_fn=seed_worker)
    weights = compute_class_weights([l for _, l in splits["train"]],
                                    cfg.project.num_classes).to(device)
    trainer = Trainer(model, build_loss(cfg, weights), cfg, device, "probe",
                      cfg.project.class_names)
    trainer.fit(loaders, epochs=epochs)
    return trainer.best_metric


def train_model(name: str, model: nn.Module, splits: Dict, cfg,
                freeze_epochs: int = 0, unfreeze_fn=None) -> Dict:
    """Train one model end-to-end and evaluate on the test split."""
    from src.model import count_parameters, estimate_flops
    from src.preprocessing.augmentation import train_transform, val_transform
    from src.preprocessing.dataset import build_loaders, compute_class_weights

    device = get_device(cfg)
    use_sampler = cfg.imbalance.technique == "weighted_sampler"
    loaders = build_loaders(splits, cfg, train_transform(cfg), val_transform(cfg),
                            use_weighted_sampler=use_sampler,
                            worker_init_fn=seed_worker)
    weights = compute_class_weights([l for _, l in splits["train"]],
                                    cfg.project.num_classes).to(device)
    trainer = Trainer(model, build_loss(cfg, weights), cfg, device, name,
                      cfg.project.class_names)
    t0 = time.time()
    history = trainer.fit(loaders, freeze_backbone_epochs=freeze_epochs,
                          unfreeze_fn=unfreeze_fn)
    train_time_min = (time.time() - t0) / 60.0

    best_ckpt = Path(cfg.paths.checkpoints) / f"{name}_best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
    test_out = trainer.evaluate(loaders["test"])
    tm = test_out["metrics"]

    ev.plot_training_curves(history, cfg.paths.figures, name)
    ev.plot_confusion(test_out["targets"], test_out["preds"],
                      cfg.project.class_names, cfg.paths.figures, name)
    pc = ev.plot_per_class_f1(test_out["targets"], test_out["preds"],
                              cfg.project.class_names, cfg.paths.figures, name)
    ev.plot_confidence_histogram(test_out, cfg.paths.figures, name)

    params = count_parameters(model)
    gflops = estimate_flops(model, cfg.data.image_size, "cpu")
    print(f"{name} | test acc {tm['accuracy']:.4f} | test macroF1 {tm['macro_f1']:.4f} | {train_time_min:.1f} min")

    return {
        "name": name, "trainer": trainer, "history": history,
        "test_out": test_out, "metrics": tm,
        "hardest": ev.hardest_classes(pc),
        "confused": ev.most_confused_pairs(
            test_out["targets"], test_out["preds"], cfg.project.class_names),
        "record": {
            "model": name,
            "params_m": round(params["total"] / 1e6, 2),
            "accuracy": round(tm["accuracy"], 4),
            "macro_f1": round(tm["macro_f1"], 4),
            "train_time_min": round(train_time_min, 2),
            "gflops": round(gflops, 2) if gflops > 0 else None,
        },
    }


def _cfg_with_hp(cfg, hp: dict):
    """Return a deep copy of cfg with Optuna best-params applied.

    cfg is a DotDict — plain attribute assignment works directly.
    """
    from copy import deepcopy
    c = deepcopy(cfg)
    if "lr"           in hp: c.training.lr           = hp["lr"]
    if "weight_decay" in hp: c.training.weight_decay = hp["weight_decay"]
    if "batch_size"   in hp: c.training.batch_size   = hp["batch_size"]
    if "optimizer"    in hp: c.training.optimizer    = hp["optimizer"]
    if "scheduler"    in hp: c.training.scheduler    = hp["scheduler"]
    if "drop_rate"    in hp: c.model.drop_rate       = hp["drop_rate"]
    if "lora_r"       in hp:
        c.lora.r     = hp["lora_r"]
        c.lora.alpha = hp["lora_r"] * 2
    return c


def run_training(cfg, do_lora: bool = True,
                 model_hp: dict | None = None) -> Dict:
    """Train Custom CNN + DenseNet-121 + ViT+LoRA; compare; report.

    model_hp: optional dict mapping model key → Optuna best_params dict, e.g.
        {"custom_cnn": {...}, "densenet121": {...}, "vit_lora": {...}}
    Each model trains with its own tuned HPs; cfg is never mutated.
    """
    from src.model import (build_custom_cnn, build_pretrained, set_backbone_trainable,
                           build_vit_lora)
    from src.preprocessing.splitting import load_splits

    model_hp = model_hp or {}
    splits    = load_splits(cfg.paths.splits)
    cfg.project.class_names = sorted(splits["label_map"], key=splits["label_map"].get)
    results: List[Dict] = []

    #  Custom CNN 
    cnn_cfg = _cfg_with_hp(cfg, model_hp.get("custom_cnn", {}))
    print(f"[custom_cnn] lr={cnn_cfg.training.lr:.2e}  wd={cnn_cfg.training.weight_decay:.2e}"
          f"  drop={cnn_cfg.model.drop_rate:.2f}  bs={cnn_cfg.training.batch_size}"
          f"  opt={cnn_cfg.training.optimizer}  sched={cnn_cfg.training.scheduler}")
    results.append(train_model("custom_cnn", build_custom_cnn(cnn_cfg), splits, cnn_cfg))

    #  DenseNet-121 
    pre_cfg = _cfg_with_hp(cfg, model_hp.get("densenet121", {}))
    print(f"[densenet121] lr={pre_cfg.training.lr:.2e}  wd={pre_cfg.training.weight_decay:.2e}"
          f"  drop={pre_cfg.model.drop_rate:.2f}  bs={pre_cfg.training.batch_size}"
          f"  opt={pre_cfg.training.optimizer}  sched={pre_cfg.training.scheduler}")
    pre = build_pretrained("densenet121", pre_cfg.project.num_classes,
                           pre_cfg.model.pretrained, pre_cfg.model.drop_rate)
    set_backbone_trainable(pre, False)
    results.append(train_model(
        "densenet121", pre, splits, pre_cfg,
        freeze_epochs=pre_cfg.model.freeze_backbone_epochs,
        unfreeze_fn=lambda m: set_backbone_trainable(m, True),
    ))

    #  ViT-B/16 + LoRA 
    if do_lora:
        vit_cfg = _cfg_with_hp(cfg, model_hp.get("vit_lora", {}))
        print(f"[vit_lora]  lr={vit_cfg.training.lr:.2e}  wd={vit_cfg.training.weight_decay:.2e}"
              f"  lora_r={vit_cfg.lora.r}  lora_alpha={vit_cfg.lora.alpha}"
              f"  bs={vit_cfg.training.batch_size}  opt={vit_cfg.training.optimizer}")
        vit, lora_stats = build_vit_lora(vit_cfg)
        print(f"ViT+LoRA trainable: {lora_stats['trainable_pct']:.2f}% of params")
        results.append(train_model("vit_lora", vit, splits, vit_cfg))

    comp   = comparison_table([r["record"] for r in results])
    winner = pick_winner(comp)
    comp.to_csv(Path(cfg.paths.outputs) / "model_comparison.csv", index=False)

    print(f"Winner: {winner}")
    return {"results": results, "comparison": comp, "winner": winner}
