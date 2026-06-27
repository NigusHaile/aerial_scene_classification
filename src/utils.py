"""Utilities — configuration, figure saving, and reproducibility."""
from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# ============================================================================
# Configuration
# ============================================================================

class DotDict(dict):
    """A dict whose keys are also reachable as attributes, recursively."""

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, DotDict):
                return {k: unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unwrap(v) for v in value]
            return value
        return unwrap(self)


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _apply_override(cfg: DotDict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    node: Any = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = _coerce(value)


def load_config(path: str | Path = "configs/config.yaml",
                overrides: List[str] | None = None) -> DotDict:
    """Load YAML config and apply ``key=value`` dotted overrides."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = DotDict(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        _apply_override(cfg, key.strip(), value.strip())
    return cfg


def save_config(cfg: DotDict, path: str | Path) -> None:
    """Persist the (possibly overridden) config next to the run outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)


def ensure_dirs(cfg: DotDict) -> None:
    """Create every output directory declared in the config."""
    for key in ("outputs", "checkpoints"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Figure saving
# ============================================================================

def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure at consistent DPI and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch; optionally force deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — deterministically seeds each worker."""
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collect_env_metadata(seed: int) -> Dict[str, Any]:
    """Capture library versions and hardware info for the experiment log."""
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    meta["torch"] = torch.__version__
    meta["cuda_available"] = torch.cuda.is_available()
    meta["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
    for pkg in ("torchvision", "timm", "albumentations", "peft", "optuna", "sklearn", "cv2"):
        try:
            meta[pkg] = getattr(__import__(pkg), "__version__", "unknown")
        except ImportError:
            meta[pkg] = None
    return meta


def run_setup(cfg) -> Dict[str, Any]:
    """Seed RNG, create output dirs, snapshot config."""
    ensure_dirs(cfg)
    set_seed(cfg.project.seed)
    meta = collect_env_metadata(cfg.project.seed)
    save_config(cfg, Path(cfg.paths.outputs) / "config_used.yaml")
    return meta


def get_device(cfg) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on torch availability."""
    if cfg.project.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.project.device
