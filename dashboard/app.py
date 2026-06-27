"""Aerial Scene Classifier — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerial Scene Classifier",
    page_icon="🛰️",
    layout="wide",
)

CKPT_DIR = Path(__file__).resolve().parents[1] / "bestmodels"
SKIP_NAMES = {"probe"}   # internal checkpoints, not for display


# ── helpers ───────────────────────────────────────────────────────────────────

def _available_models():
    if not CKPT_DIR.exists():
        return {}
    out = {}
    for pt in sorted(CKPT_DIR.glob("*_best.pt")):
        run = pt.stem.replace("_best", "")
        if run not in SKIP_NAMES:
            label = run.replace("_", " ").title()
            out[label] = pt
    return out


@st.cache_resource
def load_model(checkpoint_path: str):
    import torch
    import timm
    from src.model import CustomCNN

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    class_names = ckpt["class_names"]
    num_classes = cfg["project"]["num_classes"]
    run_name = Path(checkpoint_path).stem.replace("_best", "")

    if run_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes)
    elif run_name == "vit_lora":
        model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                  num_classes=num_classes)
    else:
        model = timm.create_model(run_name, pretrained=False,
                                  num_classes=num_classes)

    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError:
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, class_names, cfg, run_name


def preprocess(image_rgb: np.ndarray, cfg) -> "torch.Tensor":
    import torch
    size = cfg["data"]["image_size"]
    mean = np.array(cfg["data"]["mean"], np.float32)
    std  = np.array(cfg["data"]["std"],  np.float32)
    x = cv2.resize(image_rgb, (size, size)).astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def get_target_layer(model, run_name: str):
    """Return the last feature Conv2d suitable for Grad-CAM."""
    import torch.nn as nn
    if run_name == "custom_cnn":
        return model.features[-1]          # last ResidualBlock
    if run_name == "vit_lora":
        return None                        # attention-based; skip
    # timm CNN: walk modules, take last Conv2d not in head/classifier
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and "head" not in name and "classifier" not in name:
            last = mod
    return last


def run_gradcam(model, target_layer, tensor: "torch.Tensor",
                class_idx: int) -> np.ndarray | None:
    """Return a [0,1] CAM heat-map or None if unavailable."""
    import torch
    if target_layer is None:
        return None
    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["v"] = out.detach()

    def bwd_hook(m, gin, gout):
        gradients["v"] = gout[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
    try:
        tensor = tensor.requires_grad_(True)
        logits = model(tensor)
        model.zero_grad()
        logits[0, class_idx].backward()
        weights = gradients["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["v"]).sum(1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
    except Exception:
        cam = None
    finally:
        fh.remove(); bh.remove()
    return cam


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_r = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * image_rgb).astype(np.uint8)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛰️ Aerial Scene Classifier")
    st.caption("UC Merced Land-Use — 21 classes")
    st.divider()

    available = _available_models()
    if not available:
        st.error(f"No `*_best.pt` files found in `{CKPT_DIR}`. Train a model first.")
        st.stop()

    model_label = st.selectbox("Model", list(available.keys()))
    ckpt_path   = str(available[model_label])
    do_gradcam  = st.toggle("Grad-CAM analysis", value=True)
    top_k       = st.slider("Top-K classes to show", 3, 21, 5)

    st.divider()
    st.markdown(
        "**How to use:** upload any aerial image (TIF / JPG / PNG). "
        "Switch models from the dropdown above."
    )


# ── load model ────────────────────────────────────────────────────────────────

with st.spinner(f"Loading {model_label}…"):
    model, class_names, cfg, run_name = load_model(ckpt_path)

st.markdown(f"## Results — *{model_label}*")


# ── image upload ──────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload an aerial image",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload an aerial image to classify it.")
    st.stop()


# ── inference ─────────────────────────────────────────────────────────────────

import torch
from PIL import Image as PILImage

pil_img = PILImage.open(uploaded).convert("RGB")
arr     = np.array(pil_img)
tensor  = preprocess(arr, cfg)

with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0].numpy()

top_idx   = int(probs.argmax())
top_label = class_names[top_idx]
top_conf  = float(probs[top_idx])

# ── layout: image | grad-cam | metrics ───────────────────────────────────────

col_img, col_cam, col_metrics = st.columns([1, 1, 1.4], gap="large")

size = cfg["data"]["image_size"]
display_arr = cv2.resize(arr, (size, size))

with col_img:
    st.markdown("**Input image**")
    st.image(display_arr, use_container_width=True)

with col_cam:
    st.markdown("**Grad-CAM** — where the model looks")
    if do_gradcam:
        target_layer = get_target_layer(model, run_name)
        if target_layer is None:
            st.info("Grad-CAM is not supported for ViT architectures.")
        else:
            cam = run_gradcam(model, target_layer, tensor, top_idx)
            if cam is not None:
                st.image(overlay_cam(display_arr, cam), use_container_width=True)
            else:
                st.warning("Grad-CAM failed for this image.")
    else:
        st.image(display_arr, use_container_width=True, caption="(Grad-CAM off)")

with col_metrics:
    # prediction badge
    colour = "#1a7f4b" if top_conf > 0.7 else "#b87800" if top_conf > 0.4 else "#c0392b"
    st.markdown(
        f"""<div style="background:{colour};border-radius:10px;padding:14px 18px;
        color:white;font-size:1.1rem;font-weight:600;margin-bottom:12px;">
        Prediction: {top_label}<br>
        <span style="font-size:2rem;">{top_conf:.1%}</span> confidence
        </div>""",
        unsafe_allow_html=True,
    )

    # top-k horizontal bars
    order = np.argsort(probs)[::-1][:top_k]
    st.markdown(f"**Top-{top_k} probabilities**")
    for rank, i in enumerate(order):
        bar_col, label_col = st.columns([3, 1])
        with bar_col:
            st.progress(float(probs[i]))
        with label_col:
            st.caption(f"{class_names[i]}")


# ── full probability table ────────────────────────────────────────────────────

with st.expander("All 21 class probabilities"):
    import pandas as pd
    rows = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    df   = pd.DataFrame(rows, columns=["Class", "Probability"])
    df["Probability"] = df["Probability"].map("{:.2%}".format)
    st.dataframe(df, use_container_width=True, hide_index=True)
