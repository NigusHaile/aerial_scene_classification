"""Aerial Scene Classifier — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Aerial Scene Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants 
CKPT_DIR   = ROOT / "bestmodels"
FIG_DIR    = ROOT / "results_figures"
SKIP_NAMES = {"probe"}

MODEL_META = {
    "custom_cnn":  {"icon": "🔷", "full": "Custom CNN",      "desc": "Residual + SE-attention CNN",   "color": "#5fa88f"},
    "densenet121": {"icon": "🔶", "full": "DenseNet-121",     "desc": "Pretrained dense backbone",     "color": "#c79a4b"},
    "vit_lora":    {"icon": "✨", "full": "ViT-B/16 + LoRA", "desc": "Frozen ViT + PEFT adapters",   "color": "#4f96a0"},
}

CLASS_EMOJIS = {
    "agricultural": "🌾", "airplane": "✈️",  "baseballdiamond": "⚾",
    "beach": "🏖️",        "buildings": "🏢", "chaparral": "🌿",
    "denseresidential": "🏘️", "forest": "🌲", "freeway": "🛣️",
    "golfcourse": "⛳",    "harbor": "⚓",    "intersection": "🔀",
    "mediumresidential": "🏡", "mobilehomepark": "🚐", "overpass": "🌉",
    "parkinglot": "🅿️",   "river": "🌊",    "runway": "🛫",
    "sparseresidential": "🏠", "storagetanks": "🛢️", "tenniscourt": "🎾",
}

ARCH_INFO = {
    "custom_cnn":  ("~11.3 M", "Conv-BN-ReLU stem → 4× ResidualBlock + SE-Attention → GAP → Dropout → Linear"),
    "densenet121": ("~28.6 M", "DenseNet-121 pretrained on ImageNet · all layers fine-tuned · fresh head"),
    "vit_lora":    ("~86 M / 0.55% trainable", "Frozen ViT-B/16 · LoRA on QKV+Proj · PEFT"),
}

MODES = [
    "🔍  Single Prediction",
    "📦  Batch Prediction",
    "🌡️  Grad-CAM",
    "📈  Model Report",
    "🗺️  Dataset",
]

#  CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ════════════════════════════════════════════════════════════
   DESIGN TOKENS
   bg base #161a22 · panel #1d2330 · border #2a3140
   signature accent — phosphor green #5fa88f / #4f96a0
   secondary — amber #c79a4b (warn) · coral #bf6760 (error/low conf)
   type — JetBrains Mono for data/labels, Inter for prose
   ════════════════════════════════════════════════════════════ */

/* ── reset & base ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background: #161a22 !important;
    background-image:
        radial-gradient(ellipse 1000px 560px at 12% -8%, #2a3a3422, transparent 60%),
        radial-gradient(ellipse 800px 620px at 100% 6%, #26303a33, transparent 60%) !important;
    color: #e6e1d6;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 19px;
}

/* faint scanline texture across the whole app */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: repeating-linear-gradient(
        0deg, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px,
        rgba(95,168,143,0.04) 3px, rgba(0,0,0,0) 4px
    );
    mix-blend-mode: screen;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1a212e;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #aab4c4 !important;
    font-size: 17px !important;
}
[data-testid="stSidebar"] .stSelectbox label { color: #4a5568 !important; }

/* ── selectbox: outer trigger box ── */
[data-testid="stSelectbox"] > div > div {
    background: #1d2330 !important;
    border: 1.5px solid #2a3140 !important;
    border-radius: 10px !important;
    min-height: 58px !important;
    padding: 0 !important;
    transition: border-color .15s;
}
[data-testid="stSelectbox"] > div > div:hover    { border-color: #4a7d6c !important; }
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #5fa88f !important;
    box-shadow: 0 0 0 2px #5fa88f22 !important;
}

/* nuclear rule: force ALL text nodes inside the selectbox to be visible,
   excluding only SVG shape elements */
[data-testid="stSelectbox"] *:not(svg):not(path):not(g):not(circle):not(rect) {
    color: #e6e1d6 !important;
    -webkit-text-fill-color: #e6e1d6 !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* inner value + input padding */
[data-baseweb="select"] > div:first-child {
    background: #1d2330 !important;
    min-height: 56px !important;
    padding: 10px 18px !important;
    display: flex !important;
    align-items: center !important;
}

/* dropdown list panel */
[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #1d2330 !important;
    border: 1px solid #2a3140 !important;
    border-radius: 10px !important;
}
[data-baseweb="option"] {
    background: #1d2330 !important;
    color: #9aa0ab !important;
    -webkit-text-fill-color: #9aa0ab !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    padding: 14px 18px !important;
    min-height: 50px !important;
    display: flex !important;
    align-items: center !important;
}
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
    background: #1c2b28 !important;
    color: #4f96a0 !important;
    -webkit-text-fill-color: #4f96a0 !important;
}

/* ── radio in body (model selector) ── */
[data-testid="stRadio"] label {
    font-size: 16px !important;
    color: #9aa0ab !important;
    font-weight: 500;
    padding: 6px 0;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { font-size: 16px !important; }

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #1a1f2a !important;
    border: 1.5px dashed #2f3848 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #5fa88f !important; }
[data-testid="stFileUploaderDropZone"] {
    min-height: 200px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 32px 20px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small { font-size: 17px !important; color: #9aa0ab !important; }

/* ── buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #447b6c, #5fa88f) !important;
    color: #16231c !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    letter-spacing: 0.3px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    transition: filter .15s, transform .15s;
}
[data-testid="stButton"] > button:hover { filter: brightness(1.12) !important; transform: translateY(-1px); }
[data-testid="stDownloadButton"] > button {
    background: #1d2330 !important;
    border: 1px solid #2a3140 !important;
    color: #4f96a0 !important;
    font-size: 17px !important;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 8px !important;
}

/* ── sliders / toggles ── */
[data-testid="stSlider"] p { font-size: 17px !important; color: #9aa0ab !important; }
[data-testid="stToggle"] p { font-size: 17px !important; color: #9aa0ab !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #5fa88f !important; }
[data-testid="stSlider"] > div > div > div > div { background: #447b6c !important; }

/* ── progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #447b6c, #4f96a0) !important;
    border-radius: 4px !important;
}

/* ── images ── */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #2a3140;
    box-shadow: 0 4px 24px rgba(95,168,143,0.16);
}

/* ── expanders ── */
[data-testid="stExpander"] {
    background: #1d2330 !important;
    border: 1px solid #2a3140 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { font-size: 18px !important; font-weight: 600; color: #4f96a0 !important; font-family: 'JetBrains Mono', monospace; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #1d2330 !important; color: #5fa88f !important; font-size: 17px !important; font-family: 'JetBrains Mono', monospace; }
[data-testid="stDataFrame"] td { font-size: 17px !important; color: #9aa0ab !important; }

/* ── captions ── */
[data-testid="stCaptionContainer"] p { font-size: 16px !important; color: #737a87 !important; font-family: 'JetBrains Mono', monospace; }

/* ── dividers ── */
hr { border-color: #2a3140 !important; }

/* ════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ════════════════════════════════════════════════════════════ */

/* ── mono utility ── */
.mono { font-family: 'JetBrains Mono', monospace; }

/* ── hero banner ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #1a2230 0%, #161b24 55%, #13171f 100%);
    border: 1px solid #2f3848;
    border-radius: 12px;
    padding: 26px 34px;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    gap: 22px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 500px 300px at 85% 30%, #5fa88f12, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; top: 0; left: -60%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, #5fa88f08, transparent);
    animation: sweep 7s ease-in-out infinite;
    pointer-events: none;
}
@keyframes sweep { 0%,100% { left: -60%; } 50% { left: 110%; } }

.hero-icon {
    position: relative;
    width: 60px; height: 60px;
    border: 1.5px solid #5fa88f55;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.7rem;
    background: #1c2b28;
    flex-shrink: 0;
}
.hero-icon::before, .hero-icon::after {
    content: ''; position: absolute; width: 9px; height: 9px;
    border: 1.5px solid #4f96a0;
}
.hero-icon::before { top: -1.5px; left: -1.5px; border-right: none; border-bottom: none; }
.hero-icon::after { bottom: -1.5px; right: -1.5px; border-left: none; border-top: none; }

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600; color: #4f96a0;
    text-transform: uppercase; letter-spacing: 2.2px; margin: 0 0 6px;
    display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: #4f96a0; box-shadow: 0 0 8px #4f96a0; animation: blink 2s ease-in-out infinite; display: inline-block; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.hero-title {
    font-size: 2rem; font-weight: 800; color: #e6e1d6;
    letter-spacing: -0.4px; margin: 0 0 4px;
}
.hero-sub { font-size: 0.92rem; color: #9aa0ab; margin: 0; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.2px; }
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
.badge {
    border-radius: 4px; padding: 7px 14px;
    font-size: 0.75rem; font-weight: 600; white-space: nowrap; letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}
.badge-purple { background: #1c2b28; border: 1px solid #4a7d6c; color: #4f96a0; }
.badge-cyan   { background: #2b2419; border: 1px solid #a8843f; color: #c79a4b; }

/* ── section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem; font-weight: 600; color: #8a8f9c;
    text-transform: uppercase; letter-spacing: 1.6px;
    margin: 0 0 16px;
}

/* ── panel card ── */
.panel {
    background: #1d2330;
    border: 1px solid #2a3140;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ── model selector card ── */
.mcard {
    background: #1d2330;
    border: 1px solid #2a3140;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: border-color .15s, transform .15s;
    position: relative;
}
.mcard:hover { transform: translateY(-1px); }
.mcard-active {
    border-color: #5fa88f;
    background: linear-gradient(160deg, #1c2b28, #1d2330);
    box-shadow: 0 0 0 1px #5fa88f33, 0 8px 28px -8px #5fa88f2a;
}
.mcard-active::before {
    content: 'ACTIVE'; position: absolute; top: -1px; right: 14px;
    background: #5fa88f; color: #16231c; font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 1px;
    padding: 2px 8px; border-radius: 0 0 4px 4px;
}
.mcard-label { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #8a8f9c; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-weight: 600; }
.mcard-val   { font-size: 2.1rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }

/* ── info card ── */
.icard {
    background: #1d2330;
    border: 1px solid #2a3140;
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 14px;
    font-size: 1.08rem;
    color: #9aa0ab;
    line-height: 2.1;
}

/* ── prediction hero card (viewfinder lock) ── */
.pred-hero {
    position: relative;
    border-radius: 12px;
    padding: 34px 32px;
    margin-bottom: 20px;
    text-align: center;
    background: radial-gradient(ellipse 600px 300px at 50% 0%, #1c2b28, #13171f 70%);
    border: 1px solid #2f3848;
    overflow: hidden;
}
.pred-hero .lock-corner { position: absolute; width: 20px; height: 20px; border: 2px solid #5fa88faa; }
.pred-hero .lock-tl { top: 14px; left: 14px; border-right: none; border-bottom: none; }
.pred-hero .lock-tr { top: 14px; right: 14px; border-left: none; border-bottom: none; }
.pred-hero .lock-bl { bottom: 14px; left: 14px; border-right: none; border-top: none; }
.pred-hero .lock-br { bottom: 14px; right: 14px; border-left: none; border-top: none; }
.pred-scene  { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: #8a8f9c; text-transform: uppercase; letter-spacing: 2.8px; margin-bottom: 14px; font-weight: 600; }
.pred-name   { font-size: 2.4rem; font-weight: 800; color: #e6e1d6; margin-bottom: 10px; letter-spacing: -0.3px; }
.pred-conf   { font-size: 4.4rem; font-weight: 800; line-height: 1; letter-spacing: -1.5px; font-family: 'JetBrains Mono', monospace; }
.pred-model  { font-size: 1.05rem; color: #8a8f9c; margin-top: 16px; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

/* ── probability bars ── */
.pb-row   { display: flex; align-items: center; gap: 14px; margin-bottom: 13px; }
.pb-label { min-width: 200px; font-size: 1.08rem; color: #9aa0ab; font-weight: 500; }
.pb-track { flex: 1; background: #222837; border-radius: 4px; height: 22px; overflow: hidden; border: 1px solid #2a3140; }
.pb-fill  { height: 100%; border-radius: 3px; transition: width .4s cubic-bezier(.16,1,.3,1); }
.pb-pct   { min-width: 64px; text-align: right; font-size: 1.08rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* ── stat chips ── */
.stat-row { display: flex; gap: 14px; margin-bottom: 20px; }
.stat-chip {
    flex: 1; background: #1d2330; border: 1px solid #2a3140;
    border-radius: 10px; padding: 18px 14px; text-align: center;
    transition: border-color .15s;
}
.stat-chip:hover { border-color: #4a7d6c; }
.stat-chip-label { font-family: 'JetBrains Mono', monospace; font-size: 0.86rem; color: #8a8f9c; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600; }
.stat-chip-val   { font-size: 1.95rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }

/* ── batch thumbnail ── */
.bthumb { background: #1d2330; border: 1px solid #2a3140; border-radius: 8px; padding: 12px; text-align: center; transition: border-color .15s; }
.bthumb:hover { border-color: #4a7d6c; }
.bthumb-fname { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #737a87; margin-top: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bthumb-pred  { font-size: 1.08rem; font-weight: 700; color: #e6e1d6; margin: 6px 0 3px; }
.bthumb-conf  { font-size: 1.08rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }

/* ── validation pills ── */
.v-ok   { background: #1c2b28; border-left: 3px solid #5fa88f; border-radius: 6px; padding: 13px 18px; color: #4f96a0; margin-bottom: 10px; font-size: 17px; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.v-warn { background: #2b2419; border-left: 3px solid #c79a4b; border-radius: 6px; padding: 13px 18px; color: #c79a4b; margin-bottom: 8px;  font-size: 17px; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.v-err  { background: #2b1f20; border-left: 3px solid #bf6760; border-radius: 6px; padding: 13px 18px; color: #bf6760; margin-bottom: 8px;  font-size: 17px; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

/* ── empty state ── */
.empty-state {
    text-align: center;
    padding: 64px 20px;
    border: 1px dashed #2f3848;
    border-radius: 14px;
    background: #1a1f2a;
}
.empty-icon  { font-size: 3.5rem; opacity: 0.5; }
.empty-text  { font-size: 1.15rem; font-weight: 500; color: #737a87; margin-top: 18px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px; }

/* ── sidebar nav item ── */
.snav {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 16px; border-radius: 8px;
    margin-bottom: 4px; cursor: default;
    font-size: 15.5px; font-weight: 500; color: #5b6679;
    border: 1px solid transparent;
    transition: all .15s;
}
.snav-active {
    background: #0d2a20;
    border-color: #2a5f4a;
    color: #5eead4 !important;
    font-weight: 600;
}
.snav-icon { font-size: 1.15rem; }

/* ── sidebar logo ── */
.slogo {
    text-align: center;
    padding: 26px 0 18px;
    border-bottom: 1px solid #1a212e;
    margin-bottom: 20px;
}
.slogo-icon  { font-size: 2.8rem; }
.slogo-title { font-size: 1.15rem; font-weight: 800; color: #e8ecf4; margin: 10px 0 4px; }
.slogo-sub   { font-size: 0.8rem; color: #3a4254; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px; }

/* ── pill ── */
.pill {
    display: inline-block;
    border-radius: 4px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.3px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


#  helpers 

def _available_models():
    out = {}
    for pt in sorted(CKPT_DIR.glob("*_best.pt")):
        run = pt.stem.replace("_best", "")
        if run in SKIP_NAMES:
            continue
        meta  = MODEL_META.get(run, {"icon": "🤖", "full": run.replace("_"," ").title(), "desc": "", "color": "#5fa88f"})
        label = f"{meta['icon']} {meta['full']}"
        out[label] = {"path": pt, "run_name": run, **meta}
    return out


def _ckpt_val_f1(pt: Path) -> float:
    try:
        ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
        return float(ckpt.get("best_metric", 0.0))
    except Exception:
        return 0.0


@st.cache_resource
def load_model(checkpoint_path: str):
    import timm
    from src.model import CustomCNN
    ckpt        = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict    = ckpt["config"]
    class_names = ckpt["class_names"]
    num_classes = cfg_dict["project"]["num_classes"]
    run_name    = Path(checkpoint_path).stem.replace("_best", "")

    if run_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif run_name == "vit_lora":
        from peft import LoraConfig, get_peft_model
        lora = cfg_dict.get("lora", {})
        base = timm.create_model(lora.get("base_model", "vit_base_patch16_224"),
                                 pretrained=False, num_classes=num_classes)
        for p in base.parameters():
            p.requires_grad = False
        model = get_peft_model(base, LoraConfig(
            r=lora.get("r", 8), lora_alpha=lora.get("alpha", 16),
            lora_dropout=lora.get("dropout", 0.1),
            target_modules=list(lora.get("target_modules", ["qkv", "proj"])),
            modules_to_save=["head"], bias="none",
        ))
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model = timm.create_model(run_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)

    model.eval()
    return model, class_names, cfg_dict, run_name


def validate_image(file_bytes: bytes):
    warns, errs = [], []
    if len(file_bytes) / 1e6 > 20:
        errs.append(f"File too large ({len(file_bytes)/1e6:.1f} MB > 20 MB).")
        return False, warns, errs, None
    arr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        errs.append("Could not decode image — file may be corrupted.")
        return False, warns, errs, None
    h, w = arr.shape[:2]
    if h < 32 or w < 32:
        errs.append(f"Image too small ({w}×{h} px). Min 32×32.")
        return False, warns, errs, None
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lum  = float(gray.mean())
    if blur < 40:
        warns.append(f"Image appears blurry (sharpness score = {blur:.0f}).")
    if lum < 25:
        warns.append(f"Very dark image (luminance = {lum:.0f}).")
    elif lum > 230:
        warns.append(f"Overexposed image (luminance = {lum:.0f}).")
    return True, warns, errs, cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb: np.ndarray, cfg_dict: dict):
    sz   = cfg_dict["data"]["image_size"]
    mean = np.array(cfg_dict["data"]["mean"], np.float32)
    std  = np.array(cfg_dict["data"]["std"],  np.float32)
    x    = cv2.resize(image_rgb, (sz, sz)).astype(np.float32) / 255.0
    x    = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def get_target_layer(model, run_name: str):
    import torch.nn as nn
    if run_name == "custom_cnn":
        return model.features[-1]
    if run_name == "vit_lora":
        return None
    last = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and "head" not in _ and "classifier" not in _:
            last = mod
    return last


def run_gradcam(model, layer, tensor, class_idx: int):
    if layer is None:
        return None
    acts, grads = {}, {}
    fh = layer.register_forward_hook(lambda m, i, o: acts.update({"v": o.detach()}))
    bh = layer.register_full_backward_hook(lambda m, gi, go: grads.update({"v": go[0].detach()}))
    try:
        t = tensor.requires_grad_(True)
        logits = model(t)
        model.zero_grad()
        logits[0, class_idx].backward()
        w   = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["v"]).sum(1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min(); cam /= cam.max() + 1e-8
    except Exception:
        cam = None
    finally:
        fh.remove(); bh.remove()
    return cam


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_r = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return (alpha * cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) + (1 - alpha) * img).astype(np.uint8)


def conf_color(c: float) -> str:
    return "#5fa88f" if c >= 0.80 else "#c79a4b" if c >= 0.50 else "#bf6760"


def prob_bars_html(probs, class_names, top_k, highlight_idx):
    order = np.argsort(probs)[::-1][:top_k]
    rows  = []
    for rank, i in enumerate(order):
        p     = float(probs[i])
        name  = class_names[i]
        emoji = CLASS_EMOJIS.get(name, "📍")
        if i == highlight_idx:
            color = "#5fa88f"
            grad  = "linear-gradient(90deg,#447b6c,#5fa88f)"
        elif rank == 1:
            color = "#4f96a0"; grad = "linear-gradient(90deg,#3f7c84,#4f96a0)"
        elif rank == 2:
            color = "#c79a4b"; grad = "#b08e4f"
        else:
            color = "#8a8f9c"; grad = "#2a3140"
        rows.append(f"""
        <div class="pb-row">
          <div class="pb-label">{emoji} {name.replace("_"," ").title()}</div>
          <div class="pb-track">
            <div class="pb-fill" style="width:{p*100:.1f}%;background:{grad};"></div>
          </div>
          <div class="pb-pct" style="color:{color};">{p*100:.1f}%</div>
        </div>""")
    return "".join(rows)


def fig_path(name: str):
    p = FIG_DIR / name
    return p if p.exists() else None


#  sidebar 
with st.sidebar:
    st.markdown("""
    <div class="slogo">
      <div class="slogo-icon">🛰️</div>
      <div class="slogo-title">Aerial Scene AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:1.4rem;font-family:JetBrains Mono,monospace;color:#3a4254;text-transform:uppercase;letter-spacing:2px;margin:0 0 10px 4px;font-weight:600;'>Choose Task</p>", unsafe_allow_html=True)
    mode = st.selectbox("mode", MODES, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.74rem;font-family:JetBrains Mono,monospace;color:#3a4254;text-transform:uppercase;letter-spacing:2px;margin:0 0 10px 4px;font-weight:600;'>Settings</p>", unsafe_allow_html=True)

    if "Single" in mode:
        top_k       = st.slider("Top-K predictions", 3, 21, 7)
        compare_all = st.toggle("Compare all models", value=False)
        gradcam_alpha = 0.45
    elif "Batch" in mode:
        top_k = 3; compare_all = False; gradcam_alpha = 0.45
    elif "Grad" in mode:
        gradcam_alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
        top_k = 5; compare_all = False
    else:
        top_k = 5; compare_all = False; gradcam_alpha = 0.45

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:16px;background:#0a0d14;border:1px solid #1a212e;border-radius:10px;">
      <div style="font-size:0.74rem;font-family:'JetBrains Mono',monospace;color:#3a4254;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:12px;">System Info</div>
      <div style="font-size:14.5px;font-family:'JetBrains Mono',monospace;color:#4a5568;line-height:2.1;">
        3 model architectures<br>
        2,100 training images<br>
        21 land-use categories<br>
        0.3 m/pixel resolution
      </div>
    </div>
    """, unsafe_allow_html=True)


#  hero 
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛰️</div>
  <div>
    <p class="hero-eyebrow"><span class="dot"></span></p>
    <p class="hero-title" style="margin:0 0 4px;">Aerial Scene Classifier</p>
    <p class="hero-sub"></p>
  </div>
  <div class="hero-badges">
    <span class="badge badge-purple">Custom CNN</span>
    <span class="badge badge-purple">DenseNet-121</span>
    <span class="badge badge-cyan">ViT-B/16 + LoRA</span>
  </div>
</div>
""", unsafe_allow_html=True)

available = _available_models()
if not available:
    st.error(f"No checkpoints found in `{CKPT_DIR}`. Train a model first.")
    st.stop()


# ══════════════════════════════════
# FULL-WIDTH MODES: Report & Dataset
# ═══════════════════════════════════

if "Report" in mode:
    model_options = {f"{m['icon']} {m['full']}": k for k, m in MODEL_META.items()
                     if (CKPT_DIR / f"{k}_best.pt").exists()}
    rep_label = st.selectbox("Select model to inspect", list(model_options.keys()),
                             label_visibility="visible")
    rep_run   = model_options[rep_label]
    rep_meta  = MODEL_META.get(rep_run, {})
    rep_pt    = CKPT_DIR / f"{rep_run}_best.pt"
    rep_f1    = _ckpt_val_f1(rep_pt)
    params, arch_desc = ARCH_INFO.get(rep_run, ("?", ""))

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        acc_color = rep_meta.get("color", "#5fa88f")
        st.markdown(f"""
        <div class="icard">
          <div class="mcard-label">Architecture</div>
          <div style="font-size:1.4rem;font-weight:800;color:#e6e1d6;margin-bottom:8px;">
            {rep_meta.get('icon','')} {rep_meta.get('full','')}
          </div>
          <div style="color:#9aa0ab;line-height:1.9;">{arch_desc}</div>
          <div style="margin-top:14px;font-size:1rem;">
            Parameters: <span style="color:{acc_color};font-weight:700;">{params}</span><br>
            Best val macro-F1: <span style="color:{acc_color};font-weight:800;font-size:1.2rem;">{rep_f1:.4f}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="icard">
          <div class="mcard-label">Training Setup</div>
          📦 UC Merced — 2,100 images · 21 classes<br>
          ✂️ Split: 70 / 20 / 10 % stratified<br>
          🔁 40 epochs · early stopping (patience 5)<br>
          ⚡ AMP mixed precision · grad clip 1.0<br>
          📐 224 × 224 px · ImageNet normalisation<br>
          📉 Label smoothing α = 0.1
        </div>""", unsafe_allow_html=True)

    for fname, title in [
        (f"training_curves_{rep_run}.png",   "Training Curves"),
    ]:
        p = fig_path(fname)
        if p:
            st.markdown(f'<p class="sec-label">{title}</p>', unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)

    r1, r2 = st.columns(2, gap="large")
    for col, (fname, title) in zip([r1, r2], [
        (f"confusion_{rep_run}.png",      "Confusion Matrix"),
        (f"confusion_norm_{rep_run}.png", "Normalised Confusion Matrix"),
    ]):
        with col:
            p = fig_path(fname)
            if p:
                st.markdown(f'<p class="sec-label">{title}</p>', unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    for fname, title in [
        (f"per_class_f1_{rep_run}.png",      "Per-Class F1 Score"),
        (f"confidence_hist_{rep_run}.png",   "Confidence Distribution"),
    ]:
        p = fig_path(fname)
        if p:
            st.markdown(f'<p class="sec-label">{title}</p>', unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
    st.stop()


if "Dataset" in mode:
    st.markdown("""
    <div class="icard" style="margin-bottom:24px;">
      <div class="mcard-label">About the dataset</div>
      📦 <strong>UC Merced Land-Use Dataset</strong> — 2,100 aerial images · 21 land-use categories<br>
      🖼️ <strong>100 images per class</strong> — balanced · 256 × 256 px GeoTIFF tiles<br>
      🛰️ <strong>0.3 m/pixel</strong> from USGS National Map Urban Area Imagery<br>
      ✅ <strong>Validated:</strong> 16 quality-flagged · 1 near-duplicate pair detected
    </div>
    """, unsafe_allow_html=True)
    for fname, title in [
        ("eda_sample_grid.png",            "Sample Image Grid"),
        ("eda_class_distribution_bar.png", "Class Distribution"),
        ("eda_rgb_distribution.png",       "RGB Channel Distributions"),
        ("eda_class_imbalance.png",        "Class Imbalance Analysis"),
        ("eda_embedding_tsne.png",         "t-SNE Feature Embedding"),
        ("eda_class_similarity.png",       "Inter-class Similarity"),
    ]:
        p = fig_path(fname)
        st.markdown(f'<p class="sec-label">{title}</p>', unsafe_allow_html=True)
        if p:
            st.image(str(p), use_container_width=True)
        else:
            st.markdown(f'<div class="v-warn">⚠️ {fname} not found — run EDA step to generate it.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
# 3-COLUMN LAYOUT  [Upload Image | Model Selector | Analysis]
# ════════════════════════════════════════════════════════════

col_upload, col_model, col_analysis = st.columns([0.27, 0.23, 0.50], gap="large")

# COLUMN 1: Upload 
with col_upload:
    st.markdown('<p class="sec-label">Upload Image</p>', unsafe_allow_html=True)

    if "Batch" in mode:
        batch_files = st.file_uploader(
            "Drop multiple images",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_up",
            label_visibility="collapsed",
        )
        uploaded = None
    else:
        uploaded = st.file_uploader(
            "Drop an aerial image",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            key="single_up",
            label_visibility="collapsed",
        )
        batch_files = []

    img_rgb = None

    if uploaded is not None:
        file_bytes = uploaded.read()
        ok, val_warns, val_errs, img_rgb = validate_image(file_bytes)

        for e in val_errs:
            st.markdown(f'<div class="v-err">❌ {e}</div>', unsafe_allow_html=True)
        for w in val_warns:
            st.markdown(f'<div class="v-warn">⚠️ {w}</div>', unsafe_allow_html=True)
        if not val_errs and not val_warns and img_rgb is not None:
            h0, w0 = img_rgb.shape[:2]
            st.markdown(f'<div class="v-ok">✅ {w0} × {h0} px · {len(file_bytes)/1e3:.0f} KB</div>',
                        unsafe_allow_html=True)
        if not ok:
            st.stop()

        st.markdown('<p class="sec-label" style="margin-top:16px;">Preview</p>', unsafe_allow_html=True)
        st.image(cv2.resize(img_rgb, (224, 224)), use_container_width=True)
        st.caption(uploaded.name)

    elif "Batch" in mode and batch_files:
        st.markdown(f'<div class="v-ok">✅ {len(batch_files)} file(s) selected</div>',
                    unsafe_allow_html=True)
        if len(batch_files) <= 4:
            for f in batch_files[:4]:
                raw = f.read()
                _, _, _, preview = validate_image(raw)
                if preview is not None:
                    st.image(cv2.resize(preview, (112, 112)), use_container_width=True)
                    st.caption(f.name)
                f.seek(0)


# COLUMN 2: Model Selector 
with col_model:
    st.markdown('<p class="sec-label">Select Model</p>', unsafe_allow_html=True)

    model_keys = list(available.keys())
    sel_label  = st.selectbox("model_select", model_keys, label_visibility="collapsed")
    sel        = available[sel_label]
    val_f1     = _ckpt_val_f1(sel["path"])
    mc         = sel.get("color", "#5fa88f")

    st.markdown(f"""
    <div class="mcard mcard-active" style="border-color:{mc}55;box-shadow:0 0 0 1px {mc}22;">
      <div style="font-size:1.8rem;margin-bottom:6px;">{sel['icon']}</div>
      <div style="font-size:1.1rem;font-weight:800;color:#e6e1d6;">{sel['full']}</div>
      <div style="font-size:0.95rem;color:#9aa0ab;margin-top:5px;">{sel['desc']}</div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid #2a3140;">
        <div style="font-size:0.78rem;color:#8a8f9c;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:6px;">Val macro-F1</div>
        <div style="font-size:2rem;font-weight:900;color:{mc};">{val_f1:.4f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    params_str, arch_str = ARCH_INFO.get(sel["run_name"], ("?", ""))
    st.markdown(f"""
    <div class="mcard" style="margin-top:4px;">
      <div style="font-size:0.78rem;color:#8a8f9c;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">Architecture</div>
      <div style="font-size:0.95rem;color:#9aa0ab;line-height:1.8;">{arch_str}</div>
      <div style="margin-top:10px;font-size:0.95rem;color:#9aa0ab;">
        <span style="color:{mc};font-weight:700;">{params_str}</span> parameters
      </div>
    </div>
    """, unsafe_allow_html=True)


# COLUMN 3: Analysis 
with col_analysis:
    with st.spinner(f"Loading {sel['full']}…"):
        model, class_names, cfg_dict, run_name = load_model(str(sel["path"]))
    sz = cfg_dict["data"]["image_size"]

    #  Single Prediction
    if "Single" in mode:
        st.markdown('<p class="sec-label">Prediction Analysis</p>', unsafe_allow_html=True)

        if img_rgb is None:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🔍</div>
              <div class="empty-text">Upload an aerial image on the left<br>to classify it</div>
            </div>""", unsafe_allow_html=True)
        else:
            tensor = preprocess(img_rgb, cfg_dict)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].numpy()

            top_idx   = int(probs.argmax())
            top_label = class_names[top_idx]
            top_conf  = float(probs[top_idx])
            cc        = conf_color(top_conf)
            emoji     = CLASS_EMOJIS.get(top_label, "📍")

            st.markdown(f"""
            <div class="pred-hero" style="border-color:{cc}33;">
              <div class="pred-scene">Predicted Scene</div>
              <div class="pred-name">{emoji} {top_label.replace("_"," ").title()}</div>
              <div class="pred-conf" style="color:{cc};">{top_conf:.1%}</div>
              <div class="pred-model">{sel['icon']} {sel['full']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<p class="sec-label">Top {top_k} Predictions</p>', unsafe_allow_html=True)
            st.markdown(prob_bars_html(probs, class_names, top_k, top_idx), unsafe_allow_html=True)

            with st.expander("📋 All 21 class probabilities"):
                rows = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
                df   = pd.DataFrame(rows, columns=["Class", "Probability"])
                df.insert(0, "Rank",  range(1, len(df) + 1))
                df.insert(1, "Emoji", df["Class"].map(lambda c: CLASS_EMOJIS.get(c, "📍")))
                df["Class"]       = df["Class"].str.replace("_", " ").str.title()
                df["Probability"] = df["Probability"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(df, use_container_width=True, hide_index=True)

            if compare_all:
                st.markdown('<p class="sec-label" style="margin-top:22px;">All Models Comparison</p>', unsafe_allow_html=True)
                cols = st.columns(len(available), gap="medium")
                pred_labels = []
                for gcol, (lbl, meta) in zip(cols, available.items()):
                    with gcol:
                        with st.spinner(meta["full"]):
                            m2, cn2, cfg2, _ = load_model(str(meta["path"]))
                        with torch.no_grad():
                            p2 = torch.softmax(m2(preprocess(img_rgb, cfg2)), dim=1)[0].numpy()
                        idx2 = int(p2.argmax()); lbl2 = cn2[idx2]; conf2 = float(p2[idx2])
                        c2   = conf_color(conf2); mc2  = meta.get("color", "#5fa88f")
                        pred_labels.append(lbl2)
                        st.markdown(f"""
                        <div class="mcard" style="text-align:center;border-color:{mc2}44;">
                          <div style="font-size:1.6rem;">{meta['icon']}</div>
                          <div style="font-weight:700;color:#e6e1d6;font-size:1rem;margin:6px 0 2px;">{meta['full']}</div>
                          <div style="font-size:1.3rem;">{CLASS_EMOJIS.get(lbl2,'📍')}</div>
                          <div style="font-weight:700;color:#e6e1d6;font-size:1rem;margin:4px 0;">{lbl2.replace('_',' ').title()}</div>
                          <div style="font-size:1.6rem;font-weight:900;color:{c2};">{conf2:.1%}</div>
                        </div>""", unsafe_allow_html=True)

                if len(set(pred_labels)) == 1:
                    st.markdown('<div class="v-ok" style="text-align:center;margin-top:12px;">✅ All models agree on the same class!</div>', unsafe_allow_html=True)
                else:
                    agree = max(pred_labels.count(l) for l in set(pred_labels))
                    st.markdown(f'<div class="v-warn" style="text-align:center;margin-top:12px;">⚠️ Models disagree — {agree}/{len(pred_labels)} agree on the majority class.</div>', unsafe_allow_html=True)

    # Batch Prediction 
    elif "Batch" in mode:
        st.markdown('<p class="sec-label">Batch Results</p>', unsafe_allow_html=True)

        if not batch_files:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">📦</div>
              <div class="empty-text">Upload multiple images on the left<br>to classify them all at once</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="v-ok" style="margin-bottom:16px;">
              📂 {len(batch_files)} image(s) queued · using {sel['icon']} {sel['full']}
            </div>""", unsafe_allow_html=True)

            if st.button("▶  Run Batch Prediction", type="primary", use_container_width=True):
                results_list = []
                prog = st.progress(0, text="Classifying images…")
                for i, f in enumerate(batch_files):
                    raw = f.read()
                    ok_b, _, _, img_b = validate_image(raw)
                    if not ok_b:
                        results_list.append({"filename": f.name, "predicted_class": "ERROR",
                                             "confidence": 0.0, "top2": "", "top3": "",
                                             "status": "❌", "_img": None})
                    else:
                        with torch.no_grad():
                            pb = torch.softmax(model(preprocess(img_b, cfg_dict)), dim=1)[0].numpy()
                        order = np.argsort(pb)[::-1]
                        results_list.append({
                            "filename": f.name,
                            "predicted_class": class_names[order[0]],
                            "confidence": float(pb[order[0]]),
                            "top2": class_names[order[1]],
                            "top3": class_names[order[2]],
                            "status": "✅",
                            "_img": cv2.resize(img_b, (112, 112)),
                        })
                    prog.progress((i + 1) / len(batch_files),
                                  text=f"Processing {i+1}/{len(batch_files)}: {f.name}")
                prog.empty()
                st.session_state["batch_results"] = results_list

            results_list = st.session_state.get("batch_results", [])
            if results_list:
                valid_res = [r for r in results_list if r["status"] == "✅"]
                avg_conf  = float(np.mean([r["confidence"] for r in valid_res])) if valid_res else 0.0
                cc_map    = Counter(r["predicted_class"] for r in valid_res)
                top_cls   = cc_map.most_common(1)[0][0] if cc_map else "—"
                ac        = conf_color(avg_conf)

                st.markdown(f"""
                <div class="stat-row">
                  <div class="stat-chip">
                    <div class="stat-chip-label">Total</div>
                    <div class="stat-chip-val" style="color:#4f96a0;">{len(results_list)}</div>
                  </div>
                  <div class="stat-chip">
                    <div class="stat-chip-label">Valid</div>
                    <div class="stat-chip-val" style="color:#5fa88f;">{len(valid_res)}</div>
                  </div>
                  <div class="stat-chip">
                    <div class="stat-chip-label">Avg Conf</div>
                    <div class="stat-chip-val" style="color:{ac};">{avg_conf:.1%}</div>
                  </div>
                  <div class="stat-chip">
                    <div class="stat-chip-label">Top Class</div>
                    <div class="stat-chip-val" style="font-size:1rem;color:#c79a4b;">{CLASS_EMOJIS.get(top_cls,'📍')} {top_cls.replace('_',' ').title()}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<p class="sec-label">Image Grid</p>', unsafe_allow_html=True)
                NCOLS = 4
                for row_i in range(0, len(results_list), NCOLS):
                    gcols = st.columns(NCOLS, gap="small")
                    for gc, res in zip(gcols, results_list[row_i:row_i + NCOLS]):
                        with gc:
                            if res["_img"] is not None:
                                st.image(res["_img"], use_container_width=True)
                            ci = conf_color(res["confidence"])
                            em = CLASS_EMOJIS.get(res["predicted_class"], "📍")
                            st.markdown(f"""
                            <div class="bthumb">
                              <div class="bthumb-fname" title="{res['filename']}">{res['filename']}</div>
                              <div class="bthumb-pred">{em} {res['predicted_class'].replace('_',' ').title()}</div>
                              <div class="bthumb-conf" style="color:{ci};">{res['confidence']:.1%}</div>
                            </div>""", unsafe_allow_html=True)

                if cc_map:
                    st.markdown('<p class="sec-label" style="margin-top:18px;">Class Distribution</p>', unsafe_allow_html=True)
                    df_d = pd.DataFrame(cc_map.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)
                    df_d["Class"] = df_d["Class"].str.replace("_", " ").str.title()
                    st.bar_chart(df_d.set_index("Class"), color="#5fa88f", height=260)

                st.markdown('<p class="sec-label" style="margin-top:18px;">Full Results Table</p>', unsafe_allow_html=True)
                df_out = pd.DataFrame([{
                    "File":       r["filename"],
                    "Prediction": r["predicted_class"].replace("_", " ").title(),
                    "Confidence": f"{r['confidence']:.2%}",
                    "2nd":        r["top2"].replace("_", " ").title(),
                    "3rd":        r["top3"].replace("_", " ").title(),
                    "Status":     r["status"],
                } for r in results_list])
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button("⬇ Download CSV", df_out.to_csv(index=False).encode(),
                                   f"batch_{run_name}.csv", "text/csv", use_container_width=True)

    #  Grad-CAM 
    elif "Grad" in mode:
        st.markdown('<p class="sec-label">Grad-CAM Visualisation</p>', unsafe_allow_html=True)

        if img_rgb is None:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🌡️</div>
              <div class="empty-text">Upload an aerial image on the left<br>to visualise model attention</div>
            </div>""", unsafe_allow_html=True)
        elif run_name == "vit_lora":
            st.markdown("""
            <div class="v-warn" style="margin:20px 0;">
              ⚠️  <strong>Grad-CAM is not available for ViT-B/16 + LoRA.</strong><br>
              ViT uses self-attention, not convolutions. Switch to Custom CNN or DenseNet-121.
            </div>""", unsafe_allow_html=True)
        else:
            display_arr  = cv2.resize(img_rgb, (sz, sz))
            tensor       = preprocess(img_rgb, cfg_dict)
            target_layer = get_target_layer(model, run_name)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].numpy()
            top_idx   = int(probs.argmax())
            top_label = class_names[top_idx]
            top_conf  = float(probs[top_idx])
            cc        = conf_color(top_conf)
            cam       = run_gradcam(model, target_layer, tensor, top_idx)

            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.markdown('<p class="sec-label" style="font-size:0.7rem;">Original</p>', unsafe_allow_html=True)
                st.image(display_arr, use_container_width=True)
            with c2:
                st.markdown('<p class="sec-label" style="font-size:0.7rem;">CAM Overlay</p>', unsafe_allow_html=True)
                if cam is not None:
                    st.image(overlay_cam(display_arr, cam, gradcam_alpha), use_container_width=True)
            with c3:
                st.markdown('<p class="sec-label" style="font-size:0.7rem;">Heatmap</p>', unsafe_allow_html=True)
                if cam is not None:
                    cam_r = cv2.resize(cam, (sz, sz))
                    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), use_container_width=True)

            emoji = CLASS_EMOJIS.get(top_label, "📍")
            st.markdown(f"""
            <div class="pred-hero" style="margin-top:18px;border-color:{cc}33;padding:22px 28px;">
              <div class="pred-scene">Model attention focused on</div>
              <div class="pred-name">{emoji} {top_label.replace("_"," ").title()}</div>
              <div class="pred-conf" style="color:{cc};">{top_conf:.1%}</div>
              <div class="pred-model">{sel['icon']} {sel['full']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="icard" style="margin-top:14px;">
              <div class="mcard-label">Reading the heatmap</div>
              🔴 <strong>Red / hot regions</strong> — areas the model weighted most heavily in its decision<br>
              🔵 <strong>Blue / cool regions</strong> — less influential to the final prediction<br>
              Adjust <em>Heatmap opacity</em> in the sidebar to control overlay blending
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<p class="sec-label" style="margin-top:16px;">Top {top_k} Predictions</p>', unsafe_allow_html=True)
            st.markdown(prob_bars_html(probs, class_names, top_k, top_idx), unsafe_allow_html=True)
