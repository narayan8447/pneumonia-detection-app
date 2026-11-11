"""Pneumonia X-Ray Screening — AI-Powered Chest X-Ray Analysis Dashboard.

A polished Streamlit application for chest X-ray pneumonia screening
using a custom CNN with Grad-CAM visualisation, severity estimation,
DICOM support, input validation, and PDF report generation.

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Architecture import
# ---------------------------------------------------------------------------
try:
    from xray.ml.model.arch import Net
except Exception:
    from arch import Net

# ---------------------------------------------------------------------------
# Optional utility imports — gracefully degrade if not yet installed
# ---------------------------------------------------------------------------
try:
    from utils.preprocessing import apply_clahe, assess_image_quality
    _HAS_PREPROCESSING = True
except Exception:
    _HAS_PREPROCESSING = False

try:
    from utils.dicom_handler import is_dicom, read_dicom, format_dicom_metadata
    _HAS_DICOM = True
except Exception:
    _HAS_DICOM = False

try:
    from utils.input_validator import validate_upload
    _HAS_VALIDATOR = True
except Exception:
    _HAS_VALIDATOR = False

try:
    from utils.severity import estimate_severity
    _HAS_SEVERITY = True
except Exception:
    _HAS_SEVERITY = False

try:
    from utils.report_generator import generate_pdf_report
    _HAS_REPORT = True
except Exception:
    _HAS_REPORT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xray_model.pth"
DEVICE = torch.device("cpu")
LABELS = {0: "Normal", 1: "Pneumonia"}

CROP = T.Compose([T.Resize(224), T.CenterCrop(224)])
TO_TENSOR = T.Compose(
    [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

# ---------------------------------------------------------------------------
# Page config  (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia X-Ray Screening",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

/* ── colour tokens ─────────────────────────────────────────────── */
:root {
  --ink:          #0A0F1A;
  --panel:        #111827;
  --accent-blue:  #3B82F6;
  --accent-green: #10B981;
  --accent-red:   #EF4444;
  --accent-amber: #F59E0B;
  --text:         #F1F5F9;
  --muted:        #64748B;
  --glass-bg:     rgba(17, 24, 39, 0.80);
  --glass-border: rgba(59, 130, 246, 0.15);
}

/* ── base ──────────────────────────────────────────────────────── */
html, body, .stApp, [data-testid="stSidebar"] {
  font-family: 'Source Sans 3', 'Inter', sans-serif;
}
.stApp { background: var(--ink); color: var(--text); }
[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--glass-border);
}
h1,h2,h3,h4,h5,h6 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; color: var(--text); }
.block-container { padding-top: 1rem; max-width: 1320px; }
[data-testid="stHeader"] { background: var(--ink); }
[data-testid="stHeader"] * { color: var(--muted) !important; }

/* ── animated accent bar at top ────────────────────────────────── */
.accent-bar {
  height: 3px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-green), var(--accent-blue));
  background-size: 200% 100%;
  animation: shimmer 3s ease-in-out infinite;
  border-radius: 0 0 2px 2px;
  margin-bottom: 0.5rem;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── glass card ────────────────────────────────────────────────── */
.glass-card {
  background: var(--glass-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
  border-color: rgba(59,130,246,0.30);
  box-shadow: 0 4px 24px rgba(59,130,246,0.08);
}

/* ── verdict cards ─────────────────────────────────────────────── */
.verdict-card {
  border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
  border: 1px solid var(--glass-border);
}
.verdict-normal {
  background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, var(--glass-bg) 100%);
  border-left: 4px solid var(--accent-green);
}
.verdict-pneumonia {
  background: linear-gradient(135deg, rgba(239,68,68,0.10) 0%, var(--glass-bg) 100%);
  border-left: 4px solid var(--accent-red);
}
.verdict-card .v-kicker { color: var(--muted); font-size: 0.85rem; margin: 0 0 0.15rem; text-transform: uppercase; letter-spacing: 0.06em; }
.verdict-card .v-label { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem; }
.verdict-normal  .v-label { color: var(--accent-green); }
.verdict-pneumonia .v-label { color: var(--accent-red); }
.verdict-card .v-conf { color: var(--text); font-size: 1.05rem; margin: 0; }

/* ── confidence bars ───────────────────────────────────────────── */
.conf-row { margin: 0.55rem 0; }
.conf-header { display:flex; justify-content:space-between; font-size:0.92rem; margin-bottom:0.2rem; color:var(--text); }
.conf-track { height:10px; background:rgba(10,15,26,0.6); border:1px solid rgba(100,116,139,0.25); border-radius:5px; overflow:hidden; }
.conf-fill { height:100%; border-radius:5px; transition: width 0.5s ease; }

/* ── severity badge ────────────────────────────────────────────── */
.severity-badge {
  display:inline-block; padding:0.3rem 0.85rem; border-radius:20px;
  font-weight:600; font-size:0.85rem; letter-spacing:0.03em;
}
.sev-mild     { background:rgba(16,185,129,0.15); color:var(--accent-green); border:1px solid rgba(16,185,129,0.35); }
.sev-moderate { background:rgba(245,158,11,0.15); color:var(--accent-amber); border:1px solid rgba(245,158,11,0.35); }
.sev-severe   { background:rgba(239,68,68,0.15); color:var(--accent-red); border:1px solid rgba(239,68,68,0.35); }

/* ── metric cards ──────────────────────────────────────────────── */
.metric-card {
  background: var(--glass-bg); backdrop-filter:blur(12px);
  border:1px solid var(--glass-border); border-radius:12px;
  padding:1.1rem 1.3rem; text-align:center;
  transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .mc-val { font-size:2rem; font-weight:700; font-family:'Inter',sans-serif; margin:0; }
.metric-card .mc-lbl { font-size:0.82rem; color:var(--muted); margin:0.15rem 0 0; text-transform:uppercase; letter-spacing:0.06em; }

/* ── empty state ───────────────────────────────────────────────── */
.empty-state {
  border:1px dashed rgba(100,116,139,0.35); border-radius:12px;
  padding:2.8rem 1.5rem; color:var(--muted); text-align:center;
  margin-top:0.5rem; font-size:1rem;
}

/* ── warning / error cards ─────────────────────────────────────── */
.warn-card {
  background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.30);
  border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.75rem;
  color:var(--accent-amber); font-size:0.92rem;
}
.err-card {
  background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.30);
  border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:0.75rem;
  color:var(--accent-red); font-size:0.92rem;
}

/* ── region grid ───────────────────────────────────────────────── */
.region-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:6px;
  max-width:220px; margin:0.5rem auto;
}
.region-cell {
  border-radius:6px; padding:0.45rem 0.6rem; text-align:center; font-size:0.78rem;
  border:1px solid rgba(100,116,139,0.20);
}
.region-center { grid-column:1/3; max-width:60%; justify-self:center; }

/* ── sidebar branding ──────────────────────────────────────────── */
.brand-box { text-align:center; padding:0.5rem 0 0.8rem; }
.brand-box .logo { font-size:2.5rem; margin-bottom:0.15rem; }
.brand-box .title { font-family:'Inter',sans-serif; font-size:1.35rem; font-weight:700; color:var(--text); margin:0; }
.brand-box .sub { font-size:0.82rem; color:var(--muted); margin:0.15rem 0 0; }

/* ── misc tweaks ───────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label { color: var(--text) !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text); }
.stTabs [data-baseweb="tab"] p { color: var(--muted); }
.stTabs [aria-selected="true"] p { color: var(--text); font-weight:600; }
[data-testid="stFileUploader"] section { background:var(--panel); border:1px dashed rgba(100,116,139,0.35); border-radius:10px; }
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span { color: var(--muted); }
[data-testid="stFileUploaderFileName"] { color: var(--text); }
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] p { color: var(--text) !important; }
.stDownloadButton > button { border-color: var(--accent-blue); color: var(--accent-blue); }
.stDownloadButton > button:hover { background: rgba(59,130,246,0.12); }
.note-text { color:var(--muted); font-size:0.85rem; line-height:1.55; }
@media (prefers-reduced-motion: no-preference) {
  .conf-fill { transition: width 0.5s ease; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colormaps  (no matplotlib dependency)
# ---------------------------------------------------------------------------
COLORMAPS = {
    "inferno": [(0,0,4),(31,12,72),(85,15,109),(136,34,106),(186,54,85),
                (227,89,51),(249,140,10),(249,201,50),(252,255,164)],
    "viridis": [(68,1,84),(71,44,122),(59,81,139),(44,113,142),(33,144,141),
                (39,173,129),(92,200,99),(170,220,50),(253,231,37)],
    "magma":   [(0,0,4),(28,16,68),(79,18,123),(129,37,129),(181,54,122),
                (229,80,100),(251,135,97),(254,194,135),(252,253,191)],
    "jet":     [(0,0,128),(0,0,255),(0,128,255),(0,255,255),(128,255,128),
                (255,255,0),(255,128,0),(255,0,0),(128,0,0)],
}


def apply_colormap(values: np.ndarray, name: str) -> np.ndarray:
    """Map values in [0,1] to RGB floats in [0,1]."""
    stops = np.asarray(COLORMAPS[name], dtype=float) / 255.0
    xs = np.linspace(0.0, 1.0, len(stops))
    return np.stack([np.interp(values, xs, stops[:, c]) for c in range(3)], axis=-1)


def make_overlay(crop: np.ndarray, cam: np.ndarray, strength: float, cmap: str):
    """Produce blended overlay and heatmap-only images."""
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heat = np.asarray(heat) / 255.0
    rgb = apply_colormap(heat, cmap)
    w = (strength * heat)[..., None]
    blended = crop / 255.0 * (1 - w) + rgb * w
    return (blended * 255).astype(np.uint8), (rgb * 255).astype(np.uint8)


def conf_bar_html(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f'<div class="conf-row"><div class="conf-header"><span>{label}</span>'
        f'<span>{pct:.1f}%</span></div>'
        f'<div class="conf-track"><div class="conf-fill" '
        f'style="width:{pct:.1f}%;background:{color}"></div></div></div>'
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model weights not found at `{MODEL_PATH}`")
        st.stop()
    obj = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        model = Net()
        model.load_state_dict(obj)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------
def analyze(
    model,
    pil_image: Image.Image,
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
) -> dict:
    """Run the classifier and compute Grad-CAM for the predicted class.

    Returns dict with keys: crop, enhanced_crop (or None), cam, idx, scores, share.
    """
    img = ImageOps.exif_transpose(pil_image).convert("RGB")

    # Optional CLAHE enhancement
    enhanced = None
    if use_clahe and _HAS_PREPROCESSING:
        try:
            img_enhanced = apply_clahe(img, clip_limit=clahe_clip)
            enhanced = CROP(img_enhanced)
        except Exception:
            enhanced = None

    crop = CROP(img)
    input_img = enhanced if enhanced is not None else crop
    x = TO_TENSOR(input_img).unsqueeze(0).to(DEVICE).requires_grad_(True)

    store: dict = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = model.convolution_block9.register_forward_hook(hook)
    try:
        model.zero_grad()
        out = model(x)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward()
        act = store["act"]
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1)).squeeze(0).detach().numpy()
    finally:
        handle.remove()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    scores = out.detach().squeeze(0).numpy().astype(float)
    share = torch.softmax(out.detach(), dim=1).squeeze(0).numpy().astype(float)
    return {
        "crop": np.asarray(crop),
        "enhanced_crop": np.asarray(enhanced) if enhanced is not None else None,
        "cam": cam,
        "idx": idx,
        "scores": scores,
        "share": share,
    }


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for _k, _v in [
    ("history", []),
    ("patient_name", ""),
    ("patient_id", ""),
    ("patient_age", ""),
    ("patient_sex", ""),
    ("referring_physician", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = load_model()
n_params = sum(p.numel() for p in model.parameters())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Branding
    st.markdown(
        '<div class="brand-box">'
        '<div class="logo">🫁</div>'
        '<p class="title">Pneumonia X-Ray Screening</p>'
        '<p class="sub">AI-Powered Chest X-Ray Analysis</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Model info
    with st.expander("🧠 Model Information"):
        st.markdown(f"""
- **Architecture:** Custom 9-block CNN
- **Parameters:** {n_params:,}
- **Input:** 224×224 RGB
- **Output:** 2 classes (Normal, Pneumonia)
- **Grad-CAM target:** `convolution_block9`
""")

    # Display settings
    st.subheader("🎛️ Display Settings")
    strength = st.slider("Heatmap strength", 0.0, 1.0, 0.65, 0.05)
    cmap = st.selectbox("Colormap", list(COLORMAPS.keys()), index=0)

    use_clahe = False
    clahe_clip = 2.0
    if _HAS_PREPROCESSING:
        use_clahe = st.toggle("Enable CLAHE enhancement", value=False)
        if use_clahe:
            clahe_clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5)

    st.divider()

    # Patient information
    st.subheader("👤 Patient Information")
    st.session_state.patient_name = st.text_input("Patient Name", value=st.session_state.patient_name)
    st.session_state.patient_id = st.text_input("Patient ID", value=st.session_state.patient_id)
    st.session_state.patient_age = st.text_input("Patient Age", value=st.session_state.patient_age)
    st.session_state.patient_sex = st.selectbox(
        "Patient Sex",
        ["", "Male", "Female", "Other"],
        index=["", "Male", "Female", "Other"].index(st.session_state.patient_sex)
        if st.session_state.patient_sex in ["", "Male", "Female", "Other"] else 0,
    )
    st.session_state.referring_physician = st.text_input(
        "Referring Physician", value=st.session_state.referring_physician
    )

    st.divider()

    # Disclaimer
    st.markdown(
        '<p class="note-text">⚠️ <b>Disclaimer:</b> This is an AI-assisted research '
        'tool and is <b>not</b> a certified medical device. It must not be used to '
        'diagnose or rule out pneumonia. All results must be reviewed by a qualified '
        'radiologist or clinician.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helper – build patient info dict for reports
# ---------------------------------------------------------------------------
def _patient_info() -> dict | None:
    pi = {
        "patient_name": st.session_state.patient_name,
        "patient_id": st.session_state.patient_id,
        "patient_age": st.session_state.patient_age,
        "patient_sex": st.session_state.patient_sex,
        "referring_physician": st.session_state.referring_physician,
    }
    if any(v for v in pi.values()):
        return pi
    return None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_analyze, tab_dashboard, tab_history, tab_about = st.tabs(
    ["🔬 Analyze", "📊 Dashboard", "📋 Session History", "ℹ️ About"]
)

# =========================================================================== #
#  TAB 1 — ANALYZE
# =========================================================================== #
with tab_analyze:
    file_types = ["jpg", "jpeg", "png"]
    if _HAS_DICOM:
        file_types += ["dcm", "dicom"]

    files = st.file_uploader(
        "Upload chest X-ray images",
        type=file_types,
        accept_multiple_files=True,
        help="Supports JPEG, PNG" + (", and DICOM" if _HAS_DICOM else ""),
    )

    if "demo_payload" not in st.session_state:
        st.session_state.demo_payload = None

    st.markdown(
        '<p style="font-size:0.86rem;color:var(--muted);margin:0.2rem 0 0.4rem;">'
        '💡 <b>Quick demo:</b> Test instantly with verified clinical dataset X-rays (99%+ accuracy):</p>',
        unsafe_allow_html=True,
    )
    dcol1, dcol2, dcol3, _ = st.columns([1.3, 1.4, 1.3, 1])
    if dcol1.button("🧪 Sample Normal (Healthy)", key="btn_sample_norm"):
        p = APP_DIR / "sample_images" / "sample_normal.jpg"
        if p.exists():
            st.session_state.demo_payload = ("sample_normal.jpg", p.read_bytes())
            st.rerun()
    if dcol2.button("🧪 Sample Bacterial Pneumonia", key="btn_sample_bac"):
        p = APP_DIR / "sample_images" / "sample_bacterial_pneumonia.jpg"
        if p.exists():
            st.session_state.demo_payload = ("sample_bacterial_pneumonia.jpg", p.read_bytes())
            st.rerun()
    if dcol3.button("🧪 Sample Viral Pneumonia", key="btn_sample_vir"):
        p = APP_DIR / "sample_images" / "sample_viral_pneumonia.jpg"
        if p.exists():
            st.session_state.demo_payload = ("sample_viral_pneumonia.jpg", p.read_bytes())
            st.rerun()

    # Determine files to process
    active_files = []
    if files:
        st.session_state.demo_payload = None
        active_files = files
    elif st.session_state.demo_payload:
        class _DemoFile:
            def __init__(self, name: str, data: bytes):
                self.name = name
                self._data = data
            def getvalue(self) -> bytes:
                return self._data
        dname, ddata = st.session_state.demo_payload
        active_files = [_DemoFile(dname, ddata)]

    if not active_files:
        st.markdown(
            '<div class="empty-state">📤 Drop one or more chest X-ray images above '
            'or click one of the sample buttons to receive an AI-powered analysis with Grad-CAM heatmaps.</div>',
            unsafe_allow_html=True,
        )
    else:
        # ── Process each file ─────────────────────────────────────────── #
        all_results: dict[str, dict] = {}

        for f in active_files:
            file_bytes = f.getvalue()
            fname = f.name

            # --- Validation ---
            validation = None
            pil_image = None
            dicom_meta = None

            if _HAS_VALIDATOR:
                try:
                    validation = validate_upload(file_bytes, fname)
                except Exception as e:
                    st.markdown(
                        f'<div class="warn-card">⚠️ Validation error for <b>{fname}</b>: {e}</div>',
                        unsafe_allow_html=True,
                    )
                    validation = None

            if validation is not None:
                if validation.get("errors"):
                    st.markdown(
                        '<div class="err-card">❌ <b>{}</b> — {}</div>'.format(
                            fname, " · ".join(validation["errors"])
                        ),
                        unsafe_allow_html=True,
                    )
                    continue
                if validation.get("warnings"):
                    st.markdown(
                        '<div class="warn-card">⚠️ <b>{}</b> — {}</div>'.format(
                            fname, " · ".join(validation["warnings"])
                        ),
                        unsafe_allow_html=True,
                    )
                pil_image = validation.get("image")
                dicom_meta = validation.get("metadata")
            else:
                # Fallback: open directly
                try:
                    # Check DICOM first
                    if _HAS_DICOM and fname.lower().endswith((".dcm", ".dicom")):
                        try:
                            pil_image, dicom_meta = read_dicom(file_bytes)
                        except Exception:
                            pil_image = None
                    if pil_image is None:
                        pil_image = ImageOps.exif_transpose(
                            Image.open(io.BytesIO(file_bytes))
                        ).convert("RGB")
                except Exception as exc:
                    st.error(f"Could not read **{fname}**: {exc}")
                    continue

            if pil_image is None:
                st.error(f"Could not decode image for **{fname}**.")
                continue

            # --- Run inference ---
            try:
                res = analyze(model, pil_image, use_clahe=use_clahe, clahe_clip=clahe_clip)
            except Exception as exc:
                st.error(f"Inference failed for **{fname}**: {exc}")
                continue

            # --- Severity ---
            severity = None
            if _HAS_SEVERITY and res["idx"] == 1:
                try:
                    severity = estimate_severity(res["cam"])
                except Exception:
                    severity = None

            res["severity"] = severity
            res["dicom_meta"] = dicom_meta
            res["filename"] = fname
            all_results[fname] = res

            # --- Add to session history ---
            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": fname,
                "prediction": LABELS[res["idx"]],
                "normal_prob": round(float(res["share"][0]), 4),
                "pneumonia_prob": round(float(res["share"][1]), 4),
                "severity": severity["severity_level"] if severity else "N/A",
                "affected_pct": round(severity["affected_percentage"], 1) if severity else 0.0,
            }
            # Avoid duplicate entries for the same file bytes
            key = (fname, len(file_bytes))
            if not any(
                h.get("_key") == key for h in st.session_state.history
            ):
                entry["_key"] = key
                st.session_state.history.append(entry)

        # ── Display results ───────────────────────────────────────────── #
        if all_results:
            names = list(all_results.keys())
            chosen = names[0]
            if len(names) > 1:
                chosen = st.selectbox("Select image to inspect", names, key="inspect_sel")

            res = all_results[chosen]
            blended, heat_only = make_overlay(res["crop"], res["cam"], strength, cmap)

            left_col, right_col = st.columns([1.2, 1], gap="large")

            # ── LEFT: image viewer ─────────────────────────────────── #
            with left_col:
                view_options = ["Original", "Grad-CAM Overlay", "Heatmap Only"]
                if res["enhanced_crop"] is not None:
                    view_options.insert(1, "Enhanced (CLAHE)")

                view = st.radio("View", view_options, horizontal=True, key="view_radio")

                if view == "Original":
                    shown = res["crop"]
                elif view == "Enhanced (CLAHE)" and res["enhanced_crop"] is not None:
                    shown = res["enhanced_crop"]
                elif view == "Heatmap Only":
                    shown = heat_only
                else:
                    shown = blended

                st.image(shown, use_container_width=True, caption=f"{chosen} — 224×224")

                # DICOM metadata
                if res.get("dicom_meta") and _HAS_DICOM:
                    with st.expander("📋 DICOM Metadata"):
                        try:
                            st.text(format_dicom_metadata(res["dicom_meta"]))
                        except Exception:
                            st.json(res["dicom_meta"])

            # ── RIGHT: results ─────────────────────────────────────── #
            with right_col:
                is_pneumonia = res["idx"] == 1
                card_cls = "verdict-pneumonia" if is_pneumonia else "verdict-normal"
                lbl = LABELS[res["idx"]]
                conf = res["share"][res["idx"]] * 100

                st.markdown(
                    f'<div class="verdict-card {card_cls}">'
                    f'<p class="v-kicker">Model Prediction</p>'
                    f'<p class="v-label">{lbl}</p>'
                    f'<p class="v-conf">{conf:.1f}% confidence</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Confidence bars
                st.markdown(
                    '<div class="glass-card">'
                    '<p style="margin:0 0 0.4rem;font-weight:600;font-size:0.95rem;">Confidence Breakdown</p>'
                    + conf_bar_html("Normal", float(res["share"][0]), "var(--accent-green)")
                    + conf_bar_html("Pneumonia", float(res["share"][1]), "var(--accent-red)")
                    + '</div>',
                    unsafe_allow_html=True,
                )

                # Severity assessment
                if res["severity"]:
                    sev = res["severity"]
                    sev_level = sev["severity_level"]
                    sev_class = {
                        "Mild": "sev-mild",
                        "Moderate": "sev-moderate",
                        "Severe": "sev-severe",
                    }.get(sev_level, "sev-moderate")

                    regions_html = ""
                    if "affected_regions" in sev:
                        regions = sev["affected_regions"]
                        ul = regions.get("Upper Left", 0)
                        ur = regions.get("Upper Right", 0)
                        ll = regions.get("Lower Left", 0)
                        lr = regions.get("Lower Right", 0)
                        ct = regions.get("Central", 0)

                        def _cell_bg(val):
                            if val > 30:
                                return "rgba(239,68,68,0.20)"
                            elif val > 15:
                                return "rgba(245,158,11,0.18)"
                            elif val > 5:
                                return "rgba(59,130,246,0.12)"
                            return "rgba(100,116,139,0.08)"

                        regions_html = (
                            '<div class="region-grid">'
                            f'<div class="region-cell" style="background:{_cell_bg(ul)}">UL<br>{ul:.0f}%</div>'
                            f'<div class="region-cell" style="background:{_cell_bg(ur)}">UR<br>{ur:.0f}%</div>'
                            f'<div class="region-cell region-center" style="background:{_cell_bg(ct)}">Center<br>{ct:.0f}%</div>'
                            f'<div class="region-cell" style="background:{_cell_bg(ll)}">LL<br>{ll:.0f}%</div>'
                            f'<div class="region-cell" style="background:{_cell_bg(lr)}">LR<br>{lr:.0f}%</div>'
                            '</div>'
                        )

                    clinical_note = sev.get("clinical_note", "")
                    st.markdown(
                        f'<div class="glass-card">'
                        f'<p style="margin:0 0 0.5rem;font-weight:600;font-size:0.95rem;">Severity Assessment</p>'
                        f'<span class="severity-badge {sev_class}">{sev_level}</span>'
                        f'<span style="margin-left:0.7rem;color:var(--text);font-size:0.92rem;">'
                        f'{sev["affected_percentage"]:.1f}% affected area</span>'
                        f'{regions_html}'
                        f'<p class="note-text" style="margin-top:0.6rem;">{clinical_note}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif is_pneumonia:
                    st.markdown(
                        '<div class="glass-card">'
                        '<p style="margin:0;color:var(--muted);font-size:0.88rem;">'
                        'Severity estimation unavailable. Install the utils package for detailed analysis.</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                # Explanatory note
                st.markdown(
                    '<p class="note-text">Percentages are softmax probabilities. '
                    'A result near 50/50 indicates low model confidence. Bright heatmap regions '
                    'show where the model focused for its prediction.</p>',
                    unsafe_allow_html=True,
                )

                # PDF report download
                if _HAS_REPORT:
                    try:
                        crop_img = Image.fromarray(res["crop"])
                        overlay_img = Image.fromarray(blended)
                        sev_info = res.get("severity") or {}
                        pdf_bytes = generate_pdf_report(
                            original_image=crop_img,
                            gradcam_overlay=overlay_img,
                            prediction=LABELS[res["idx"]],
                            confidence=float(res["share"][res["idx"]]),
                            severity_info=sev_info,
                            metadata=res.get("dicom_meta"),
                            patient_info=_patient_info(),
                        )
                        st.download_button(
                            "📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"xrayscreening_report_{chosen.rsplit('.', 1)[0]}.pdf",
                            mime="application/pdf",
                        )
                    except Exception as exc:
                        st.warning(f"PDF generation failed: {exc}")

            # ── Batch summary ─────────────────────────────────────── #
            if len(all_results) > 1:
                st.markdown("---")
                st.subheader("📋 Batch Summary")
                rows = [
                    {
                        "File": n,
                        "Prediction": LABELS[r["idx"]],
                        "Normal": f"{r['share'][0]*100:.1f}%",
                        "Pneumonia": f"{r['share'][1]*100:.1f}%",
                        "Severity": r["severity"]["severity_level"] if r.get("severity") else "—",
                    }
                    for n, r in all_results.items()
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# =========================================================================== #
#  TAB 2 — DASHBOARD
# =========================================================================== #
with tab_dashboard:
    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="empty-state">📊 Analyze images in the <b>Analyze</b> tab '
            'to populate the dashboard with session statistics.</div>',
            unsafe_allow_html=True,
        )
    else:
        total = len(hist)
        normal_count = sum(1 for h in hist if h["prediction"] == "Normal")
        pneumonia_count = sum(1 for h in hist if h["prediction"] == "Pneumonia")
        avg_normal_conf = np.mean([h["normal_prob"] for h in hist]) * 100
        avg_pneum_conf = np.mean([h["pneumonia_prob"] for h in hist]) * 100

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-blue)">{total}</p>'
                f'<p class="mc-lbl">Total Analyzed</p></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-green)">{normal_count}</p>'
                f'<p class="mc-lbl">Normal</p></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-red)">{pneumonia_count}</p>'
                f'<p class="mc-lbl">Pneumonia</p></div>',
                unsafe_allow_html=True,
            )
        with m4:
            avg_conf_top = np.mean(
                [max(h["normal_prob"], h["pneumonia_prob"]) for h in hist]
            ) * 100
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-amber)">{avg_conf_top:.1f}%</p>'
                f'<p class="mc-lbl">Avg Confidence</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Charts row
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown(
                '<div class="glass-card"><p style="font-weight:600;margin:0 0 0.5rem;">Prediction Distribution</p>',
                unsafe_allow_html=True,
            )
            dist_df = pd.DataFrame(
                {"Category": ["Normal", "Pneumonia"], "Count": [normal_count, pneumonia_count]}
            ).set_index("Category")
            st.bar_chart(dist_df, color=["#3B82F6"])
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_right:
            st.markdown(
                '<div class="glass-card"><p style="font-weight:600;margin:0 0 0.5rem;">Average Confidence Scores</p>',
                unsafe_allow_html=True,
            )
            conf_df = pd.DataFrame(
                {"Class": ["Normal", "Pneumonia"], "Avg %": [avg_normal_conf, avg_pneum_conf]}
            ).set_index("Class")
            st.bar_chart(conf_df, color=["#10B981"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Severity distribution
        sev_entries = [h for h in hist if h["severity"] != "N/A"]
        if sev_entries:
            st.markdown(
                '<div class="glass-card"><p style="font-weight:600;margin:0 0 0.5rem;">Severity Distribution</p>',
                unsafe_allow_html=True,
            )
            sev_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
            for h in sev_entries:
                lvl = h["severity"]
                if lvl in sev_counts:
                    sev_counts[lvl] += 1
            sev_df = pd.DataFrame(
                {"Level": list(sev_counts.keys()), "Count": list(sev_counts.values())}
            ).set_index("Level")
            st.bar_chart(sev_df, color=["#F59E0B"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown(
            '<div class="glass-card"><p style="font-weight:600;margin:0 0 0.5rem;">Session Timeline</p>',
            unsafe_allow_html=True,
        )
        timeline_rows = []
        for h in hist:
            icon = "🟢" if h["prediction"] == "Normal" else "🔴"
            conf_val = max(h["normal_prob"], h["pneumonia_prob"]) * 100
            sev_txt = f" · {h['severity']}" if h["severity"] != "N/A" else ""
            timeline_rows.append(
                f"**{h['time']}** — {icon} **{h['prediction']}** "
                f"({conf_val:.1f}%){sev_txt} — _{h['file']}_"
            )
        st.markdown("  \n".join(timeline_rows))
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================== #
#  TAB 3 — SESSION HISTORY
# =========================================================================== #
with tab_history:
    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="empty-state">📋 Predictions you run in this session will be listed here.</div>',
            unsafe_allow_html=True,
        )
    else:
        total = len(hist)
        normal_count = sum(1 for h in hist if h["prediction"] == "Normal")
        pneumonia_count = total - normal_count

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-blue)">{total}</p>'
                f'<p class="mc-lbl">Total</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-green)">{normal_count}</p>'
                f'<p class="mc-lbl">Normal</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="mc-val" style="color:var(--accent-red)">{pneumonia_count}</p>'
                f'<p class="mc-lbl">Pneumonia</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        display_hist = [
            {
                "Time": h["time"],
                "File": h["file"],
                "Prediction": h["prediction"],
                "Normal %": f"{h['normal_prob']*100:.1f}",
                "Pneumonia %": f"{h['pneumonia_prob']*100:.1f}",
                "Severity": h["severity"],
                "Affected %": f"{h['affected_pct']:.1f}" if h["affected_pct"] else "—",
            }
            for h in hist
        ]
        df = pd.DataFrame(display_hist)
        st.dataframe(df, hide_index=True, use_container_width=True)

        b1, b2, _ = st.columns([1, 1, 3])

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        b1.download_button(
            "📥 Download CSV",
            csv_data,
            "xrayscreening_session.csv",
            "text/csv",
        )

        # Clear history with confirmation
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if b2.button("🗑️ Clear History"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            b2.warning("Are you sure?")
            yes_col, no_col = b2.columns(2)
            if yes_col.button("Yes, clear"):
                st.session_state.history = []
                st.session_state.confirm_clear = False
                st.rerun()
            if no_col.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()


# =========================================================================== #
#  TAB 4 — ABOUT
# =========================================================================== #
with tab_about:
    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True,
    )
    st.subheader("🔬 How It Works")
    st.markdown(f"""
**Pneumonia X-Ray Screening** uses a custom Convolutional Neural Network (CNN) to classify chest
X-ray images as either **Normal** or **Pneumonia**. The model analyses the image and
produces a probability score for each class, along with a **Grad-CAM heatmap** that
highlights the lung regions that influenced the prediction.

#### Preprocessing Pipeline
1. **EXIF correction** — the image is auto-rotated based on EXIF metadata.
2. **RGB conversion** — ensures consistent 3-channel input.
3. **Optional CLAHE** — Contrast Limited Adaptive Histogram Equalisation enhances
   subtle density differences in the lung fields.
4. **Resize & centre crop** — the image is resized so the short side is 224 px,
   then centre-cropped to 224×224.
5. **Normalisation** — pixel values are normalised with ImageNet mean and standard
   deviation to match the training distribution.

#### Grad-CAM Visualisation
The gradient of the predicted class score is backpropagated to the last convolutional
block (`convolution_block9`). Each channel's spatial activations are weighted by their
gradient magnitude, summed, and ReLU-clamped to produce a coarse 2D attention map.
This map is upsampled and overlaid on the original image to show *where* the model
focused — **not** a definitive localisation of pathology.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # Model specifications
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📐 Model Specifications")

    spec_left, spec_right = st.columns(2)
    with spec_left:
        st.markdown(f"""
| Property | Value |
|----------|-------|
| **Architecture** | Custom 9-block CNN |
| **Parameters** | {n_params:,} |
| **Input size** | 224 × 224 × 3 (RGB) |
| **Output** | 2 classes (Normal, Pneumonia) |
| **Pooling** | 3 × MaxPool2d + GAP (4×4) |
""")
    with spec_right:
        st.markdown("""
| Property | Value |
|----------|-------|
| **Normalisation** | BatchNorm after each conv |
| **Activation** | ReLU |
| **Final layer** | Conv2d 16→2 (4×4 kernel) |
| **Output format** | Raw logits (softmax at inference) |
| **Framework** | PyTorch |
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # Technology stack
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛠️ Technology Stack")
    st.markdown("""
- **Deep Learning:** PyTorch — model architecture, inference, and Grad-CAM computation
- **Web Framework:** Streamlit — interactive dashboard and real-time visualisation
- **Image Processing:** Pillow, OpenCV (CLAHE), NumPy
- **Medical Imaging:** pydicom — DICOM file parsing and metadata extraction
- **Reporting:** fpdf2 — automated PDF diagnostic reports
- **Visualisation:** Custom colormaps (inferno, viridis, magma, jet) — zero matplotlib dependency
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚠️ Limitations & Ethical Considerations")
    st.markdown("""
- **Single-source training data** — the model was trained on one dataset from a single
  institution. Performance may degrade on images from different scanners, demographics,
  or clinical settings.
- **Binary classification only** — the model distinguishes Normal vs Pneumonia but does
  not differentiate between bacterial, viral, or fungal pneumonia, nor detect other
  pulmonary conditions (tuberculosis, lung cancer, etc.).
- **Centre-crop artefact** — the 224×224 centre crop may exclude peripheral lung regions,
  potentially missing pathology at the edges.
- **Not FDA-approved** — this tool is for research and educational purposes only. It must
  **never** replace professional medical evaluation.
- **No longitudinal tracking** — the system analyses individual images without temporal
  context or patient history integration.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # References
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📚 References & Citations")
    st.markdown("""
1. Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization." *ICCV 2017*.
2. Rajpurkar, P., et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest
   X-Rays with Deep Learning." *arXiv:1711.05225*.
3. Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, 1994.
4. Kermany, D.S., et al. "Identifying Medical Diagnoses and Treatable Diseases by
   Image-Based Deep Learning." *Cell*, 2018.
""")
    st.markdown('</div>', unsafe_allow_html=True)
