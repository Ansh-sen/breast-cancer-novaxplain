import os
import sys
import base64
from datetime import datetime

current_file = os.path.abspath(__file__)
app_dir      = os.path.dirname(current_file)
project_root = os.path.dirname(app_dir)
sys.path.insert(0, project_root)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import streamlit as st
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64 as b64lib

from analysis.gradcam import (load_and_preprocess_image,
                                   get_gradcam_heatmap,
                                   create_gradcam_figure)
from analysis.tumor_localization import detect_tumor_region
from analysis.morphology_analysis import analyze_cell_morphology
from analysis.severity_estimator import estimate_severity
from reports.report_generator import generate_medical_report, save_report_to_file
from models.model_loader import load_model
from models.inference import safe_predict
from src.batch_processing import BatchAnalyzer, validate_batch_files
from src.nova_explanation import NovaExplainer
from src.patient_utils import generate_patient_data, get_patient_display_html
from src.image_validator import validate_histopathology_image
from src.wsi_analyzer import run_wsi_analysis

IMG_SIZE  = (96, 96)
NOW       = datetime.now().strftime("%d %b %Y  %H:%M")

st.set_page_config(
    page_title="GradVision | Clinical Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Playfair+Display:wght@700;900&display=swap');

/* ── TOKENS ─────────────────────────────────────────────────── */
:root {
  --bg:          #0E1117;
  --surface:     #111827;
  --surface-2:   #1a2235;
  --surface-3:   #212d42;
  --border:      rgba(255,255,255,0.07);
  --border-hi:   rgba(255,255,255,0.14);

  --text-1: #f0f4ff;
  --text-2: #8b9cbf;
  --text-3: #4a5878;

  --amber:       #f59e0b;
  --amber-dim:   rgba(245,158,11,0.12);
  --amber-glow:  rgba(245,158,11,0.25);

  --green:       #10b981;
  --green-dim:   rgba(16,185,129,0.12);
  --red:         #f43f5e;
  --red-dim:     rgba(244,63,94,0.12);
  --blue:        #3b82f6;
  --blue-dim:    rgba(59,130,246,0.12);

  --font-body: 'IBM Plex Sans', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-disp: 'Playfair Display', serif;

  --radius-sm: 8px;
  --radius-md: 14px;
  --radius-lg: 20px;
}

/* ── RESET ──────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
header, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }

/* ── APP SHELL ──────────────────────────────────────────────── */
.stApp {
  background: var(--bg) !important;
  font-family: var(--font-body) !important;
  color: var(--text-1) !important;
  background-image:
    radial-gradient(ellipse 800px 600px at 20% -10%, rgba(59,130,246,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 600px 400px at 85% 90%, rgba(245,158,11,0.04) 0%, transparent 50%);
}

/* ── MAIN CONTENT AREA ──────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
  margin-left: 320px !important;
}

/* ── SIDEBAR ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border-hi) !important;
  display: block !important;
  visibility: visible !important;
  width: 320px !important;
  min-width: 280px !important;
  position: fixed !important;
  left: 0 !important;
  top: 0 !important;
  height: 100vh !important;
  z-index: 99 !important;
  overflow-y: auto !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text-1) !important; }

/* Force sidebar always visible - hide collapse button */
[data-testid="stSidebarCollapseButton"] { display: none !important; }  
.btnBackToCollapseOpts { display: none !important; }

/* Ensure emoji and symbols display properly */
* { font-variant-numeric: normal !important; }
body, html { font-feature-settings: "normal" !important; }

/* ── TYPOGRAPHY ─────────────────────────────────────────────── */
.gv-wordmark {
  font-family: var(--font-disp);
  font-size: 1.5rem;
  font-weight: 900;
  color: var(--text-1);
  letter-spacing: -0.02em;
  line-height: 1;
}
.gv-tagline {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  color: var(--amber);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 4px;
}
.section-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-3);
  font-weight: 500;
  margin-bottom: 0.75rem;
}
.hero-title {
  font-family: var(--font-disp);
  font-size: 2.4rem;
  font-weight: 900;
  color: var(--text-1);
  letter-spacing: -0.04em;
  line-height: 1;
  margin: 0;
}
.hero-sub {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--amber);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-top: 6px;
}

/* ── CARDS ──────────────────────────────────────────────────── */
.cv-card {
  background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.6rem;
  margin-bottom: 1.2rem;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.cv-card:hover { 
  border-color: var(--border-hi);
  box-shadow: 0 4px 16px rgba(59,130,246,0.1);
  transform: translateY(-2px);
}

/* ── EHR SIDEBAR COMPONENTS ─────────────────────────────────── */
.ehr-header {
  background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%);
  border-bottom: 2px solid var(--border-hi);
  padding: 1.6rem 1.2rem 1.2rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.ehr-section {
  padding: 1.2rem 1.2rem;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(255,255,255,0.02) 0%, transparent 100%);
  transition: all 0.2s;
}
.ehr-section:hover {
  background: linear-gradient(135deg, rgba(59,130,246,0.05) 0%, transparent 100%);
}
.ehr-section-title {
  font-family: var(--font-mono);
  font-size: 0.55rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 0.7rem;
}
.ehr-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.3rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: 0.8rem;
}
.ehr-row:last-child { border-bottom: none; }
.ehr-key { color: var(--text-3); font-family: var(--font-mono); font-size: 0.68rem; }
.ehr-val { color: var(--text-1); font-weight: 500; font-size: 0.78rem; }
.ehr-val.warn  { color: var(--amber); }
.ehr-val.alert { color: var(--red);   font-weight: 700; }
.ehr-val.ok    { color: var(--green); }
.status-dot {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; margin-right: 6px;
  animation: pulse-dot 2.4s ease-in-out infinite;
}
.dot-green { background: var(--green); box-shadow: 0 0 0 3px rgba(16,185,129,0.2); }
.dot-red   { background: var(--red);   box-shadow: 0 0 0 3px rgba(244,63,94,0.2); }
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.5; }
}

/* ── METADATA STRIP ──────────────────────────────────────────── */
.meta-strip {
  background: #060b14;
  border: 1px solid rgba(245,158,11,0.2);
  border-bottom: none;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  padding: 0.5rem 0.9rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.meta-item {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-3);
}
.meta-item span { color: var(--amber); font-weight: 600; }
.meta-live {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  color: var(--green);
  letter-spacing: 2px;
  animation: blink 1.8s step-end infinite;
}
@keyframes blink { 50% { opacity: 0.25; } }
.img-frame {
  border: 1px solid rgba(245,158,11,0.2);
  border-top: none;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm);
  overflow: hidden;
}

/* ── SCAN ANIMATION ──────────────────────────────────────────── */
.scan-wrap { position: relative; overflow: hidden; }
.scan-wrap::after {
  content: '';
  position: absolute;
  left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--amber) 40%,
              var(--green) 60%, transparent 100%);
  animation: sweep 2.2s ease-in-out infinite;
  top: 0; pointer-events: none;
}
@keyframes sweep {
  0%   { top: 0%;   opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

/* ── RESULT STATE ────────────────────────────────────────────── */
.result-banner {
  border-radius: var(--radius-md);
  padding: 1.8rem;
  margin-bottom: 1.2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-left: 4px solid transparent;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.banner-mal {
  background: linear-gradient(135deg, rgba(244,63,94,0.15) 0%, rgba(244,63,94,0.05) 100%);
  border-color: #f43f5e;
  border: 2px solid #f43f5e;
  border-left: 4px solid #f43f5e;
}
.banner-ben {
  background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.05) 100%);
  border-color: #10b981;
  border: 2px solid #10b981;
  border-left: 4px solid #10b981;
}
.banner-unc {
  background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(245,158,11,0.05) 100%);
  border-color: #f59e0b;
  border: 2px solid #f59e0b;
  border-left: 4px solid #f59e0b;
}
.result-label {
  font-family: var(--font-disp);
  font-size: 2.8rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1;
}
.result-sublabel {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  opacity: 0.55;
  margin-bottom: 4px;
}
.p-score {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-2);
  margin-top: 6px;
}

/* ── GAUGE ───────────────────────────────────────────────────── */
.gauge-wrap { margin: 1.4rem 0; }
.gauge-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.gauge-title {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-3);
  font-weight: 600;
}
.gauge-pct {
  font-family: var(--font-mono);
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: 1px;
}
.gauge-track {
  height: 8px;
  background: var(--surface-3);
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}
.gauge-bar {
  height: 100%;
  border-radius: 999px;
  position: relative;
  box-shadow: 0 0 8px currentColor;
}
.gauge-bar::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 70%, rgba(255,255,255,0.4));
}
.bar-mal { background: linear-gradient(90deg, #be123c, #f43f5e); }
.bar-ben { background: linear-gradient(90deg, #059669, #10b981); }
.bar-unc { background: linear-gradient(90deg, #b45309, #f59e0b); }

/* ── NOVA SYNTHESIS ──────────────────────────────────────────── */
.nova-block {
  background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(59,130,246,0.05) 100%);
  border-left: 4px solid var(--blue);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 1.2rem 1.4rem;
  margin-top: 1.2rem;
  box-shadow: 0 2px 8px rgba(59,130,246,0.1);
}
.nova-block-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--blue);
  margin-bottom: 0.75rem;
  font-weight: 700;
}
.nova-block-text {
  font-size: 0.93rem;
  line-height: 1.8;
  color: var(--text-2);
  letter-spacing: 0.3px;
  max-height: 280px;
  overflow-y: auto;
  padding-right: 0.8rem;
}
.nova-block-text::-webkit-scrollbar {
  width: 6px;
}
.nova-block-text::-webkit-scrollbar-track {
  background: rgba(59,130,246,0.05);
  border-radius: 3px;
}
.nova-block-text::-webkit-scrollbar-thumb {
  background: rgba(59,130,246,0.4);
  border-radius: 3px;
}
.nova-block-text::-webkit-scrollbar-thumb:hover {
  background: rgba(59,130,246,0.6);
}

/* ── METRIC TILES ────────────────────────────────────────────── */
.tile-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.8rem;
  margin: 1.2rem 0;
}
.tile {
  background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%);
  border: 1px solid var(--border-hi);
  border-radius: var(--radius-sm);
  padding: 0.95rem 1.2rem;
  transition: all 0.2s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.tile:hover {
  border-color: var(--amber);
  box-shadow: 0 4px 8px rgba(245,158,11,0.1);
  transform: translateY(-1px);
}
.tile-label {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 0.5rem;
  font-weight: 600;
}
.tile-val {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-1);
  letter-spacing: 0.3px;
}

/* ── UNCERTAIN BADGE ─────────────────────────────────────────── */
.unc-badge {
  display: inline-block;
  background: var(--amber-dim);
  border: 1px solid rgba(245,158,11,0.35);
  color: var(--amber);
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 999px;
  margin-top: 8px;
}

/* ── DEEP ANALYSIS OUTPUT ────────────────────────────────────── */
.morph-output {
  background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%);
  border: 1px solid var(--border-hi);
  border-left: 3px solid var(--green);
  border-radius: var(--radius-md);
  padding: 1.6rem;
  font-size: 0.92rem;
  line-height: 1.9;
  color: var(--text-2);
  min-height: 200px;
  margin-top: 1rem;
  box-shadow: 0 2px 8px rgba(16,185,129,0.1);
}
.morph-empty {
  background: linear-gradient(135deg, rgba(59,130,246,0.05) 0%, transparent 100%);
  border: 2px dashed var(--border-hi);
  border-radius: var(--radius-md);
  padding: 3.5rem 2rem;
  text-align: center;
  color: var(--text-3);
  margin-top: 1rem;
}
.morph-empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.7; }
.morph-empty-title { font-weight: 600; color: var(--text-2); font-size: 0.95rem; margin-bottom: 0.3rem; }
.morph-empty-sub {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  margin-top: 0.4rem;
  color: var(--text-3);
  letter-spacing: 1px;
}

/* ── TABS ────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--surface) !important;
  border-radius: var(--radius-md) !important;
  padding: 5px !important;
  border: 1px solid var(--border) !important;
  gap: 3px !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-body) !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  color: var(--text-2) !important;
  padding: 0.45rem 1.1rem !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.18s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: var(--surface-2) !important;
  color: var(--text-1) !important;
  border: 1px solid var(--border-hi) !important;
}

/* ── BUTTONS ─────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--amber) 0%, #fbbf24 100%) !important;
  color: #000 !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-body) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  padding: 0.7rem 1.8rem !important;
  width: 100% !important;
  letter-spacing: 0.8px !important;
  transition: all 0.3s !important;
  box-shadow: 0 4px 16px var(--amber-glow) !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: rgba(255,255,255,0.2);
  transition: left 0.3s;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #fbbf24 0%, #fcd34d 100%) !important;
  box-shadow: 0 6px 24px var(--amber-glow) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:hover::before {
  left: 100%;
}

/* ── CHAT ────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: var(--radius-md) !important;
  margin-bottom: 0.7rem !important;
  padding: 1rem !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}
[data-testid="stChatInput"] > div {
  background: var(--surface-2) !important;
  border: 2px solid var(--border-hi) !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
}

/* ── MISC STREAMLIT OVERRIDES ────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, transparent 100%) !important;
  border: 2px dashed var(--amber) !important;
  border-radius: var(--radius-md) !important;
  padding: 2rem !important;
  transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
  background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, transparent 100%) !important;
  border-color: #fbbf24 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 4px rgba(245,158,11,0.2) !important;
}
.stSuccess, .stError, .stInfo, .stWarning { 
  border-radius: var(--radius-sm) !important;
  border-left: 4px solid !important;
}
.stSuccess {
  background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, transparent 100%) !important;
  border-color: var(--green) !important;
}
.stError {
  background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, transparent 100%) !important;
  border-color: var(--red) !important;
}
.stInfo {
  background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, transparent 100%) !important;
  border-color: var(--blue) !important;
}
.stWarning {
  background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, transparent 100%) !important;
  border-color: var(--amber) !important;
}

/* ── FOOTER ──────────────────────────────────────────────────── */
.gv-footer {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-3);
  text-align: center;
  padding: 2rem 0 1rem;
  border-top: 1px solid var(--border);
  margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model + Nova ───────────────────────────────────────────────────────────────
try:
    model, model_path = load_model(project_root)
    if model is None:
        st.error("Model not found. Run: python src/train.py")
        st.stop()
        
    # Read IMG_SIZE directly from the loaded model
    try:
        if hasattr(model, 'layers') and len(model.layers) > 0:
            s = model.layers[0].input_shape
            if isinstance(s, list):
                s = s[0]
            IMG_SIZE = (s[1], s[2])
    except Exception:
        pass  # Keep default if not possible
        
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

print(f"[App] Using IMG_SIZE = {IMG_SIZE}")

nova = NovaExplainer()


def pil_to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return b64lib.b64encode(buf.getvalue()).decode()


def call_nova(prompt, system="", image_b64=None):
    if not (nova.aws_available or nova.gemini_available):
        return "[NOTICE] AI services not configured - add API keys to .env file"
    try:
        image_bytes = b64lib.b64decode(image_b64) if image_b64 else None
        response = nova._call(prompt, system, image_bytes)
        return response
    except Exception as e:
        error_str = str(e)
        if "rate limit" in error_str.lower() or "quota" in error_str.lower():
            return f"[RATE LIMITED] API quota exceeded - please try again in a few moments"
        return f"[ERROR] AI service error: {error_str[:150]}"


# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "done": False, 
    "pred": None,
    "patient_data": None,  # NEW: Store patient data dynamically
    "image_uploaded": False,  # NEW: Track if image was uploaded
    "chat": [{"role": "assistant",
              "content": "GradVision system online. Awaiting specimen upload."}],
    "insight": None,
    "batch_done": False,  # NEW: Batch processing flag
    "batch_result": None,  # NEW: Batch analysis results
    "pdf_bytes": None,  # NEW: Store generated PDF
    "pdf_generated": False,  # NEW: Track if PDF was generated
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR — EHR Patient Context ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="ehr-header">
        <div class="gv-wordmark">🔬 GradVision</div>
        <div class="gv-tagline">Clinical Intelligence · v3.1</div>
        <div style="font-family:var(--font-mono);font-size:0.58rem;color:var(--green);margin-top:0.5rem;">✓ SYSTEM ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # Display patient EHR - DYNAMIC based on image upload
    try:
        if st.session_state.patient_data:
            # Patient data exists - show it
            patient_html = get_patient_display_html(st.session_state.patient_data)
            st.markdown(patient_html, unsafe_allow_html=True)
            st.success("✓ Patient context loaded")
        else:
            # No patient data - show placeholder
            st.markdown("""
            <div class="ehr-section">
                <div class="ehr-section-title">EHR · Patient Context</div>
                <div style="text-align:center;padding:1.5rem;color:var(--text-3);">
                    <div style="font-size:2.5rem;margin-bottom:0.5rem;">📋</div>
                    <div style="font-size:0.95rem;font-weight:500;">No specimen uploaded</div>
                    <div style="font-family:var(--font-mono);font-size:0.7rem;margin-top:0.5rem;line-height:1.4;">
                        ⬆ Upload histopathology image above to generate patient context and EHR data
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠ Sidebar error: {str(e)[:50]}")

    # System status with expander
    is_available = nova.aws_available or nova.gemini_available
    dot_nova = "dot-green" if is_available else "dot-red"
    nova_txt = "ONLINE" if is_available else "OFFLINE"
    
    # Add expander for system status
    with st.expander("[SYSTEM STATUS] Click to expand/collapse", expanded=False):
        st.markdown(f"""
        <div class="ehr-section">
            <div class="ehr-section-title">AI Services Status</div>
            <div class="ehr-row">
                <span class="ehr-key">Nova AI (AWS)</span>
                <span class="ehr-val ok">
                    <span class="status-dot {'dot-green' if nova.aws_available else 'dot-red'}"></span>
                    {'ONLINE' if nova.aws_available else 'OFFLINE - Using Demo Mode'}</span>
            </div>
            <div class="ehr-row">
                <span class="ehr-key">Gemini API</span>
                <span class="ehr-val ok">
                    <span class="status-dot {'dot-green' if nova.gemini_available else 'dot-red'}"></span>
                    {'ONLINE (Fallback)' if nova.gemini_available else 'OFFLINE'}</span>
            </div>
            <div class="ehr-row">
                <span class="ehr-key">Overall Status</span>
                <span class="ehr-val ok">
                    <span class="status-dot {dot_nova}"></span>{nova_txt}</span>
            </div>
            <div class="ehr-row">
                <span class="ehr-key">CNN Engine</span>
                <span class="ehr-val ok">
                    <span class="status-dot dot-green"></span>LOADED</span>
            </div>
            <div class="ehr-row">
                <span class="ehr-key">XAI Module</span>
                <span class="ehr-val ok">
                    <span class="status-dot dot-green"></span>GRAD-CAM</span>
            </div>
            <div class="ehr-row">
                <span class="ehr-key">Model</span>
                <span class="ehr-val" style="font-size:0.68rem;">
                    {os.path.basename(model_path)}</span>
                </div>
                <div class="ehr-row">
                    <span class="ehr-key">Session</span>
                    <span class="ehr-val" style="font-size:0.68rem;">{NOW}</span>
                </div>
                <div class="ehr-row">
                    <span class="ehr-key">Analysis Mode</span>
                    <span class="ehr-val" style="color:var(--amber);">
                    {'DEMO (APIs Limited)' if not (nova.aws_available or nova.gemini_available) else 'LIVE'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Threshold
    st.markdown("""
    <div class="ehr-section">
        <div class="ehr-section-title">Diagnostic Config</div>
    </div>""", unsafe_allow_html=True)
    threshold = st.slider("Sensitivity Threshold", 0.50, 0.95, 0.50, 0.01,
                          label_visibility="collapsed")
    st.markdown(
        f"<div style='font-family:var(--font-mono);font-size:0.65rem;"
        f"color:var(--text-3);padding:0 1.2rem 0.5rem;'>"
        f"Threshold: <span style='color:var(--amber);'>{threshold:.2f}</span>"
        f" &nbsp;·&nbsp; Standard: 0.50</div>",
        unsafe_allow_html=True)

    st.markdown("<div style='padding:0.75rem 1.2rem;'>", unsafe_allow_html=True)
    if st.button("↺  Reset Session"):
        st.session_state.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── UNIFIED IMAGE PROCESSING FUNCTION ──────────────────────────────────────
def process_image(uploaded_file, file_name: str, model, img_size=(96, 96), 
                   threshold=0.50, nova_explainer=None) -> dict:
    """
    Unified image processing function for both single and batch modes.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        file_name: Display name of the file
        model: Loaded TensorFlow model
        img_size: Target image size (default 96x96)
        threshold: Classification threshold (default 0.50)
        nova_explainer: NovaExplainer instance for report generation
    
    Returns:
        dict with complete prediction and visualization, or None if processing fails.
        
        Success format:
        {
            "image_name": str,
            "pil_img": PIL.Image,
            "img_array": np.ndarray,
            "raw_pred": float,
            "label": str,
            "confidence": float,
            "heatmap": np.ndarray,
            "figure": PIL.Image,
            "uncertain": bool,
            "report": str,
            "b64": str,
            "is_malignant": bool,
            "status": "success"
        }
    """
    try:
        # Step 1: Load and preprocess image
        try:
            img_array, pil_img = load_and_preprocess_image(uploaded_file, img_size, model)
            if img_array is None or pil_img is None:
                return None
        except Exception as load_err:
            st.warning(f"❌ {file_name}: Image loading failed - {str(load_err)[:50]}")
            return None
        
        # Step 2: Generate safe prediction
        prediction_data = safe_predict(model, img_array, threshold)
        if prediction_data["status"] != "success":
            st.warning(f"❌ {file_name}: Model safe prediction aborted.")
            return None
            
        raw_pred = prediction_data["raw_pred"]
        label = prediction_data["label"]
        confidence = prediction_data["confidence"]
        uncertain = prediction_data["uncertain"]
        
        # Step 4: Generate Grad-CAM heatmap
        heatmap = None
        try:
            heatmap = get_gradcam_heatmap(img_array, model)
        except Exception as hm_err:
            st.warning(f"⚠ {file_name}: Heatmap generation skipped")
        
        # Step 4b: Detect tumor region using heatmap
        tumor_result = None
        if heatmap is not None:
            try:
                tumor_result = detect_tumor_region(pil_img, heatmap, threshold=0.6)
            except Exception as tumor_err:
                st.warning(f"⚠ {file_name}: Tumor localization skipped")
                tumor_result = None
        
        # Step 4c: Analyze cell morphology
        morphology_result = None
        try:
            morphology_result = analyze_cell_morphology(pil_img)
        except Exception as morph_err:
            st.warning(f"⚠ {file_name}: Cell morphology analysis skipped")
            morphology_result = None
        
        # Step 5: Create visualization figure
        figure = None
        if heatmap is not None:
            try:
                figure = create_gradcam_figure(
                    heatmap, pil_img, label, confidence, uncertain, img_size)
            except Exception as fig_err:
                st.warning(f"⚠ {file_name}: Visualization creation skipped")
        
        # Step 6: Generate clinical report
        report = None
        if nova_explainer is not None:
            try:
                report = nova_explainer.generate_explanation(
                    pil_img, label, confidence, uncertain)
            except Exception as report_err:
                report = "[Report generation unavailable]"
        else:
            report = "[Report generation unavailable]"
        
        # Ensure report is string
        if report is None or not isinstance(report, str):
            report = "[Report generation unavailable]"
        
        # Step 7a: Calculate Severity
        hm_max = heatmap.max() if heatmap is not None else 0.0
        tumor_area = tumor_result["largest_area"] if tumor_result and tumor_result.get("largest_area") else 0
        severity = estimate_severity(raw_pred, tumor_area, hm_max)
        
        # Step 7b: Return complete prediction dictionary
        return {
            "image_name": file_name,
            "pil_img": pil_img,
            "img_array": img_array,
            "raw_pred": raw_pred,
            "label": label,
            "confidence": confidence,
            "heatmap": heatmap,
            "figure": figure,
            "tumor_result": tumor_result,
            "morphology_result": morphology_result,
            "uncertain": uncertain,
            "severity": severity,
            "report": str(report) if report else "[No report available]",
            "b64": pil_to_b64(pil_img) if pil_img else None,
            "is_malignant": label == "MALIGNANT",
            "status": "success"
        }
    
    except Exception as outer_err:
        # Catch-all for unexpected errors
        st.warning(f"❌ {file_name}: Processing failed - {str(outer_err)[:50]}")
        return None



# ── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div class="ehr-header" style="margin-bottom:1rem;">
        <div class="gv-wordmark">🔬 GradVision</div>
        <div class="gv-tagline">Clinical Intelligence · v4.0</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    nav_selection = st.radio(
        "Navigation", 
        ["Upload Images", "Model Information", "Batch Results", "Diagnostics", "Support"],
        label_visibility="collapsed"
    )

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:1.8rem 0 1.4rem;border-bottom:1px solid var(--border);
     margin-bottom:1.8rem;">
    <div>
        <div class="hero-title" style="font-size:2.2rem;">GradVision – AI Breast Cancer Detection System</div>
        <div class="hero-sub" style="font-size:0.8rem; letter-spacing:1px;">AI-powered histopathology diagnostic assistant.</div>
    </div>
    <div style="text-align:right;font-family:var(--font-mono);font-size:0.65rem;
         color:var(--text-3);line-height:1.8;">
        <div><span style="color:var(--amber);">SRG-2024-1101</span></div>
        <div>{NOW}</div>
        <div style="color:var(--green);">
            <span class="status-dot dot-green"></span>SYSTEM ACTIVE</div>
    </div>
</div>
''', unsafe_allow_html=True)


# ── PAGE CONTENT ───────────────────────────────────────────────────────────────

if nav_selection == "Upload Images":
    st.container()
    st.markdown("### Patient Diagnostics Dashboard")
    
    top_c1, top_c2, top_c3 = st.columns(3, gap="large")
    
    with top_c1:
        st.markdown('<div class="section-label">📥 Upload Image</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload image",
            type=["jpg","jpeg","png","tif"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
    with top_c2:
        st.markdown('<div class="section-label">🔖 Case ID</div>', unsafe_allow_html=True)
        if uploaded_files:
            st.info("PT-2024-08863")
        else:
            st.markdown("<div style='padding:1rem;border:1px dashed var(--border-hi);border-radius:8px;text-align:center;color:var(--text-3);'>No Case Active</div>", unsafe_allow_html=True)
            
    with top_c3:
        st.markdown('<div class="section-label">⚙ Model Status</div>', unsafe_allow_html=True)
        is_available = nova.aws_available or nova.gemini_available
        nova_status = "ONLINE" if is_available else "OFFLINE"
        
        st.markdown(f'''
        <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--green);">
            <div style="font-size:0.85rem;color:var(--text-1);"><strong>CNN Engine:</strong> LOADED</div>
            <div style="font-size:0.85rem;color:var(--text-1);margin-top:0.3rem;"><strong>Threshold:</strong> {threshold}</div>
            <div style="font-size:0.85rem;color:var(--text-1);margin-top:0.3rem;"><strong>Nova AI:</strong> {nova_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    
    if uploaded_files:
        is_batch_mode = len(uploaded_files) > 1
        
        if is_batch_mode:
            st.warning(f"Batch mode detected: {len(uploaded_files)} images. Please click analyze and view results in the 'Batch Results' tab.")
            if st.button("▶ ANALYZE BATCH", use_container_width=True):
                with st.spinner(f"Analyzing {len(uploaded_files)} specimens…"):
                    analyzer = BatchAnalyzer(model, IMG_SIZE, threshold)
                    batch_result = analyzer.analyze_batch(uploaded_files)
                    st.session_state.batch_result = batch_result
                    st.session_state.batch_done = True
                    st.session_state.done = False
                    st.session_state.pred = None
                    st.rerun()
                    
        else:
            uploaded_file = uploaded_files[0]
            if st.button("▶ START DIAGNOSTIC SCAN", use_container_width=True):
                with st.spinner("Processing specimen…"):
                    result = process_image(uploaded_file, uploaded_file.name, model, IMG_SIZE, threshold, nova)
                    if result is not None and isinstance(result, dict):
                        st.session_state.pred = result
                        st.session_state.done = True
                        st.session_state.batch_done = False
                    else:
                        st.warning("⚠️ Could not process image.")
                        st.session_state.done = False

    if st.session_state.done and st.session_state.pred:
        r = st.session_state.pred
        
        st.markdown("---")
        
        # PREDICTION RESULT CARDS
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            is_malignant = r.get("is_malignant", False)
            pred_label = "MALIGNANT" if is_malignant else "BENIGN"
            pred_color = "var(--red)" if is_malignant else "var(--green)"
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">PREDICTION</div>
                <div style="font-size:2rem;font-weight:900;color:{pred_color};">{pred_label}</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with rc2:
            conf_val = r.get("confidence", 0) * 100
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">CONFIDENCE</div>
                <div style="font-size:2rem;font-weight:900;color:var(--text-1);">{conf_val:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with rc3:
            severity = r.get("severity", "LOW RISK")
            sev_color = "🔴" if severity == "HIGH RISK" else "🟡" if severity == "MEDIUM RISK" else "🟢"
            txt_color = "var(--red)" if severity == "HIGH RISK" else "var(--amber)" if severity == "MEDIUM RISK" else "var(--green)"
            st.markdown(f'''
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:2px;margin-bottom:0.5rem;">SEVERITY LEVEL</div>
                <div style="font-size:2rem;font-weight:900;color:{txt_color};">{sev_color} {severity}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        
        # IMAGE VISUALIZATION PANEL (3 COLUMNS)
        st.markdown('<div class="section-label">👁 Image Visualization Panel</div>', unsafe_allow_html=True)
        img1, img2, img3 = st.columns(3)
        
        with img1:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">Original Image</div>', unsafe_allow_html=True)
            st.image(r.get('pil_img'), use_container_width=True)
            
        with img2:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">GradCAM Heatmap</div>', unsafe_allow_html=True)
            if r.get('figure'):
                st.image(r.get('figure'), use_container_width=True)
                
        with img3:
            st.markdown('<div style="text-align:center;font-weight:bold;margin-bottom:0.5rem;">Tumor Localization</div>', unsafe_allow_html=True)
            tumor_res = r.get('tumor_result', {})
            if tumor_res and tumor_res.get('tumor_box_image'):
                st.image(tumor_res['tumor_box_image'], use_container_width=True)
            else:
                st.info("No clearly defined tumor bounds isolated.")

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        
        # MORPHOLOGY SECTION
        with st.expander("🔬 Morphology Analysis", expanded=True):
            morph = r.get('morphology_result', {})
            if morph:
                m1, m2, m3 = st.columns(3)
                m1.metric("Texture Irregularity", morph.get('texture_irregularity', 0))
                m2.metric("Nucleus Density", morph.get('cell_density', 0))
                m3.metric("Tissue Uniformity", morph.get('tissue_uniformity', 0))
            else:
                st.warning("Morphology data unavailable.")
                
        # REPORT SECTION
        with st.expander("📄 AI Diagnostic Report", expanded=True):
            st.markdown(f'<div class="nova-block-text">{r.get("report", "No report available.")}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            
            if st.button("Generate & Download PDF", key="pdf_dl"):
                with st.spinner("Generating PDF..."):
                    report_data = {
                        'prediction': r.get('label', 'Unknown'),
                        'confidence': r.get('confidence', 0.0),
                        'image_name': r.get('image_name', 'Specimen'),
                        'cell_count': morph.get('cell_count', 0) if morph else 0,
                        'cell_density': morph.get('cell_density', 0.0) if morph else 0.0,
                        'irregular_nuclei_ratio': morph.get('irregular_nuclei_ratio', 0.0) if morph else 0.0,
                        'cluster_count': morph.get('cluster_count', 0) if morph else 0,
                        'largest_cluster': morph.get('largest_cluster', 0) if morph else 0,
                        'suspicion_level': morph.get('suspicion_level', 'Unknown') if morph else 'Unknown'
                    }
                    pdf_bytes, json_str, status_msg = generate_medical_report(
                        report_data,
                        heatmap_img=r.get('figure'),
                        tumor_img=r.get('tumor_result', {}).get('tumor_box_image') if r.get('tumor_result') else None
                    )
                    st.session_state.pdf_bytes = pdf_bytes
                    st.success("Report generated!")
            
            if st.session_state.get('pdf_bytes'):
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"AI_Pathology_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

elif nav_selection == "Batch Results":
    st.container()
    st.markdown("### Batch Results Section")
    
    if st.session_state.batch_done and st.session_state.batch_result:
        br = st.session_state.batch_result
        
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Images", br['total_count'])
        col_b2.metric("Malignant Count", br['malignant_count'])
        col_b3.metric("Benign Count", br['benign_count'])
        col_b4.metric("Consensus Result", br['final_diagnosis'])
        
        # Simple Chart
        st.markdown("#### Malignant vs Benign Ratio")
        chart_data = {"Malignant": br['malignant_count'], "Benign": br['benign_count']}
        st.bar_chart(chart_data)
        
        st.markdown("#### Detailed Results")
        for i, res in enumerate(br['results']):
            if res['status'] == 'success':
                st.write(f"**{res['file_name']}** - {res.get('label', 'UNKNOWN')} ({res.get('confidence', 0)*100:.1f}%)")

    else:
        st.info("No batch results available. Please upload multiple images in the 'Upload Images' section.")

elif nav_selection == "Model Information":
    st.container()
    st.markdown("### Model Information")
    st.write("**Model Architecture:** MobileNetV2 with Transfer Learning")
    st.write("**XAI Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)")
    st.write("**Input Size:** 96×96 pixels")
    st.write("**Analysis Type:** Binary Classification (Benign vs Malignant)")
    st.code("Current Threshold: " + str(threshold))

elif nav_selection == "Diagnostics":
    st.container()
    st.markdown("### Advanced Diagnostics")
    st.info("Use the primary 'Upload Images' tab to view individual prediction panels.")
    if st.session_state.done and "wsi_result" in st.session_state.pred:
        wsi = st.session_state.pred["wsi_result"]
        st.write(f"WSI Total Patches: **{wsi['total_patches']}**")
        st.write(f"WSI Tumor Patches: **{wsi['tumor_patches']}**")
        if wsi.get("heatmap_image"):
            st.image(wsi["heatmap_image"], caption="Whole Slide Scan Heatmap")

elif nav_selection == "Support":
    st.container()
    st.markdown("### Clinician Support Desk")
    st.write("For automated analysis assistance, view the AI Diagnostic Report in the Upload page.")
    chat_box = st.container(height=400)
    for msg in st.session_state.chat:
        with chat_box.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if q := st.chat_input("Query Nova..."):
        st.session_state.chat.append({"role": "user", "content": q})
        st.rerun()

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='gv-footer'>"
    "For Research &amp; Demonstration Only &nbsp;·&nbsp; "
    "Not a Clinical Diagnostic Tool &nbsp;·&nbsp; "
    "Consult a Qualified Pathologist"
    "</div>",
    unsafe_allow_html=True)
