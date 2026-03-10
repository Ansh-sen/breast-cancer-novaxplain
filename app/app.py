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

from src.gradcam_utils    import (load_and_preprocess_image,
                                   get_gradcam_heatmap,
                                   create_gradcam_figure,
                                   detect_tumor_region)
from src.morphology_analysis import analyze_cell_morphology
from src.report_generator import generate_medical_report
from src.batch_processing import BatchAnalyzer, validate_batch_files
from src.nova_explanation import NovaExplainer
from src.patient_utils import generate_patient_data, get_patient_display_html
from src.image_validator import validate_histopathology_image

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
  --bg:          #0b0f1a;
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
@st.cache_resource
def load_model():
    paths = [
        os.path.join(project_root, "models", "breast_cancer_model_best.keras"),
        os.path.join(project_root, "models", "breast_cancer_model.keras"),
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        st.error("Model not found. Run: python src/train.py")
        st.stop()
    m = tf.keras.models.load_model(path)
    m(tf.zeros((1, *IMG_SIZE, 3), dtype=tf.float32), training=False)
    return m, path

try:
    model, model_path = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# read IMG_SIZE directly from the loaded model — works for any size
try:
    _s = model.layers[0].input_shape
    IMG_SIZE = (_s[1], _s[2])
except Exception:
    pass  # keep default (96,96)
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
        
        # Step 2: Generate prediction
        try:
            raw_pred = float(model.predict(img_array, verbose=0)[0][0])
        except Exception as pred_err:
            st.warning(f"❌ {file_name}: Model prediction failed - {str(pred_err)[:50]}")
            return None
        
        # Step 3: Calculate classification and confidence
        uncertain = 0.40 < raw_pred < 0.60
        label = "MALIGNANT" if raw_pred > threshold else "BENIGN"
        confidence = raw_pred if label == "MALIGNANT" else 1.0 - raw_pred
        
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
        
        # Step 7: Return complete prediction dictionary
        return {
            "image_name": file_name,
            "pil_img": pil_img,
            "img_array": img_array,
            "raw_pred": float(raw_pred) if raw_pred is not None else 0.0,
            "label": str(label) if label else "UNKNOWN",
            "confidence": float(confidence) if confidence is not None else 0.0,
            "heatmap": heatmap,
            "figure": figure,
            "tumor_result": tumor_result,  # Tumor localization data
            "morphology_result": morphology_result,  # Cell morphology analysis data
            "uncertain": bool(uncertain) if uncertain is not None else False,
            "report": str(report) if report else "[No report available]",
            "b64": pil_to_b64(pil_img) if pil_img else None,
            "is_malignant": label == "MALIGNANT" if label else False,
            "status": "success"
        }
    
    except Exception as outer_err:
        # Catch-all for unexpected errors
        st.warning(f"❌ {file_name}: Processing failed - {str(outer_err)[:50]}")
        return None


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:1.8rem 0 1.4rem;border-bottom:1px solid var(--border);
     margin-bottom:1.8rem;">
    <div>
        <div class="hero-title">Diagnostic Intelligence</div>
        <div class="hero-sub">Histopathology · XAI · Amazon Nova · Unit-01</div>
    </div>
    <div style="text-align:right;font-family:var(--font-mono);font-size:0.65rem;
         color:var(--text-3);line-height:1.8;">
        <div><span style="color:var(--amber);">SRG-2024-1101</span></div>
        <div>{NOW}</div>
        <div style="color:var(--green);">
            <span class="status-dot dot-green"></span>SYSTEM ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── UNIFIED UPLOAD SECTION ─────────────────────────────────────────────────────
# Supports both single image and batch processing (1 or multiple images)
col_up, col_res = st.columns([1, 1.4], gap="large")

with col_up:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📥 Specimen Import</div>',
                unsafe_allow_html=True)

    # Unified file uploader - accepts 1 to multiple images
    uploaded_files = st.file_uploader(
        "Drop histopathology images (1 or multiple)",
        type=["jpg","jpeg","png","tif"],
        accept_multiple_files=True,
        label_visibility="collapsed")

    # Determine processing mode based on number of files
    is_batch_mode = len(uploaded_files) > 1
    
    if uploaded_files:
        try:
            # ── BATCH MODE: Multiple images ─────────────────────────────────
            if is_batch_mode:
                # Validate batch (2-10 images)
                is_valid, msg = validate_batch_files(uploaded_files)
                if not is_valid:
                    st.error(f"⚠ {msg}")
                else:
                    st.success(f"✓ {len(uploaded_files)} images ready for analysis")
                    if st.button("▶  ANALYZE BATCH", key="analyze_batch_unified"):
                        with st.spinner(f"Analyzing {len(uploaded_files)} specimens…"):
                            analyzer = BatchAnalyzer(model, IMG_SIZE, threshold)
                            batch_result = analyzer.analyze_batch(uploaded_files)
                            st.session_state.batch_result = batch_result
                            st.session_state.batch_done = True
                            st.session_state.done = False
                            st.session_state.pred = None
                            st.session_state.pdf_generated = False
                            st.session_state.pdf_bytes = None
                            st.rerun()
            
            # ── SINGLE MODE: Single image ───────────────────────────────────
            else:
                uploaded_file = uploaded_files[0]
                
                # Load image for validation
                try:
                    img_array_temp, pil_img_temp = load_and_preprocess_image(uploaded_file, IMG_SIZE)
                except:
                    img_array_temp = None
                    pil_img_temp = None
                
                # Validate that image is histopathology
                validation_result = validate_histopathology_image(pil_img_temp) if pil_img_temp else {'is_valid': False, 'message': 'Failed to load image'}
                
                # Check if this is a NEW image
                is_new_image = st.session_state.image_uploaded != uploaded_file.name
                
                if not validation_result['is_valid']:
                    st.error(validation_result['message'])
                    st.session_state.patient_data = None
                    st.session_state.done = False
                    st.session_state.pred = None
                    st.session_state.pdf_generated = False
                    st.session_state.pdf_bytes = None
                
                elif validation_result.get('is_warning', False):
                    st.warning(validation_result['message'])
                    
                    # Generate patient data
                    if st.session_state.patient_data is None or is_new_image:
                        if is_new_image:
                            st.session_state.done = False
                            st.session_state.pred = None
                            st.session_state.insight = None
                            st.session_state.pdf_generated = False
                            st.session_state.pdf_bytes = None
                            st.session_state.chat = [{"role": "assistant", "content": "GradVision system online. Awaiting specimen upload."}]
                        
                        st.session_state.patient_data = generate_patient_data(uploaded_file.name)
                        st.session_state.image_uploaded = uploaded_file.name
                        st.info("✓ Patient context generated (⚠️ Low confidence mode)")
                    
                    # Show image
                    st.markdown(f"""
                    <div class="meta-strip">
                        <div class="meta-item">ID: <span>PT-2024-08863</span></div>
                        <div class="meta-item">Stain: <span>H&E</span></div>
                        <div class="meta-item">Mag: <span>40×</span></div>
                        <div class="meta-item">Res: <span>96×96</span></div>
                        <div class="meta-item">Model: <span>MobileNetV2</span></div>
                        <div class="meta-live">● LIVE</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="img-frame scan-wrap">', unsafe_allow_html=True)
                    st.image(pil_img_temp, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                    
                    if st.button("▶  START DIAGNOSTIC SCAN", key="scan_warning"):
                        with st.spinner("Processing specimen…"):
                            result = process_image(uploaded_file, uploaded_file.name, model, 
                                                 IMG_SIZE, threshold, nova)
                            if result is not None and isinstance(result, dict):
                                st.session_state.pred = result
                                st.session_state.done = True
                                st.session_state.batch_done = False
                                st.session_state.batch_result = None
                                st.session_state.insight = None
                                st.session_state.chat = [{
                                    "role": "assistant",
                                    "content": (
                                        f"Analysis complete. Specimen classified as **{result.get('label', 'UNKNOWN')}** "
                                        f"({result.get('confidence', 0)*100:.1f}% confidence)."
                                        + (" ⚠️ Borderline — expert review recommended."
                                           if result.get('uncertain', False) else "")
                                    )
                                }]
                                st.rerun()
                            else:
                                st.warning("⚠️ Could not process image. Try a different specimen.")
                                st.session_state.done = False
                                st.session_state.pred = None
                
                else:
                    # Valid image
                    if st.session_state.patient_data is None or is_new_image:
                        if is_new_image:
                            st.session_state.done = False
                            st.session_state.pred = None
                            st.session_state.insight = None
                            st.session_state.pdf_generated = False
                            st.session_state.pdf_bytes = None
                            st.session_state.chat = [{"role": "assistant", "content": "GradVision system online. Awaiting specimen upload."}]
                        
                        st.session_state.patient_data = generate_patient_data(uploaded_file.name)
                        st.session_state.image_uploaded = uploaded_file.name
                        st.success("✓ Patient context loaded")
                    
                    # Show image
                    st.markdown(f"""
                    <div class="meta-strip">
                        <div class="meta-item">ID: <span>PT-2024-08863</span></div>
                        <div class="meta-item">Stain: <span>H&E</span></div>
                        <div class="meta-item">Mag: <span>40×</span></div>
                        <div class="meta-item">Res: <span>96×96</span></div>
                        <div class="meta-item">Model: <span>MobileNetV2</span></div>
                        <div class="meta-live">● LIVE</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="img-frame scan-wrap">', unsafe_allow_html=True)
                    st.image(pil_img_temp, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

                    if st.button("▶  START DIAGNOSTIC SCAN"):
                        with st.spinner("Processing specimen…"):
                            result = process_image(uploaded_file, uploaded_file.name, model, 
                                                 IMG_SIZE, threshold, nova)
                            if result is not None and isinstance(result, dict):
                                st.session_state.pred = result
                                st.session_state.done = True
                                st.session_state.batch_done = False
                                st.session_state.batch_result = None
                                st.session_state.insight = None
                                st.session_state.chat = [{
                                    "role": "assistant",
                                    "content": (
                                        f"Analysis complete. Specimen classified as **{result.get('label', 'UNKNOWN')}** "
                                        f"({result.get('confidence', 0)*100:.1f}% confidence)."
                                        + (" ⚠️ Borderline — expert review recommended."
                                           if result.get('uncertain', False) else "")
                                    )
                                }]
                                st.rerun()
                            else:
                                st.warning("⚠️ Could not process image. Try a different specimen.")
                                st.session_state.done = False
                                st.session_state.pred = None
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Image processing failed: {str(e)[:100]}")
            st.warning("⚠️ Please try uploading a different image.")
            # Don't stop execution - let other sections render
            st.session_state.patient_data = None
            st.session_state.done = False
            st.session_state.pred = None
            st.session_state.pdf_generated = False
            st.session_state.pdf_bytes = None
            st.session_state.insight = None
            st.session_state.chat = [{"role": "assistant", "content": "GradVision system online. Awaiting specimen upload."}]
    else:
        # No images uploaded - clear all state
        st.session_state.patient_data = None
        st.session_state.done = False
        st.session_state.pred = None
        st.session_state.batch_done = False
        st.session_state.batch_result = None
        st.session_state.pdf_generated = False
        st.session_state.pdf_bytes = None
        st.session_state.insight = None
        st.session_state.chat = [{"role": "assistant", "content": "GradVision system online. Awaiting specimen upload."}]
        st.markdown("""
        <div style="height:300px;display:flex;flex-direction:column;
             justify-content:center;align-items:center;
             border:1.5px dashed var(--border-hi);border-radius:var(--radius-md);
             background:var(--surface-2);">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;opacity:0.3;">🔬</div>
            <div style="font-weight:600;color:var(--text-3);font-size:0.9rem;">
                Drop specimen image(s)</div>
            <div style="font-family:var(--font-mono);font-size:0.62rem;
                 color:var(--text-3);margin-top:0.3rem;">
                Single image or batch (2-10) · JPG · PNG · TIF</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


with col_res:
    if not st.session_state.done and not st.session_state.batch_done:
        st.markdown("""
        <div class="cv-card" style="min-height:400px;display:flex;
             flex-direction:column;justify-content:center;align-items:center;">
            <div style="font-size:3rem;opacity:0.12;margin-bottom:1rem;">📋</div>
            <div style="font-weight:600;color:var(--text-3);">
                Awaiting Specimen Input</div>
            <div style="font-family:var(--font-mono);font-size:0.65rem;
                 color:var(--text-3);margin-top:0.4rem;">
                Pathology Unit-01 · READY</div>
        </div>""", unsafe_allow_html=True)
    else:
        r = st.session_state.pred
        # Safety validation: ensure pred is not None and has required keys
        if r is None or not isinstance(r, dict):
            st.error("Error: Invalid prediction object. Please try uploading again.")
        else:
            # Safe value extraction with defaults
            confidence = r.get("confidence", 0)
            is_malignant = r.get("is_malignant", False)
            uncertain = r.get("uncertain", False)
            report = r.get("report", "No report available")
            
            if confidence is None:
                confidence = 0
            
            pct = confidence * 100
            col = (var := lambda n: f"var(--{n})")(
                "red" if is_malignant else
                "amber" if uncertain else "green")
            banner_cls = ("banner-mal" if is_malignant else
                          "banner-unc" if uncertain else "banner-ben")
            bar_cls    = ("bar-mal" if is_malignant else
                          "bar-unc" if uncertain else "bar-ben")
            badge      = ("<span class='unc-badge'>⚠ Borderline · Expert Review</span>"
                          if uncertain else "")

            st.markdown(f"""
            <div class="cv-card">
                <div class="section-label">Diagnostic Result</div>

                <div class="nova-block">
                    <div class="nova-block-label">Nova Clinical Synthesis</div>
                    <div class="nova-block-text">{report}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── BATCH DETAILED RESULTS ────────────────────────────────────────────────────
if st.session_state.batch_done and st.session_state.batch_result:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="border-top:2px solid var(--border);padding:1.6rem 0 0;"></div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="padding:1.8rem 0 1.4rem;">
        <div style="font-size:1.6rem;font-weight:900;color:var(--text-1);">Individual Results</div>
        <div style="font-family:var(--font-mono);font-size:0.65rem;
             color:var(--text-3);letter-spacing:2px;text-transform:uppercase;margin-top:0.3rem;">
            Specimen-by-specimen predictions and heatmaps</div>
    </div>
    """, unsafe_allow_html=True)
    
    br = st.session_state.batch_result
    
    # Create tabs for each image
    if br['success_count'] > 0:
        # Create dynamic tabs - one per successful image
        tab_list = []
        for i, result in enumerate(br['results']):
            if result['status'] == 'success':
                tab_list.append(f"🖼 {i+1}. {result['file_name'][:20]}")
        
        if tab_list:
            result_tabs = st.tabs(tab_list)
            
            tab_idx = 0
            for i, result in enumerate(br['results']):
                # Safety validation: ensure result is not None and has required keys
                if (result is not None and isinstance(result, dict) and 
                    'status' in result and result['status'] == 'success' and 
                    tab_idx < len(result_tabs) and 
                    'confidence' in result and result['confidence'] is not None):
                    
                    with result_tabs[tab_idx]:
                        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
                        
                        # Image info header
                        pred_color = "var(--red)" if result.get('label', 'UNKNOWN') == "MALIGNANT" else "var(--green)"
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                            <div>
                                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:1px;">SPECIMEN</div>
                                <div style="font-size:1.2rem;font-weight:700;color:var(--text-1);">{result.get('file_name', 'Unknown')}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-3);letter-spacing:1px;">PREDICTION</div>
                                <div style="font-size:1.2rem;font-weight:700;color:{pred_color};">{result.get('label', 'UNKNOWN')}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Image preview and metadata
                        col_img, col_meta = st.columns([1, 0.7])
                        
                        with col_img:
                            st.markdown('<div style="border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;">',
                                       unsafe_allow_html=True)
                            if 'pil_img' in result:
                                st.image(result['pil_img'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_meta:
                            conf_pct = (result.get('confidence', 0) * 100) if result.get('confidence') is not None else 0
                            raw_score = result.get('raw_pred', 0.0)
                            is_uncertain = result.get('uncertain', False)
                            st.markdown(f"""
                            <div style="padding:0.5rem;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);">
                                <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-3);margin-bottom:0.5rem;letter-spacing:1px;">METRICS</div>
                                <div style="font-size:0.95rem;line-height:2;color:var(--text-1);">
                                    <div><span style="color:var(--text-3);">Confidence:</span> <strong>{conf_pct:.1f}%</strong></div>
                                    <div><span style="color:var(--text-3);">Raw Score:</span> <strong>{raw_score:.4f}</strong></div>
                                    <div><span style="color:var(--text-3);">Status:</span> <strong>{'Borderline' if is_uncertain else 'Confident'}</strong></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Grad-CAM heatmap
                        st.markdown('<div style="margin-top:1rem;border:1px solid var(--border);border-radius:var(--radius-sm);overflow:hidden;">',
                                   unsafe_allow_html=True)
                        if 'figure' in result:
                            st.image(result['figure'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        tab_idx += 1
    
    # Show errors if any
    if br['error_count'] > 0:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        with st.expander(f"⚠ {br['error_count']} Image(s) Failed to Process", expanded=False):
            for result in br['results']:
                # Safety validation before accessing error result
                if (result is not None and isinstance(result, dict) and 
                    'status' in result and result['status'] == 'error'):
                    file_name = result.get('file_name', 'Unknown')
                    error_msg = result.get('error_msg', 'Unknown error')
                    st.error(f"❌ **{file_name}**: {error_msg[:100]}")


# ── TABBED LOWER SECTION ───────────────────────────────────────────────────────
if st.session_state.done:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📡  Primary Scan · Grad-CAM",
        "🔬  Morphological Deep-Dive",
        "💬  Support Desk",
        "📊  Diagnostics Info",
    ])

    # Tab 1 — Grad-CAM
    with tab1:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Neural ROI Heatmap · Grad-CAM XAI</div>',
                    unsafe_allow_html=True)
        # Safe access to pred figure
        if (st.session_state.pred is not None and 
            isinstance(st.session_state.pred, dict) and 
            'figure' in st.session_state.pred):
            st.image(st.session_state.pred["figure"], use_container_width=True)
        st.markdown(f"""
        <div style="display:flex;gap:1.5rem;margin-top:0.75rem;
             font-family:var(--font-mono);font-size:0.6rem;color:var(--text-3);">
            <span>Panel 1 · Original Specimen</span>
            <span>·</span>
            <span>Panel 2 · Predicted Class</span>
            <span>·</span>
            <span>Panel 3 · HOT Heatmap for IDC (○ = peak ROI)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TUMOR LOCALIZATION SECTION ──────────────────────────────────────────────
    # Display three-panel tumor localization visualization
    if st.session_state.done and st.session_state.pred:
        r = st.session_state.pred
        tumor_result = r.get('tumor_result')
        
        if tumor_result and isinstance(tumor_result, dict):
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">AI Tumor Localization · Region Detection</div>',
                        unsafe_allow_html=True)
            
            if tumor_result.get('has_tumor', False):
                # Display three images: Original, Heatmap, Localization
                col_orig, col_heat, col_tumor = st.columns(3)
                
                with col_orig:
                    st.markdown('<div style="text-align:center;"><strong>Original Image</strong></div>',
                               unsafe_allow_html=True)
                    st.image(r.get('pil_img'), use_container_width=True)
                
                with col_heat:
                    st.markdown('<div style="text-align:center;"><strong>Grad-CAM Heatmap</strong></div>',
                               unsafe_allow_html=True)
                    if tumor_result.get('heatmap_image'):
                        st.image(tumor_result['heatmap_image'], use_container_width=True)
                
                with col_tumor:
                    st.markdown('<div style="text-align:center;"><strong>Tumor Localization</strong></div>',
                               unsafe_allow_html=True)
                    if tumor_result.get('tumor_box_image'):
                        st.image(tumor_result['tumor_box_image'], use_container_width=True)
                
                # Display tumor coordinates
                coords = tumor_result.get('tumor_coordinates')
                if coords:
                    x, y, w, h = coords
                    st.markdown(f"""
                    <div style="margin-top:1rem;padding:0.75rem;background:var(--surface-2);border-left:3px solid var(--red);border-radius:var(--radius-sm);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--red);margin-bottom:0.3rem;letter-spacing:1px;">TUMOR REGION COORDINATES</div>
                        <div style="font-size:0.9rem;color:var(--text-1);line-height:1.8;">
                            <strong>Position:</strong> ({x}, {y})<br/>
                            <strong>Dimensions:</strong> {w} × {h} pixels<br/>
                            <strong>Area:</strong> {w * h} square pixels
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding:1rem;background:var(--amber-dim);border-left:3px solid var(--amber);border-radius:var(--radius-sm);">
                    <strong style="color:var(--amber);">⚠ No Clear Tumor Region Detected</strong>
                    <div style="font-size:0.85rem;color:var(--text-2);margin-top:0.5rem;">
                    The Grad-CAM heatmap did not identify a distinct tumor region using current thresholds.
                    This may indicate the tissue is benign or the malignancy is diffuse.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ── CELL MORPHOLOGY ANALYSIS SECTION ────────────────────────────────────────
    # Display cell morphology features and detected nuclei
    if st.session_state.done and st.session_state.pred:
        r = st.session_state.pred
        morphology_result = r.get('morphology_result')
        
        if morphology_result and isinstance(morphology_result, dict):
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Cell Morphology Analyzer · Biological Evidence</div>',
                        unsafe_allow_html=True)
            
            if morphology_result.get('cell_count', 0) > 0:
                # Display metrics in a grid layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--cyan);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--cyan);letter-spacing:1px;margin-bottom:0.3rem;">CELLS DETECTED</div>
                        <div style="font-size:1.8rem;font-weight:600;color:var(--text-1);">{morphology_result.get('cell_count', 0)}</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">nuclei identified</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--magenta);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--magenta);letter-spacing:1px;margin-bottom:0.3rem;">CELL DENSITY</div>
                        <div style="font-size:1.8rem;font-weight:600;color:var(--text-1);">{morphology_result.get('cell_density', 0)}</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">cells/million pixels</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    irregular_pct = morphology_result.get('irregular_nuclei_ratio', 0) * 100
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--orange);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--orange);letter-spacing:1px;margin-bottom:0.3rem;">IRREGULAR NUCLEI</div>
                        <div style="font-size:1.8rem;font-weight:600;color:var(--text-1);">{irregular_pct:.1f}%</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">circularity &lt; 0.7</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Second row of metrics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--violet);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--violet);letter-spacing:1px;margin-bottom:0.3rem;">CLUSTER COUNT</div>
                        <div style="font-size:1.8rem;font-weight:600;color:var(--text-1);">{morphology_result.get('cluster_count', 0)}</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">abnormal groupings</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid var(--indigo);">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--indigo);letter-spacing:1px;margin-bottom:0.3rem;">LARGEST CLUSTER</div>
                        <div style="font-size:1.8rem;font-weight:600;color:var(--text-1);">{morphology_result.get('largest_cluster', 0)}</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">cells per group</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    suspicion = morphology_result.get('suspicion_level', 'Unknown')
                    # Color code by suspicion level
                    if suspicion == 'High':
                        sus_color = 'var(--red)'
                    elif suspicion == 'Moderate':
                        sus_color = 'var(--amber)'
                    else:
                        sus_color = 'var(--green)'
                    
                    st.markdown(f"""
                    <div style="padding:0.75rem;background:var(--surface-2);border-radius:var(--radius-sm);border-left:3px solid {sus_color};">
                        <div style="font-family:var(--font-mono);font-size:0.65rem;color:{sus_color};letter-spacing:1px;margin-bottom:0.3rem;">MORPHOLOGY SUSPICION</div>
                        <div style="font-size:1.8rem;font-weight:600;color:{sus_color};">{suspicion}</div>
                        <div style="font-size:0.75rem;color:var(--text-3);margin-top:0.3rem;">biological evidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display annotated morphology image
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown('<div style="text-align:center;"><strong>Detected Cell Nuclei · Annotated Image</strong></div>',
                           unsafe_allow_html=True)
                if morphology_result.get('annotated_image'):
                    st.image(morphology_result['annotated_image'], use_container_width=True)
                    st.markdown("""
                    <div style="font-size:0.75rem;color:var(--text-3);text-align:center;margin-top:0.5rem;margin-bottom:0.5rem;">
                    <span style="color:var(--green);">● Green contours</span> = regular nuclei · 
                    <span style="color:var(--red);">● Red contours</span> = irregular nuclei
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div style="padding:1rem;background:var(--amber-dim);border-left:3px solid var(--amber);border-radius:var(--radius-sm);">
                    <strong style="color:var(--amber);">⚠ Cell Analysis Incomplete</strong>
                    <div style="font-size:0.85rem;color:var(--text-2);margin-top:0.5rem;">
                    No nuclei detected in the image. Morphology analysis could not be completed.
                    This may indicate extremely benign tissue or image quality issues.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ── AI PATHOLOGY REPORT SECTION ─────────────────────────────────────────────
    # Generate professional medical report with PDF download
    if st.session_state.done and st.session_state.pred:
        r = st.session_state.pred
        
        # Only show report generation if we have successful prediction
        if r.get('status') == 'success':
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">AI Pathology Report · Clinical Documentation</div>',
                        unsafe_allow_html=True)
            
            st.markdown("""
            <div style="padding:0.75rem;background:var(--surface-2);border-left:3px solid var(--green);border-radius:var(--radius-sm);margin-bottom:1rem;">
                <div style="font-size:0.85rem;color:var(--text-1);line-height:1.6;">
                Generate a professional pathology-style medical report summarizing all analysis results.
                The report includes AI prediction, morphology findings, and clinical recommendations.
                <strong>Important:</strong> This is an AI-assisted report for clinical decision support only.
                Final diagnosis must be confirmed by a qualified pathologist.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Report generation button
            col_gen, col_space = st.columns([1, 3])
            
            with col_gen:
                if st.button("📄 Generate Medical Report", use_container_width=True, key="gen_report_btn"):
                    with st.spinner("🔄 Generating professional report..."):
                        # Prepare data for report generation
                        report_data = {
                            'prediction': r.get('label', 'Unknown'),
                            'confidence': r.get('confidence', 0.0),
                            'cell_count': 0,
                            'cell_density': 0.0,
                            'irregular_nuclei_ratio': 0.0,
                            'cluster_count': 0,
                            'largest_cluster': 0,
                            'suspicion_level': 'Unknown',
                            'image_name': r.get('image_name', 'Specimen')
                        }
                        
                        # Add morphology data if available
                        morph = r.get('morphology_result')
                        if morph and isinstance(morph, dict):
                            report_data['cell_count'] = morph.get('cell_count', 0)
                            report_data['cell_density'] = morph.get('cell_density', 0.0)
                            report_data['irregular_nuclei_ratio'] = morph.get('irregular_nuclei_ratio', 0.0)
                            report_data['cluster_count'] = morph.get('cluster_count', 0)
                            report_data['largest_cluster'] = morph.get('largest_cluster', 0)
                            report_data['suspicion_level'] = morph.get('suspicion_level', 'Unknown')
                        
                        # Generate PDF
                        pdf_bytes, status_msg = generate_medical_report(
                            report_data,
                            heatmap_img=r.get('figure'),  # GradCAM visualization
                            tumor_img=r.get('tumor_result', {}).get('tumor_box_image') if r.get('tumor_result') else None
                        )
                        
                        if pdf_bytes:
                            st.session_state.pdf_bytes = pdf_bytes
                            st.session_state.pdf_generated = True
                            st.success(status_msg)
                        else:
                            st.error(status_msg)
            
            # Download button (show only if PDF was generated)
            if st.session_state.get('pdf_generated', False) and st.session_state.get('pdf_bytes'):
                st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
                col_down, col_space2 = st.columns([1, 3])
                
                with col_down:
                    st.download_button(
                        label="⬇️ Download AI Pathology Report (PDF)",
                        data=st.session_state.pdf_bytes,
                        file_name=f"AI_Pathology_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2 — Morphological Deep-Dive
    with tab2:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Morphological Deep-Dive · Nova Vision</div>',
                    unsafe_allow_html=True)

        if st.button("✦  Generate Morphological Breakdown"):
            with st.spinner("Nova Vision analysing cellular architecture…"):
                r = st.session_state.pred
                # Safe extraction of prediction values with defaults
                if r is not None and isinstance(r, dict):
                    label = r.get('label', 'UNKNOWN')
                    confidence = r.get('confidence', 0)
                    raw_pred = r.get('raw_pred', 0.0)
                    b64 = r.get("b64", "")
                    
                    st.session_state.insight = call_nova(
                        f"Specimen classified as {label} "
                        f"({confidence*100:.1f}% confidence, "
                        f"raw malignancy score: {raw_pred:.4f}). "
                        "Provide a structured expert morphological breakdown using these headings: "
                        "1) Cell Density & Arrangement "
                        "2) Nuclear Pleomorphism & Chromatin "
                        "3) Gland/Duct Formation "
                        "4) Mitotic Figures "
                        "5) Stromal Changes. "
                        "Under 200 words. Clinical, precise tone.",
                        system="You are a senior breast pathologist AI assistant. "
                               "Provide rigorous morphological analysis. "
                               "Never make a definitive diagnosis.",
                        image_b64=b64
                    )

        if st.session_state.insight:
            st.markdown(
                f'<div class="morph-output">{st.session_state.insight}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="morph-empty">
                <div class="morph-empty-icon">🔬</div>
                <div class="morph-empty-title">
                    Click above to generate breakdown</div>
                <div class="morph-empty-sub">
                    Nova Vision · Cell nuclei · Chromatin · Glands · Stroma
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3 — Support Desk
    with tab3:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Clinician Support Desk · Nova AI</div>',
                    unsafe_allow_html=True)

        chat_box = st.container(height=430)
        for msg in st.session_state.chat:
            with chat_box.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if q := st.chat_input("Query Nova about this specimen…"):
            st.session_state.chat.append({"role": "user", "content": q})
            r = st.session_state.pred
            # Safe value extraction
            if r is not None and isinstance(r, dict):
                label = r.get('label', 'UNKNOWN')
                confidence = r.get('confidence', 0)
                raw_pred = r.get('raw_pred', 0.0)
                is_uncertain = r.get('uncertain', False)
                
                reply = call_nova(
                    f"Case: {label} | Confidence: {confidence*100:.1f}% | "
                    f"P(malignant): {raw_pred:.4f} | "
                    f"Borderline: {'Yes' if is_uncertain else 'No'}\n\n"
                    f"Clinician question: {q}",
                    system="You are a clinical AI diagnostic assistant supporting "
                           "pathologists. Answer concisely. Never diagnose definitively."
                )
                st.session_state.chat.append({"role": "assistant", "content": reply})
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 4 — Diagnostics Info
    with tab4:
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Diagnostic Information & Model Details</div>',
                    unsafe_allow_html=True)
        
        r = st.session_state.pred
        
        # Safe value extraction with defaults
        if r is not None and isinstance(r, dict):
            label = r.get('label', 'UNKNOWN')
            confidence = r.get('confidence', 0)
            raw_pred = r.get('raw_pred', 0.0)
            is_uncertain = r.get('uncertain', False)
        else:
            label = 'UNKNOWN'
            confidence = 0
            raw_pred = 0.0
            is_uncertain = False
        
        # Confidence and Diagnosis Scores
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;margin-bottom:0.5rem;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--amber);margin-bottom:0.3rem;letter-spacing:2px;">CONFIDENCE SCORE</div>
                <div style="font-size:2.2rem;font-weight:900;color:var(--amber);line-height:1;letter-spacing:-0.02em;">{confidence*100:.1f}%</div>
                <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-3);margin-top:0.3rem;">CNN Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d2:
            is_malignant = r.get('is_malignant', False) if r is not None and isinstance(r, dict) else False
            pred_label = "MALIGNANT" if is_malignant else "BENIGN"
            pred_color = "var(--red)" if is_malignant else "var(--green)"
            st.markdown(f"""
            <div style="background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;margin-bottom:0.5rem;">
                <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--blue);margin-bottom:0.3rem;letter-spacing:2px;">CLASSIFICATION</div>
                <div style="font-size:2rem;font-weight:900;color:{pred_color};line-height:1;letter-spacing:-0.02em;">{pred_label}</div>
                <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-3);margin-top:0.3rem;\">Final Diagnosis</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Details
        st.markdown(f"""
        <div style="margin:1rem 0;padding:1rem;background:var(--blue-dim);border-left:3px solid var(--blue);border-radius:0 var(--radius-sm) var(--radius-sm) 0;">
            <div style="font-family:var(--font-mono);font-size:0.6rem;letter-spacing:2.5px;text-transform:uppercase;color:var(--blue);margin-bottom:0.5rem;\">TECHNICAL DETAILS</div>
            <div style="font-size:0.85rem;line-height:1.8;color:var(--text-2);">
                <strong>Model Architecture:</strong> MobileNetV2 with Transfer Learning<br/>
                <strong>XAI Method:</strong> Grad-CAM (Gradient-weighted Class Activation Mapping)<br/>
                <strong>Input Size:</strong> 96×96 pixels (Histopathology)<br/>
                <strong>Analysis Type:</strong> Binary Classification (Benign vs Malignant)<br/>
                <strong>Confidence Threshold:</strong> {threshold:.2f}<br/>
                <strong>Raw Prediction:</strong> {raw_pred:.4f}<br/>
                <strong>Heatmap Colormap:</strong> JET (Blue→Green→Yellow→Red for activation intensity)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Uncertainty warning
        if r["uncertain"]:
            st.warning("⚠️ BORDERLINE CASE: Confidence score is between 40-60%, indicating a borderline prediction. Expert pathologist review is strongly recommended.")
        else:
            status_msg = "High confidence prediction - consistent with CNN assessment." if r["confidence"] > 0.75 else "Moderate confidence prediction - review recommended."
            st.info(f"✓ {status_msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='gv-footer'>"
    "For Research &amp; Demonstration Only &nbsp;·&nbsp; "
    "Not a Clinical Diagnostic Tool &nbsp;·&nbsp; "
    "Consult a Qualified Pathologist &nbsp;·&nbsp; "
    "Amazon Nova Health Infrastructure · Pathology-Core-01"
    "</div>",
    unsafe_allow_html=True)