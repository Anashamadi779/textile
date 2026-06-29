import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from interpreter import InterpretedResult, interpret
from predictor import FabricPredictor
from preprocessor import preprocess, preprocess_pil
from settings import (
    API_KEY,
    CAPTURE_INTERVAL_SEC,
    CONFIDENCE_THRESHOLD,
    ENDPOINT_ID,
    FRAME_WIDTH,
)

st.set_page_config(
    page_title="Fabric Defect Inspector",
    page_icon="🧵",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    font-family: 'Inter', sans-serif !important;
}
header[data-testid="stHeader"],
[data-testid="stToolbar"]          { background-color: #000000 !important; }
[data-testid="stDecoration"]       { display: none; }
.block-container                   { padding-top: 1.5rem !important; }

@keyframes fadeInUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes blink {
    0%,100% { opacity:1; }
    50%     { opacity:0.25; }
}
@keyframes glow-green {
    0%,100% { box-shadow: 0 0 12px 0 rgba(16,185,129,.25); }
    50%     { box-shadow: 0 0 28px 4px rgba(16,185,129,.5); }
}
@keyframes glow-red {
    0%,100% { box-shadow: 0 0 12px 0 rgba(239,68,68,.25); }
    50%     { box-shadow: 0 0 28px 4px rgba(239,68,68,.5); }
}
@keyframes barFill { from { width: 0%; } }

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg,#05101f 0%,#080d1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
}
section[data-testid="stSidebar"] *      { color:#94a3b8 !important; font-family:'Inter',sans-serif !important; }
section[data-testid="stSidebar"] strong { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] h3     { color:#e2e8f0 !important; font-size:1rem !important; font-weight:700 !important; }
section[data-testid="stSidebar"] hr     { border-color:rgba(255,255,255,.08) !important; }
section[data-testid="stSidebar"] button {
    background:rgba(255,255,255,.05) !important;
    border:1px solid rgba(255,255,255,.1) !important;
    color:#94a3b8 !important;
    border-radius:8px !important;
}

/* Global text */
.stApp p, .stApp li, .stApp label { color:#cbd5e1; font-family:'Inter',sans-serif !important; }
h1,h2,h3,h4 { color:#f1f5f9 !important; font-family:'Inter',sans-serif !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.07);
    border-radius:14px; padding:5px; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    color:#64748b !important; border-radius:10px !important;
    font-weight:600 !important; font-size:.9rem !important;
    padding:.5rem 1.4rem !important; transition:all .2s ease !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1e3a5f,#0f3460) !important;
    color:#e2e8f0 !important;
    box-shadow:0 2px 10px rgba(15,52,96,.6) !important;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#1e3a5f 0%,#0f2d56 100%) !important;
    color:#e2e8f0 !important;
    border:1px solid rgba(99,163,232,.35) !important;
    border-radius:10px !important;
    font-family:'Inter',sans-serif !important;
    font-weight:600 !important; font-size:.88rem !important;
    letter-spacing:.03em !important; padding:.55rem 1.2rem !important;
    transition:all .2s ease !important;
}
.stButton > button:hover:not(:disabled) {
    background:linear-gradient(135deg,#2a5298 0%,#1a3a7c 100%) !important;
    border-color:rgba(99,163,232,.7) !important;
    transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(30,58,95,.5) !important;
}
.stButton > button:disabled { opacity:.35 !important; cursor:not-allowed !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,.025) !important;
    border:2px dashed rgba(99,163,232,.3) !important;
    border-radius:16px !important; padding:1.5rem !important;
    transition:border-color .3s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(99,163,232,.6) !important; }
[data-testid="stFileUploader"] * { color:#64748b !important; }

/* Expander */
.streamlit-expanderHeader {
    background:rgba(255,255,255,.04) !important;
    border:1px solid rgba(255,255,255,.07) !important;
    border-radius:10px !important; color:#64748b !important; font-size:.85rem !important;
}
.streamlit-expanderContent {
    background:rgba(255,255,255,.02) !important;
    border:1px solid rgba(255,255,255,.05) !important;
    border-top:none !important; border-radius:0 0 10px 10px !important;
}

hr { border-color:rgba(255,255,255,.07) !important; }
img { border-radius:14px !important; }

/* Hero */
.hero {
    position:relative; overflow:hidden;
    background:linear-gradient(135deg,#05101f 0%,#0d2147 55%,#05101f 100%);
    border:1px solid rgba(99,163,232,.18);
    border-radius:22px; padding:2.8rem 3.2rem; margin-bottom:2rem;
    animation:fadeInUp .5s ease;
}
.hero::after {
    content:''; position:absolute; top:-80px; right:-80px;
    width:320px; height:320px;
    background:radial-gradient(circle,rgba(59,130,246,.08) 0%,transparent 70%);
    pointer-events:none;
}
.hero h1 { font-size:2.3rem; font-weight:800; color:#f1f5f9 !important; margin:0 0 .5rem; line-height:1.15; }
.hero p  { font-size:1rem; color:#64748b !important; margin:0; line-height:1.6; }
.hero .tag {
    display:inline-block;
    background:rgba(59,130,246,.15); color:#60a5fa !important;
    border:1px solid rgba(59,130,246,.25); border-radius:9999px;
    padding:.2rem .9rem; font-size:.78rem; font-weight:600;
    letter-spacing:.05em; text-transform:uppercase; margin-bottom:1rem;
}

/* Result cards */
.card-ok {
    background:linear-gradient(135deg,rgba(5,46,22,.9),rgba(6,78,59,.85));
    border:1.5px solid #10b981; border-radius:20px;
    padding:2.2rem 1.8rem; text-align:center;
    animation:fadeInUp .35s ease, glow-green 2.5s ease-in-out infinite;
}
.card-defect {
    background:linear-gradient(135deg,rgba(45,10,10,.9),rgba(69,10,10,.85));
    border:1.5px solid #ef4444; border-radius:20px;
    padding:2.2rem 1.8rem; text-align:center;
    animation:fadeInUp .35s ease, glow-red 2.5s ease-in-out infinite;
}
.card-icon  { font-size:4rem; line-height:1; margin-bottom:.6rem; display:block; }
.card-title { font-size:1.7rem; font-weight:800; margin:.3rem 0 .4rem; font-family:'Inter',sans-serif; }
.card-ok    .card-title { color:#6ee7b7 !important; }
.card-defect .card-title { color:#fca5a5 !important; }
.card-sub   { font-size:.88rem; color:#64748b !important; }

/* Confidence bar */
.conf-wrap  { margin-top:1.4rem; }
.conf-label { font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#475569 !important; margin-bottom:.5rem; }
.conf-track { width:100%; height:8px; background:rgba(255,255,255,.07); border-radius:9999px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:9999px; animation:barFill .6s ease; }
.conf-value { text-align:right; font-size:1.05rem; font-weight:700; color:#e2e8f0 !important; margin-top:.35rem; }

.section-label {
    font-size:.72rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#334155 !important; margin-bottom:.75rem;
}

/* Live badge */
.live-badge {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(16,185,129,.1); border:1px solid rgba(16,185,129,.3);
    border-radius:9999px; padding:.3rem .9rem;
    font-size:.82rem; font-weight:700; color:#10b981 !important;
}
.live-dot { width:8px; height:8px; background:#10b981; border-radius:50%; display:inline-block; animation:blink 1.2s ease-in-out infinite; }
.off-badge {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:9999px; padding:.3rem .9rem;
    font-size:.82rem; font-weight:600; color:#334155 !important;
}

/* History */
.hist-item {
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:.55rem .8rem; margin-bottom:.4rem;
    display:flex; align-items:center; justify-content:space-between;
    font-size:.82rem; animation:fadeInUp .25s ease;
}
.hist-name    { color:#94a3b8 !important; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:120px; }
.badge-ok     { background:rgba(16,185,129,.15); color:#6ee7b7 !important; border:1px solid rgba(16,185,129,.3); border-radius:9999px; padding:.15rem .6rem; font-size:.75rem; font-weight:700; white-space:nowrap; }
.badge-defect { background:rgba(239,68,68,.15);  color:#fca5a5 !important; border:1px solid rgba(239,68,68,.3);  border-radius:9999px; padding:.15rem .6rem; font-size:.75rem; font-weight:700; white-space:nowrap; }
.hist-conf    { color:#475569 !important; font-size:.75rem; white-space:nowrap; }

/* Empty state */
.empty-state {
    text-align:center; padding:4rem 2rem; color:#1e293b !important;
    font-size:.95rem; border:2px dashed rgba(255,255,255,.06);
    border-radius:16px; margin-top:1rem;
}
.empty-state .icon { font-size:2.5rem; display:block; margin-bottom:.8rem; opacity:.4; }

/* Glass panel */
.glass-panel {
    background:rgba(255,255,255,.025);
    border:1px solid rgba(255,255,255,.07);
    border-radius:18px; padding:1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _conf_bar(confidence: float, ok: bool) -> str:
    color = "#10b981" if ok else "#ef4444"
    pct   = int(confidence * 100)
    return f"""
    <div class="conf-wrap">
      <div class="conf-label">Model Confidence</div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color});"></div>
      </div>
      <div class="conf-value">{pct}%</div>
    </div>"""


def _result_card(result: InterpretedResult) -> None:
    if result.is_ok:
        st.markdown("""
        <div class="card-ok">
          <span class="card-icon">✅</span>
          <div class="card-title">No Defect Detected</div>
          <div class="card-sub">Fabric passed quality inspection</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card-defect">
          <span class="card-icon">❌</span>
          <div class="card-title">Defect Detected</div>
          <div class="card-sub">Fabric failed quality inspection</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(_conf_bar(result.confidence, result.is_ok), unsafe_allow_html=True)


def _draw_overlay(frame_bgr: np.ndarray, result: InterpretedResult) -> np.ndarray:
    frame = frame_bgr.copy()
    color = (0, 200, 0) if result.is_ok else (0, 0, 220)
    label = "OK" if result.is_ok else "DEFECT"
    conf  = f"{result.confidence:.0%}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for text, y, scale in [(label, 40, 1.0), (conf, 72, 0.85)]:
        cv2.putText(frame, text, (11, y+1), font, scale, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y),   font, scale, color,   2, cv2.LINE_AA)
    return frame


def _history_badge(ok: bool) -> str:
    return '<span class="badge-ok">✅ OK</span>' if ok else '<span class="badge-defect">❌ Defect</span>'


# ── Cached predictor ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_predictor() -> FabricPredictor:
    return FabricPredictor(ENDPOINT_ID, API_KEY)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "history":        [],
    "live_mode":      False,
    "live_cap":       None,
    "live_result":    InterpretedResult(is_ok=True, label="ok", confidence=1.0),
    "last_pred_time": 0.0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="tag">AI Quality Control</div>
  <h1>🧵 Fabric Defect Inspector</h1>
  <p>Real-time defect detection — live camera feed or single image upload.</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_cam, tab_upload = st.tabs(["Live Camera inspector", "Upload Image inspector"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE CAMERA
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    ctrl_l, ctrl_r, status_col = st.columns([1, 1, 5])
    with ctrl_l:
        start_clicked = st.button("▶  Start", use_container_width=True,
                                  disabled=st.session_state.live_mode)
    with ctrl_r:
        stop_clicked = st.button("⏹  Stop", use_container_width=True,
                                 disabled=not st.session_state.live_mode)
    with status_col:
        if st.session_state.live_mode:
            st.markdown('<div style="padding-top:.4rem"><div class="live-badge"><span class="live-dot"></span>LIVE</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding-top:.4rem"><div class="off-badge">⏸ &nbsp;OFF</div></div>', unsafe_allow_html=True)

    if start_clicked:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            st.error("Cannot open camera at device index 1. Make sure your camera is connected.")
        else:
            st.session_state.live_cap      = cap
            st.session_state.live_mode     = True
            st.session_state.last_pred_time = 0.0
            st.rerun()

    if stop_clicked:
        if st.session_state.live_cap is not None:
            st.session_state.live_cap.release()
            st.session_state.live_cap = None
        st.session_state.live_mode = False
        st.rerun()

    if st.session_state.live_mode:
        cap = st.session_state.live_cap
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(1)
            st.session_state.live_cap = cap

        ret, raw_frame = cap.read()
        if not ret:
            st.warning("Failed to read frame — retrying…")
            time.sleep(1)
            st.rerun()

        display_frame, pil_image = preprocess(raw_frame, target_width=FRAME_WIDTH)

        now = time.monotonic()
        if now - st.session_state.last_pred_time >= CAPTURE_INTERVAL_SEC:
            try:
                preds  = get_predictor().predict(pil_image)
                result = interpret(preds)
                st.session_state.live_result    = result
                st.session_state.last_pred_time = now
                st.session_state.history.append({
                    "name": "Live",
                    "ok":   result.is_ok,
                    "conf": f"{result.confidence:.0%}",
                })
            except Exception as exc:
                st.error(f"Inference error: {exc}")

        result    = st.session_state.live_result
        annotated = _draw_overlay(display_frame, result)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        cam_col, res_col = st.columns([3, 2], gap="large")
        with cam_col:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.image(rgb_frame, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with res_col:
            _result_card(result)

        time.sleep(0.05)
        st.rerun()

    elif not start_clicked:
        st.markdown("""
        <div class="empty-state">
          <span class="icon">📷</span>
          Press <strong>▶ Start</strong> to open the DroidCam live feed.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    uploaded = st.file_uploader(
        "Drop a fabric image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div class="empty-state">
          <span class="icon">🖼️</span>
          Drop or browse a fabric image above to run an inspection.
        </div>""", unsafe_allow_html=True)
    else:
        original = Image.open(uploaded).convert("RGB")

        with st.spinner("Analysing fabric…"):
            try:
                enhanced = preprocess_pil(original, target_width=FRAME_WIDTH)
                preds    = get_predictor().predict(enhanced)
                result   = interpret(preds)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.session_state.history.append({
            "name": uploaded.name,
            "ok":   result.is_ok,
            "conf": f"{result.confidence:.0%}",
        })

        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown('<div class="section-label">Uploaded Image</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("🔬 What the model sees"):
                st.image(enhanced, use_container_width=True)

        with res_col:
            st.markdown('<div class="section-label">Inspection Result</div>', unsafe_allow_html=True)
            _result_card(result)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 Raw model output"):
                st.write(preds)
