"""
app.py  — Spoken Language Identifier
-------------------------------------
Streamlit + streamlit-webrtc real-time demo.

Architecture
------------
Browser mic → WebRTC → AudioProcessorTrack (AudioProcessor class)
  └─ push PCM frames into VoiceActivityDetector
  └─ completed word segments queued into st.session_state.word_queue
Main thread polls word_queue → LanguageIDInference.predict_waveform()
  └─ result appended to history → UI re-renders

Run
---
  streamlit run app.py
"""

from __future__ import annotations

import queue
import threading
import time
import os
import sys
from pathlib import Path
from typing import Optional

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vad import VoiceActivityDetector, WordSegment, TARGET_SR
from inference import LanguageIDInference, Prediction

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaLens",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:         #0a0a0f;
    --surface:    #13131a;
    --border:     #1e1e2e;
    --accent-de:  #ff6b6b;
    --accent-en:  #4ecdc4;
    --accent-es:  #ffe66d;
    --text:       #e8e8f0;
    --muted:      #5a5a7a;
    --glow-de:    rgba(255,107,107,0.15);
    --glow-en:    rgba(78,205,196,0.15);
    --glow-es:    rgba(255,230,109,0.15);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Hero header ─────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.8rem, 6vw, 5rem);
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent-en) 0%, var(--accent-es) 50%, var(--accent-de) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.6rem;
}

/* ── Big prediction card ─────────────────────────────────────────── */
.pred-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.pred-card.de { border-color: var(--accent-de); box-shadow: 0 0 40px var(--glow-de); }
.pred-card.en { border-color: var(--accent-en); box-shadow: 0 0 40px var(--glow-en); }
.pred-card.es { border-color: var(--accent-es); box-shadow: 0 0 40px var(--glow-es); }

.pred-lang {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.pred-lang.de { color: var(--accent-de); }
.pred-lang.en { color: var(--accent-en); }
.pred-lang.es { color: var(--accent-es); }
.pred-lang.unknown { color: var(--muted); }

.pred-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
}

/* ── Confidence bars ─────────────────────────────────────────────── */
.bar-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 1rem 0;
}
.bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    width: 70px;
    flex-shrink: 0;
}
.bar-label.de { color: var(--accent-de); }
.bar-label.en { color: var(--accent-en); }
.bar-label.es { color: var(--accent-es); }
.bar-track {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.35s cubic-bezier(0.4,0,0.2,1);
}
.bar-fill.de { background: var(--accent-de); }
.bar-fill.en { background: var(--accent-en); }
.bar-fill.es { background: var(--accent-es); }
.bar-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    width: 42px;
    text-align: right;
}

/* ── Word history feed ───────────────────────────────────────────── */
.history-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border);
}
.history-feed {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    max-height: 160px;
    overflow-y: auto;
    padding-right: 4px;
}
.history-chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid;
    opacity: 0.85;
}
.history-chip.de {
    color: var(--accent-de); border-color: var(--accent-de);
    background: rgba(255,107,107,0.07);
}
.history-chip.en {
    color: var(--accent-en); border-color: var(--accent-en);
    background: rgba(78,205,196,0.07);
}
.history-chip.es {
    color: var(--accent-es); border-color: var(--accent-es);
    background: rgba(255,230,109,0.07);
}

/* ── Status pill ─────────────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    border: 1px solid;
}
.status-pill.listening {
    color: #4ade80; border-color: #4ade80;
    background: rgba(74,222,128,0.08);
    animation: pulse-green 1.5s ease-in-out infinite;
}
.status-pill.idle {
    color: var(--muted); border-color: var(--border);
    background: transparent;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.25); }
    50%      { box-shadow: 0 0 0 6px rgba(74,222,128,0); }
}

/* ── Stats row ───────────────────────────────────────────────────── */
.stats-row {
    display: flex;
    gap: 1.5rem;
    padding: 0.75rem 0;
    border-top: 1px solid var(--border);
    margin-top: 0.5rem;
}
.stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1.2rem;
    color: var(--text);
}
.stat-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── WebRTC widget override ──────────────────────────────────────── */
[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    border-color: var(--accent-en) !important;
    color: var(--accent-en) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Module-level queue — shared between the WebRTC background thread and the
#    Streamlit main thread. Must NOT live in st.session_state because the
#    AudioProcessor is instantiated in a background thread that has no access
#    to session state.
WORD_QUEUE: queue.Queue = queue.Queue()

# ── Session state init (UI-only state) ───────────────────────────────────────
def _init_state():
    if "history"       not in st.session_state:
        st.session_state.history       = []   # list[Prediction]
    if "latest"        not in st.session_state:
        st.session_state.latest        = None
    if "total_words"   not in st.session_state:
        st.session_state.total_words   = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0

_init_state()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(artefacts_dir: str) -> Optional[LanguageIDInference]:
    try:
        return LanguageIDInference(artefacts_dir)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

ARTEFACTS_DIR = os.environ.get("MODEL_DIR", "./model_artefacts")
model = load_model(ARTEFACTS_DIR)

# ── Audio processor ───────────────────────────────────────────────────────────
class AudioProcessor(AudioProcessorBase):
    """
    Receives PyAV AudioFrame objects from the browser mic via WebRTC.
    Converts them to float32 mono PCM → pushes into VAD.
    When the VAD signals a word boundary, the segment is enqueued for
    main-thread inference.
    """

    def __init__(self):
        self._vad   = VoiceActivityDetector()
        self._queue = WORD_QUEUE   # module-level global — safe from background thread

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to float32 numpy array (mono)
        pcm = frame.to_ndarray()             # shape: (channels, samples)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)           # mix to mono
        pcm = pcm.astype(np.float32)

        # Resample if the browser sends a different rate
        in_sr = frame.sample_rate
        if in_sr != TARGET_SR:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=in_sr, target_sr=TARGET_SR)

        # Feed to VAD
        self._vad.push(pcm)
        word = self._vad.pop_word()
        if word is not None:
            try:
                self._queue.put_nowait(word)
            except queue.Full:
                pass   # drop if queue is backed up

        return frame   # pass audio through unchanged (no output needed)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI Layout
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">LinguaLens</p>
  <p class="hero-sub">real-time spoken language detection &nbsp;·&nbsp; en / de / es</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left: mic control + live prediction ──────────────────────────────────────
with col_left:
    # Model status banner
    if model is None:
        st.warning(
            "⚠️ Model not found. Place your `model_artefacts/` directory "
            "(containing `language_id_model.onnx`, `scaler.pkl`, `label_encoder.pkl`) "
            "next to `app.py` and restart.",
            icon="⚠️",
        )
    else:
        st.markdown(
            '<span class="status-pill idle" id="model-ok">✓ model ready</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # WebRTC mic widget
    ctx = webrtc_streamer(
        key="lang-id",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    # Status pill
    is_live = ctx.state.playing if ctx and ctx.state else False
    status_class = "listening" if is_live else "idle"
    status_text  = "🎙 listening" if is_live else "⏸ idle"
    st.markdown(
        f'<br><span class="status-pill {status_class}">{status_text}</span>',
        unsafe_allow_html=True,
    )

    # ── Latest prediction card ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pred_placeholder   = st.empty()
    bars_placeholder   = st.empty()
    stats_placeholder  = st.empty()

    def _render_prediction(p: Optional[Prediction]):
        lang_code = {
            "english": "en", "german": "de", "spanish": "es"
        }.get(p.label if p else "", "unknown")

        flag = {"en": "🇬🇧", "de": "🇩🇪", "es": "🇪🇸"}.get(lang_code, "❓")

        if p:
            pred_placeholder.markdown(f"""
<div class="pred-card {lang_code}">
  <div class="pred-lang {lang_code}">{flag} {p.label.upper()}</div>
  <div class="pred-meta">
    confidence {max(p.probs.values())*100:.0f}%
    &nbsp;·&nbsp; {p.latency_ms:.0f} ms inference
    &nbsp;·&nbsp; {p.audio_duration_s:.2f} s clip
  </div>
</div>""", unsafe_allow_html=True)

            # Confidence bars
            lang_order = [("english", "en"), ("german", "de"), ("spanish", "es")]
            bars_html  = '<div class="bar-wrap">'
            for lang_name, lc in lang_order:
                pct = p.probs.get(lang_name, 0.0) * 100
                bars_html += f"""
<div class="bar-row">
  <span class="bar-label {lc}">{lc.upper()}</span>
  <div class="bar-track">
    <div class="bar-fill {lc}" style="width:{pct:.1f}%"></div>
  </div>
  <span class="bar-pct">{pct:.0f}%</span>
</div>"""
            bars_html += "</div>"
            bars_placeholder.markdown(bars_html, unsafe_allow_html=True)
        else:
            pred_placeholder.markdown("""
<div class="pred-card">
  <div class="pred-lang unknown">— speak —</div>
  <div class="pred-meta">waiting for audio…</div>
</div>""", unsafe_allow_html=True)
            bars_placeholder.empty()

    _render_prediction(st.session_state.latest)

# ── Right: word history + stats ───────────────────────────────────────────────
with col_right:
    history_header = st.empty()
    history_area   = st.empty()
    stats_area     = st.empty()

    def _render_history():
        history = st.session_state.history
        if not history:
            history_header.markdown(
                '<p class="history-header">word history</p>',
                unsafe_allow_html=True,
            )
            history_area.markdown(
                '<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
                'color:#5a5a7a;">No words detected yet. Click START and speak.</p>',
                unsafe_allow_html=True,
            )
            stats_area.empty()
            return

        history_header.markdown(
            f'<p class="history-header">word history &nbsp;·&nbsp; {len(history)} word(s)</p>',
            unsafe_allow_html=True,
        )

        lang_code_map = {"english": "en", "german": "de", "spanish": "es"}
        chips = ""
        # Show newest first (last 80 entries)
        for p in reversed(history[-80:]):
            lc = lang_code_map.get(p.label, "unknown")
            pct = int(max(p.probs.values()) * 100)
            chips += f'<span class="history-chip {lc}">{lc.upper()} {pct}%</span>'

        history_area.markdown(
            f'<div class="history-feed">{chips}</div>',
            unsafe_allow_html=True,
        )

        # Stats
        n = len(history)
        counts = {"english": 0, "german": 0, "spanish": 0}
        for p in history:
            counts[p.label] = counts.get(p.label, 0) + 1
        dominant = max(counts, key=counts.get)
        avg_lat  = st.session_state.total_latency / n if n > 0 else 0

        lang_code_map2 = {"english": "en", "german": "de", "spanish": "es"}
        dc = lang_code_map2.get(dominant, "unknown")

        stats_area.markdown(f"""
<div class="stats-row">
  <div class="stat-item">
    <span class="stat-val">{n}</span>
    <span class="stat-key">words</span>
  </div>
  <div class="stat-item">
    <span class="stat-val" style="color:var(--accent-{dc})">{dominant.upper()}</span>
    <span class="stat-key">dominant lang</span>
  </div>
  <div class="stat-item">
    <span class="stat-val">{counts['english']}</span>
    <span class="stat-key" style="color:var(--accent-en)">english</span>
  </div>
  <div class="stat-item">
    <span class="stat-val">{counts['german']}</span>
    <span class="stat-key" style="color:var(--accent-de)">german</span>
  </div>
  <div class="stat-item">
    <span class="stat-val">{counts['spanish']}</span>
    <span class="stat-key" style="color:var(--accent-es)">spanish</span>
  </div>
  <div class="stat-item">
    <span class="stat-val">{avg_lat:.0f}<span style="font-size:0.7rem;color:var(--muted)">ms</span></span>
    <span class="stat-key">avg latency</span>
  </div>
</div>
""", unsafe_allow_html=True)

    _render_history()

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.button("Clear history", key="clear"):
        st.session_state.history.clear()
        st.session_state.latest        = None
        st.session_state.total_words   = 0
        st.session_state.total_latency = 0.0
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  Main polling loop — drain the word queue and run inference
# ═══════════════════════════════════════════════════════════════════════════════
if is_live and model is not None:
    processed = 0
    while processed < 5:
        try:
            word_seg: WordSegment = WORD_QUEUE.get_nowait()
        except queue.Empty:
            break

        pred = model.predict_waveform(word_seg.samples, sr=TARGET_SR)
        st.session_state.history.append(pred)
        st.session_state.latest         = pred
        st.session_state.total_words   += 1
        st.session_state.total_latency += pred.latency_ms
        processed += 1

    if processed > 0:
        _render_prediction(st.session_state.latest)
        _render_history()

    # Auto-rerun every 300 ms while mic is live to poll the queue
    time.sleep(0.3)
    st.rerun()
