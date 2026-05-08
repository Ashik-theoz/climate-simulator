"""London Climate Engineering Simulator — Streamlit UI.

The model lives in `simulator/`. This file is presentation only.
"""

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go

from simulator import (
    simulate, monte_carlo, sensitivity_analysis, local_risk,
    LONDON_LOCATIONS, SSP_SCENARIOS, HADCRUT_HISTORICAL, REFERENCES,
    MODEL_VERSION, DEFAULT_PARAMS,
)
from simulator.data import SCENARIO_LIBRARY, WONG_PALETTE
from simulator.report import build_report


# ============================================================================
# Page config + theme
# ============================================================================
st.set_page_config(
    page_title="Can Engineering Reverse the Climate Clock?",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(46,134,222,0.18), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(238,90,36,0.18), transparent 60%),
        linear-gradient(180deg, #0b1220 0%, #0e1726 60%, #0b1220 100%);
}
.hero-wrap { padding: 24px 28px; border-radius: 18px;
    background: linear-gradient(135deg, rgba(46,134,222,0.18), rgba(238,90,36,0.18));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    margin-bottom: 8px; position: relative; overflow: hidden;
}
.hero-title { font-size: 40px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05;
    background: linear-gradient(90deg, #7dd3fc 0%, #fbbf24 50%, #f87171 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent; margin: 0;
}
.hero-sub { color: #cbd5e1; font-size: 15px; margin-top: 14px; opacity: 0.92; }
.hero-welcome {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.22em;
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 999px;
    padding: 4px 12px;
    margin-bottom: 10px;
    text-transform: uppercase;
    animation: badgeIn 0.6s cubic-bezier(0.2,0.8,0.2,1) both, badgeGlow 3s ease-in-out infinite 1s;
}
@keyframes badgeIn  { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes badgeGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(251,191,36,0.0); }
    50%     { box-shadow: 0 0 18px 2px rgba(251,191,36,0.35); }
}

.hero-tagline {
    margin-top: 12px;
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.01em;
}
.hero-tagline .tag-step {
    display: inline-block;
    opacity: 0;
    animation: tagReveal 0.5s cubic-bezier(0.2,0.8,0.2,1) both;
}
.hero-tagline .tag-step:nth-of-type(1) { animation-delay: 0.55s; color: #7dd3fc; }
.hero-tagline .tag-step:nth-of-type(2) { animation-delay: 0.85s; color: #fbbf24; }
.hero-tagline .tag-step:nth-of-type(3) { animation-delay: 1.15s; color: #f87171; font-weight: 700; }
.hero-tagline .tag-dot {
    color: rgba(255,255,255,0.3);
    margin: 0 10px;
    opacity: 0;
    animation: tagReveal 0.5s ease 0.7s both;
}
.hero-tagline .tag-dot:nth-of-type(2) { animation-delay: 1.0s; }
@keyframes tagReveal {
    from { opacity: 0; transform: translateY(8px); filter: blur(4px); }
    to   { opacity: 1; transform: translateY(0); filter: blur(0); }
}
.hero-icons { position: absolute; right: 24px; top: 18px; font-size: 36px;
    opacity: 0.95; letter-spacing: 6px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4));
}
.hero-icons .ic { display:inline-block; animation: float 4s ease-in-out infinite; }
.hero-icons .ic:nth-child(2){ animation-delay: 0.6s; }
.hero-icons .ic:nth-child(3){ animation-delay: 1.2s; }
@keyframes float { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }

.metric-card { border-radius: 14px; padding: 16px 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03); backdrop-filter: blur(6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.25);
}
.metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 32px; font-weight: 800; margin-top: 4px; }
.metric-trend { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.severity-low{color:#34d399;} .severity-mod{color:#fbbf24;} .severity-high{color:#fb923c;} .severity-vhigh{color:#f87171;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1726 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(46,134,222,0.25), rgba(238,90,36,0.25)); }
.stButton button { border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); transition: all 0.2s ease; }
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(46,134,222,0.25); }

.ai-chip { display:inline-block; padding:4px 10px; border-radius:999px;
    font-size:12px; font-weight:600; letter-spacing:0.04em; margin-left: 8px; }
.ai-chip.online  { background: rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.4); }
.ai-chip.offline { background: rgba(251,146,60,0.15); color:#fb923c; border:1px solid rgba(251,146,60,0.4); }

.refs-card { border-radius: 12px; padding: 14px 16px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px; font-size: 13px; line-height: 1.5; }
.refs-key { color: #7dd3fc; font-weight: 700; margin-right: 6px; }
.refs-doi { color: #94a3b8; font-size: 11px; }

.eq { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 10px 14px;
    border: 1px solid rgba(255,255,255,0.08); font-family: 'JetBrains Mono', 'Menlo', monospace;
    font-size: 14px; color: #e2e8f0; margin: 8px 0; }

/* === Entrance animations === */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; } to { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-24px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.hero-wrap        { animation: fadeInUp 0.7s cubic-bezier(0.2,0.8,0.2,1); }
.metric-card      { animation: fadeInUp 0.7s cubic-bezier(0.2,0.8,0.2,1) 0.15s both; }
.metric-card:nth-of-type(2) { animation-delay: 0.25s; }
.metric-card:nth-of-type(3) { animation-delay: 0.35s; }
.stTabs           { animation: fadeIn 0.9s ease 0.45s both; }
section[data-testid="stSidebar"] > div { animation: slideInLeft 0.6s cubic-bezier(0.2,0.8,0.2,1); }

/* Subtle shimmer across the hero title — reads "alive" */
.hero-title {
    background: linear-gradient(90deg, #7dd3fc 0%, #fbbf24 25%, #f87171 50%, #fbbf24 75%, #7dd3fc 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shimmer 8s linear infinite;
}

/* === Sidebar polish === */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    background: linear-gradient(90deg, #7dd3fc, #fbbf24);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 700;
    letter-spacing: -0.01em;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 6px;
    margin-bottom: 8px;
}

/* Slider track + thumb */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 0 0 4px rgba(125,211,252,0.18), 0 0 12px rgba(125,211,252,0.5);
    transition: all 0.18s ease;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(125,211,252,0.28), 0 0 18px rgba(125,211,252,0.7);
    transform: scale(1.08);
}

/* Sidebar button hover lift */
section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(46,134,222,0.3);
    border-color: rgba(125,211,252,0.4);
}

/* Scenario library + AI assistant panels look like cards */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stToggle {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 4px;
}

/* === Tab content fade on switch === */
.stTabs [role="tabpanel"] { animation: fadeIn 0.5s ease; }

/* === Ambient mood layer (set by python below via a div) === */
.mood-layer {
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    transition: opacity 1.2s ease, background 1.2s ease;
}
.mood-flood {
    background: radial-gradient(800px 600px at 70% 30%, rgba(56,189,248,0.10), transparent 70%);
}
.mood-drought {
    background: radial-gradient(800px 600px at 30% 70%, rgba(251,146,60,0.12), transparent 70%);
}
.mood-extreme {
    background: radial-gradient(900px 700px at 50% 50%, rgba(248,113,113,0.16), transparent 70%);
    animation: extremePulse 3s ease-in-out infinite;
}
@keyframes extremePulse {
    0%,100% { opacity: 0.85; } 50% { opacity: 1; }
}

/* Plotly background transparency so charts blend with our theme */
.js-plotly-plot, .plot-container { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SPLASH SCREEN — game-style intro
# ============================================================================
if not st.session_state.get("splash_shown", False):
    # Hide sidebar + header for a true full-screen takeover
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"]   { display: none !important; }
    div[data-testid="stToolbar"]     { display: none !important; }
    .block-container { padding-top: 0 !important; max-width: 100% !important; }

    .splash {
        position: relative;
        min-height: 90vh;
        padding: 40px 20px 30px;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        text-align: center;
        background:
            radial-gradient(ellipse 1200px 600px at 50% 0%, rgba(46,134,222,0.30), transparent 60%),
            radial-gradient(ellipse 800px 500px at 90% 100%, rgba(238,90,36,0.28), transparent 60%),
            radial-gradient(ellipse 800px 500px at 10% 100%, rgba(125,211,252,0.18), transparent 60%);
        border-radius: 20px;
        overflow: hidden;
    }
    /* Floating ambient particles */
    .splash::before {
        content: '';
        position: absolute; inset: 0;
        background-image:
            radial-gradient(circle at 15% 20%, rgba(125,211,252,0.4) 1px, transparent 1.5px),
            radial-gradient(circle at 85% 30%, rgba(251,191,36,0.4) 1px, transparent 1.5px),
            radial-gradient(circle at 25% 70%, rgba(248,113,113,0.4) 1px, transparent 1.5px),
            radial-gradient(circle at 75% 85%, rgba(125,211,252,0.4) 1px, transparent 1.5px);
        background-size: 240px 240px;
        animation: drift 18s linear infinite;
        opacity: 0.6;
    }
    @keyframes drift {
        0%   { background-position: 0 0, 0 0, 0 0, 0 0; }
        100% { background-position: 240px 240px, -240px 240px, 240px -240px, -240px -240px; }
    }

    .splash-icons {
        font-size: 72px;
        letter-spacing: 28px;
        margin-bottom: 28px;
        filter: drop-shadow(0 6px 18px rgba(0,0,0,0.5));
        animation: floatIcons 3.6s ease-in-out infinite;
    }
    .splash-icons span:nth-child(1) { animation: bounceIcon 3s ease-in-out infinite; display: inline-block; }
    .splash-icons span:nth-child(2) { animation: bounceIcon 3s ease-in-out infinite 0.5s; display: inline-block; }
    .splash-icons span:nth-child(3) { animation: bounceIcon 3s ease-in-out infinite 1.0s; display: inline-block; }
    @keyframes bounceIcon {
        0%,100% { transform: translateY(0) rotate(0); }
        50%     { transform: translateY(-14px) rotate(5deg); }
    }
    @keyframes floatIcons {
        0%,100% { filter: drop-shadow(0 6px 18px rgba(0,0,0,0.5)); }
        50%     { filter: drop-shadow(0 12px 26px rgba(125,211,252,0.4)); }
    }

    .splash-welcome {
        display: inline-block;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5em;
        color: #fbbf24;
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.35);
        border-radius: 999px;
        padding: 6px 22px 6px 28px;
        margin-bottom: 24px;
        animation: badgeIn 0.7s cubic-bezier(0.2,0.8,0.2,1) 0.2s both, badgeGlow 3s ease-in-out infinite 1s;
        text-transform: uppercase;
    }

    .splash-lab {
        font-size: clamp(36px, 6vw, 64px);
        font-weight: 900;
        letter-spacing: -0.02em;
        line-height: 1.0;
        background: linear-gradient(90deg, #7dd3fc 0%, #fbbf24 50%, #f87171 100%);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
        animation: fadeInUp 0.9s cubic-bezier(0.2,0.8,0.2,1) 0.5s both, shimmer 6s linear infinite 0.5s;
        text-shadow: 0 0 40px rgba(125,211,252,0.3);
    }

    .splash-divider {
        width: 120px; height: 2px;
        margin: 26px 0 20px;
        background: linear-gradient(90deg, transparent, #fbbf24, transparent);
        animation: fadeIn 0.8s ease 0.9s both;
    }

    .splash-question {
        font-size: clamp(20px, 2.4vw, 30px);
        font-style: italic;
        color: #e2e8f0;
        font-weight: 400;
        margin: 0 0 18px;
        animation: fadeInUp 0.9s cubic-bezier(0.2,0.8,0.2,1) 1.05s both;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }

    .splash-tagline {
        font-size: 15px;
        color: #94a3b8;
        letter-spacing: 0.04em;
        margin-bottom: 40px;
        animation: fadeInUp 0.8s ease 1.3s both;
    }
    .splash-tagline .ts { color: #7dd3fc; }
    .splash-tagline .ts2 { color: #fbbf24; }
    .splash-tagline .ts3 { color: #f87171; font-weight: 700; }
    .splash-tagline .sep { color: rgba(255,255,255,0.25); margin: 0 10px; }

    .splash-prompt {
        font-size: 12px; color: #64748b; letter-spacing: 0.2em; text-transform: uppercase;
        margin-top: 12px;
        animation: fadeIn 0.8s ease 1.6s both, blinkSoft 2s ease-in-out infinite 2s;
    }
    @keyframes blinkSoft { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }

    /* Style the Streamlit "Enter the Lab" button when on splash */
    .splash + div .stButton button {
        background: linear-gradient(135deg, #7dd3fc 0%, #fbbf24 50%, #f87171 100%) !important;
        background-size: 200% 100% !important;
        color: #0b1220 !important;
        font-size: 17px !important;
        font-weight: 800 !important;
        letter-spacing: 0.1em !important;
        padding: 16px 36px !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 40px rgba(125,211,252,0.4), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
        animation: shimmer 5s linear infinite, btnPulse 2.4s ease-in-out infinite, fadeInUp 0.9s ease 1.5s both !important;
        transition: transform 0.18s ease !important;
    }
    .splash + div .stButton button:hover {
        transform: translateY(-3px) scale(1.04) !important;
        box-shadow: 0 18px 50px rgba(125,211,252,0.6), 0 0 0 1px rgba(255,255,255,0.15) inset !important;
    }
    @keyframes btnPulse {
        0%,100% { box-shadow: 0 12px 40px rgba(125,211,252,0.4), 0 0 0 1px rgba(255,255,255,0.1) inset; }
        50%     { box-shadow: 0 12px 50px rgba(251,191,36,0.55), 0 0 0 1px rgba(255,255,255,0.15) inset; }
    }

    .splash-credit {
        margin-top: 30px;
        font-size: 11px;
        color: #475569;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        animation: fadeIn 1s ease 2s both;
    }
    </style>

    <div class="splash">
      <div class="splash-icons"><span>🌍</span><span>💧</span><span>🔥</span></div>
      <div class="splash-welcome">⚡ &nbsp; W E L C O M E &nbsp;&nbsp; T O &nbsp; ⚡</div>
      <div class="splash-lab">THE CLIMATE<br/>ENGINEERING LAB</div>
      <div class="splash-divider"></div>
      <div class="splash-question">"Can Engineering Reverse the Climate Clock?"</div>
      <div class="splash-tagline">
        <span class="ts">Pull the levers</span>
        <span class="sep">·</span>
        <span class="ts2">Watch London respond</span>
        <span class="sep">·</span>
        <span class="ts3">Reverse the clock</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered Enter button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("⚡  ENTER THE LAB  →", use_container_width=True, key="enter_lab"):
            st.session_state["splash_shown"] = True
            st.rerun()

    st.markdown("""
    <div style="text-align:center;">
      <div class="splash-prompt">▼  Click to begin  ▼</div>
      <div class="splash-credit">Imperial College London · MSc Environmental Engineering · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================================
# MAIN APP — only reached after splash dismissed
# ============================================================================
st.markdown("""
<div class="hero-wrap">
  <div class="hero-icons"><span class="ic">🌍</span><span class="ic">💧</span><span class="ic">🔥</span></div>
  <div class="hero-welcome">⚡ WELCOME TO THE CLIMATE ENGINEERING LAB</div>
  <div class="hero-title">Can Engineering Reverse the Climate Clock?</div>
  <div class="hero-tagline">
    <span class="tag-step">Pull the levers</span>
    <span class="tag-dot">·</span>
    <span class="tag-step">Watch London respond</span>
    <span class="tag-dot">·</span>
    <span class="tag-step">Reverse the clock</span>
  </div>
  <div class="hero-sub">An interactive London-scale climate simulator — your engineering choices shape the future of flood, drought and heat in real time.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Session state defaults — synced with URL query params for shareable links
# ============================================================================
DEFAULTS = {
    "mode": "Standard", "years": 80, "co2_ppm": 450,
    "rainfall_change_pct": 10, "green_infra_pct": 20, "urbanization_pct": 40,
    "challenge_on": False, "challenge_won": False, "difficulty_choice": "Medium",
    "compare_on": False, "scenario_A": None, "scenario_B": None,
    "chat_history": [], "show_uncertainty": True, "show_ssp": True, "show_history": True,
    "researcher_mode": False,
}
DIFFICULTY_TARGETS = {
    "Easy": {"target_flood": 55, "target_drought": 55},
    "Medium": {"target_flood": 40, "target_drought": 40},
    "Hard": {"target_flood": 30, "target_drought": 30},
}

# Apply URL query params on first load
if "_params_applied" not in st.session_state:
    qp = st.query_params
    for k in ("years", "co2_ppm", "rainfall_change_pct", "green_infra_pct", "urbanization_pct"):
        if k in qp:
            try: DEFAULTS[k] = int(float(qp[k]))
            except Exception: pass
    st.session_state["_params_applied"] = True

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def snapshot(df: pd.DataFrame) -> dict:
    return {
        "params": {k: int(st.session_state[k]) for k in
                   ("years", "co2_ppm", "rainfall_change_pct", "green_infra_pct", "urbanization_pct")},
        "df": df.copy(),
    }


def pretty_params(p: dict) -> str:
    return (f"Years={p['years']}, CO₂={p['co2_ppm']} ppm, Rain={p['rainfall_change_pct']}%, "
            f"Green={p['green_infra_pct']}%, Urban={p['urbanization_pct']}%")


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Reset to Default"):
        for k in list(DEFAULTS.keys()):
            st.session_state.pop(k, None)
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.subheader("📚 Scenario library")
    sl_keys = list(SCENARIO_LIBRARY.keys())
    chosen = st.selectbox("Pick an IPCC-anchored or London-policy scenario:",
                          ["— custom —"] + sl_keys, index=0, key="scenario_pick")
    if chosen != "— custom —" and st.button("Apply scenario"):
        for k, v in SCENARIO_LIBRARY[chosen].items():
            st.session_state[k] = v
        st.session_state["challenge_won"] = False
        st.rerun()

    st.divider()

    st.radio("Mode", ["Standard", "Kids (simple)"], key="mode")
    st.slider("Simulation horizon (years)", 20, 120, step=5, key="years")
    st.slider("CO₂ concentration (ppm)", 280, 900, step=10, key="co2_ppm")
    st.slider("Rainfall change (%)", -30, 50, step=1, key="rainfall_change_pct")

    if st.session_state["mode"] == "Kids (simple)":
        st.slider("Green solutions (%)", 0, 100, step=5, key="green_infra_pct")
        st.session_state["urbanization_pct"] = 45
        st.info("Kids mode uses fewer controls for faster exploration.")
    else:
        st.slider("Green infrastructure (%)", 0, 100, step=5, key="green_infra_pct")
        st.slider("Urbanization / imperviousness (%)", 0, 100, step=5, key="urbanization_pct")

    st.divider()

    st.subheader("🔬 Display options")
    st.toggle("Show 5–95% uncertainty bands", key="show_uncertainty")
    st.toggle("Overlay IPCC SSP reference scenarios", key="show_ssp")
    st.toggle("Splice in HadCRUT5 historical record", key="show_history")
    st.toggle("Researcher mode (advanced)", key="researcher_mode",
              help="Shows additional diagnostics in the Charts tab.")

    st.divider()

    st.subheader("🎯 Challenge mode")
    st.toggle("Enable challenge", key="challenge_on")
    if st.button("🏆 Reset challenge calibration"):
        st.session_state["challenge_won"] = False
        st.session_state.pop("difficulty_choice", None)
        st.rerun()
    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty_choice")
    target_flood = int(DIFFICULTY_TARGETS[diff]["target_flood"])
    target_drought = int(DIFFICULTY_TARGETS[diff]["target_drought"])
    st.caption(f"Targets: Flood ≤ {target_flood} | Drought ≤ {target_drought}")

    st.divider()

    st.subheader("🧪 Scenario comparison")
    st.toggle("Enable comparison", key="compare_on")

    st.divider()

    st.subheader("🤖 AI assistant")
    st.caption("Optional: paste an Anthropic API key to enable streaming chat.")
    api_key_input = st.text_input(
        "Anthropic API key (optional)", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Stored only in this session. Leave blank to use the offline brain.",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

# ============================================================================
# Run simulation
# ============================================================================
df = simulate(
    years=st.session_state["years"],
    co2_ppm=st.session_state["co2_ppm"],
    rainfall_change_pct=st.session_state["rainfall_change_pct"],
    green_infra_pct=st.session_state["green_infra_pct"],
    urbanization_pct=st.session_state["urbanization_pct"],
)

flood_val = float(df["flood_risk"].iloc[-1])
drought_val = float(df["drought_risk"].iloc[-1])
temp_val = float(df["temp_anomaly_C"].iloc[-1])

# Update URL query params for shareable links
st.query_params.update(
    years=str(st.session_state["years"]),
    co2_ppm=str(st.session_state["co2_ppm"]),
    rainfall_change_pct=str(st.session_state["rainfall_change_pct"]),
    green_infra_pct=str(st.session_state["green_infra_pct"]),
    urbanization_pct=str(st.session_state["urbanization_pct"]),
)

# Lazy Monte Carlo (only computed when user wants it shown)
@st.cache_data(show_spinner=False)
def cached_mc(years, co2, rain, green, urban, n=300):
    return monte_carlo(years=years, co2_ppm=co2, rainfall_change_pct=rain,
                       green_infra_pct=green, urbanization_pct=urban, n_samples=n)

mc = None
if st.session_state.get("show_uncertainty"):
    mc = cached_mc(st.session_state["years"], st.session_state["co2_ppm"],
                   st.session_state["rainfall_change_pct"],
                   st.session_state["green_infra_pct"],
                   st.session_state["urbanization_pct"])

# ============================================================================
# Sidebar comparison buttons (need df)
# ============================================================================
with st.sidebar:
    if st.session_state.get("compare_on", False):
        b1, b2, b3 = st.columns(3)
        if b1.button("Save A"): st.session_state["scenario_A"] = snapshot(df); st.rerun()
        if b2.button("Save B"): st.session_state["scenario_B"] = snapshot(df); st.rerun()
        if b3.button("Clear"):
            st.session_state["scenario_A"] = None
            st.session_state["scenario_B"] = None
            st.rerun()
        if st.session_state.get("scenario_A"):
            st.caption("A: " + pretty_params(st.session_state["scenario_A"]["params"]))
        if st.session_state.get("scenario_B"):
            st.caption("B: " + pretty_params(st.session_state["scenario_B"]["params"]))

# ============================================================================
# Challenge banner
# ============================================================================
if st.session_state.get("challenge_on", False):
    flood_ok = flood_val <= target_flood
    drought_ok = drought_val <= target_drought
    cA, cB = st.columns(2)
    with cA: (st.success if flood_ok else st.error)("🌊 Flood OK" if flood_ok else "🌊 Flood too high")
    with cB: (st.success if drought_ok else st.error)("🌵 Drought OK" if drought_ok else "🌵 Drought too high")
    if flood_ok and drought_ok:
        if not st.session_state.get("challenge_won", False):
            st.balloons()
            st.session_state["challenge_won"] = True
    else:
        st.session_state["challenge_won"] = False

# ============================================================================
# Top metrics
# ============================================================================
def severity_class(value, kind="risk"):
    if kind == "temp":
        if value < 1.0: return "severity-low"
        if value < 2.0: return "severity-mod"
        if value < 3.0: return "severity-high"
        return "severity-vhigh"
    if value < 25: return "severity-low"
    if value < 50: return "severity-mod"
    if value < 75: return "severity-high"
    return "severity-vhigh"

# Ambient "mood" layer — tints the page based on dominant risk
if max(flood_val, drought_val) >= 75:
    mood_class = "mood-extreme"
elif flood_val >= drought_val:
    mood_class = "mood-flood"
else:
    mood_class = "mood-drought"
st.markdown(f'<div class="mood-layer {mood_class}"></div>', unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌡️ End-of-horizon warming</div>
      <div class="metric-value {severity_class(temp_val,'temp')}">{temp_val:.2f} °C</div>
      <div class="metric-trend">{'Paris-aligned' if temp_val<2 else 'Above Paris target'}</div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌊 Flood risk</div>
      <div class="metric-value {severity_class(flood_val)}">{flood_val:.0f} <span style="font-size:16px;color:#94a3b8;">/ 100</span></div>
      <div class="metric-trend">{'Low' if flood_val<25 else 'Moderate' if flood_val<50 else 'High' if flood_val<75 else 'Very high'}</div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌵 Drought risk</div>
      <div class="metric-value {severity_class(drought_val)}">{drought_val:.0f} <span style="font-size:16px;color:#94a3b8;">/ 100</span></div>
      <div class="metric-trend">{'Low' if drought_val<25 else 'Moderate' if drought_val<50 else 'High' if drought_val<75 else 'Very high'}</div>
    </div>""", unsafe_allow_html=True)
st.write("")

# ============================================================================
# Tabs
# ============================================================================
tab_charts, tab_map, tab_sens, tab_ai, tab_methods, tab_export = st.tabs(
    ["📈 Charts", "🗺️ London map", "🔬 Sensitivity", "🤖 AI assistant",
     "📚 Methods & references", "📤 Export & share"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Charts (uncertainty + SSP + history)
# ----------------------------------------------------------------------------
with tab_charts:
    # Plotly base layout — matches our dark theme
    def _plotly_layout(title, ylabel, height=420):
        return dict(
            title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                      color="#cbd5e1", size=12),
            xaxis=dict(title="Year", gridcolor="rgba(255,255,255,0.07)",
                       zerolinecolor="rgba(255,255,255,0.12)", showspikes=True,
                       spikemode="across", spikecolor="rgba(125,211,252,0.4)", spikethickness=1),
            yaxis=dict(title=ylabel, gridcolor="rgba(255,255,255,0.07)",
                       zerolinecolor="rgba(255,255,255,0.12)"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(15,23,42,0.95)",
                            bordercolor="rgba(125,211,252,0.4)",
                            font=dict(color="#e2e8f0", size=12)),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                        borderwidth=1, font=dict(size=11)),
            margin=dict(t=50, b=40, l=10, r=10),
            height=height,
            transition=dict(duration=400, easing="cubic-in-out"),
        )

    L, R = st.columns([1, 1])

    # ---------------- Temperature ----------------
    with L:
        st.subheader("🌡️ Temperature trajectory")
        fig_t = go.Figure()

        # Historical splice (HadCRUT5)
        if st.session_state["show_history"]:
            fig_t.add_trace(go.Scatter(
                x=[y for y, _ in HADCRUT_HISTORICAL],
                y=[t for _, t in HADCRUT_HISTORICAL],
                mode="lines+markers", name="HadCRUT5 (1850–2023)",
                line=dict(color="#94a3b8", width=1.6, shape="spline"),
                marker=dict(size=4, color="#94a3b8"),
                hovertemplate="<b>%{x}</b>: %{y:.2f} °C<extra>HadCRUT5</extra>",
            ))

        # Uncertainty band — drawn first so it sits behind the line
        if mc is not None:
            fig_t.add_trace(go.Scatter(
                x=mc["year"], y=mc["temp_p95"],
                mode="lines", line=dict(width=0, color="rgba(0,114,178,0)"),
                showlegend=False, hoverinfo="skip", name="p95",
            ))
            fig_t.add_trace(go.Scatter(
                x=mc["year"], y=mc["temp_p05"],
                mode="lines", line=dict(width=0, color="rgba(0,114,178,0)"),
                fill="tonexty", fillcolor="rgba(0,114,178,0.18)",
                name="5–95% Monte Carlo",
                hovertemplate="<b>%{x}</b>: 5–95% band<extra></extra>",
            ))

        # Main scenario line
        fig_t.add_trace(go.Scatter(
            x=df["year"], y=df["temp_anomaly_C"],
            mode="lines", name="Your scenario",
            line=dict(color="#7dd3fc", width=3, shape="spline"),
            hovertemplate="<b>%{x}</b>: <b>%{y:.2f} °C</b><extra>Your scenario</extra>",
        ))

        # SSP reference horizontal lines at 2100
        if st.session_state["show_ssp"]:
            for name, info in SSP_SCENARIOS.items():
                fig_t.add_hline(
                    y=info["warming_2100"],
                    line=dict(color=info["color"], width=1, dash="dot"),
                    annotation=dict(text=name, font=dict(size=9, color=info["color"]),
                                    bgcolor="rgba(15,23,42,0.6)", borderpad=2),
                    annotation_position="right",
                )

        fig_t.update_layout(**_plotly_layout("Projected warming above pre-industrial", "°C"))
        st.plotly_chart(fig_t, use_container_width=True,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # ---------------- Risks ----------------
    with R:
        st.subheader("🌊 Flood & drought risk")
        fig_r = go.Figure()

        if mc is not None:
            # Flood band
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["flood_p95"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["flood_p05"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(86,180,233,0.16)", name="Flood 5–95% band",
                hoverinfo="skip"))
            # Drought band
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["drought_p95"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["drought_p05"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(230,159,0,0.16)", name="Drought 5–95% band",
                hoverinfo="skip"))

        fig_r.add_trace(go.Scatter(
            x=df["year"], y=df["flood_risk"], mode="lines", name="🌊 Flood",
            line=dict(color="#56B4E9", width=3, shape="spline"),
            hovertemplate="<b>%{x}</b>: <b>%{y:.0f}</b>/100<extra>Flood</extra>",
        ))
        fig_r.add_trace(go.Scatter(
            x=df["year"], y=df["drought_risk"], mode="lines", name="🌵 Drought",
            line=dict(color="#E69F00", width=3, shape="spline"),
            hovertemplate="<b>%{x}</b>: <b>%{y:.0f}</b>/100<extra>Drought</extra>",
        ))

        # Challenge target lines
        if st.session_state.get("challenge_on", False):
            fig_r.add_hline(y=target_flood, line=dict(color="#56B4E9", width=1, dash="dash"),
                            annotation=dict(text=f"Flood target ≤ {target_flood}",
                                            font=dict(size=9, color="#56B4E9"),
                                            bgcolor="rgba(15,23,42,0.6)"),
                            annotation_position="left")
            fig_r.add_hline(y=target_drought, line=dict(color="#E69F00", width=1, dash="dash"),
                            annotation=dict(text=f"Drought target ≤ {target_drought}",
                                            font=dict(size=9, color="#E69F00"),
                                            bgcolor="rgba(15,23,42,0.6)"),
                            annotation_position="right")

        layout = _plotly_layout("Risk trajectories (0–100 unitless index)", "Risk index")
        layout["yaxis"]["range"] = [0, 100]
        fig_r.update_layout(**layout)
        st.plotly_chart(fig_r, use_container_width=True,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # Researcher mode: extra diagnostics
    if st.session_state["researcher_mode"]:
        st.markdown("#### Diagnostics — runoff & evaporation indices")
        figd = plt.figure(figsize=(11.0, 3.4))
        ax = figd.add_subplot(111)
        ax.plot(df["year"], df["runoff_index"], color=WONG_PALETTE["skyblue"], linewidth=2, label="Runoff index")
        ax.plot(df["year"], df["evap_index"], color=WONG_PALETTE["orange"], linewidth=2, label="Evap index")
        ax.set_xlabel("Year"); ax.set_ylabel("Unitless index")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
        figd.tight_layout()
        st.pyplot(figd, clear_figure=True)

    # Scenario comparison block
    if st.session_state.get("compare_on", False):
        A = st.session_state.get("scenario_A"); B = st.session_state.get("scenario_B")
        st.divider(); st.subheader("🧪 Scenario A vs B")
        if not A or not B:
            st.info("Save two scenarios from the sidebar to compare them here.")
        else:
            dfA, dfB = A["df"], B["df"]
            tA, fA, dA = float(dfA["temp_anomaly_C"].iloc[-1]), float(dfA["flood_risk"].iloc[-1]), float(dfA["drought_risk"].iloc[-1])
            tB, fB, dB = float(dfB["temp_anomaly_C"].iloc[-1]), float(dfB["flood_risk"].iloc[-1]), float(dfB["drought_risk"].iloc[-1])
            m1, m2, m3 = st.columns(3)
            m1.metric("Δ Warming (A → B)", f"{(tB - tA):+.2f} °C")
            m2.metric("Δ Flood risk (A → B)", f"{(fB - fA):+.0f}")
            m3.metric("Δ Drought risk (A → B)", f"{(dB - dA):+.0f}")
            cL, cR = st.columns(2)
            with cL:
                fig = plt.figure(figsize=(6.8, 4.0)); ax = fig.add_subplot(111)
                ax.plot(dfA["year"], dfA["temp_anomaly_C"], color=WONG_PALETTE["blue"], linewidth=2, label="A")
                ax.plot(dfB["year"], dfB["temp_anomaly_C"], color=WONG_PALETTE["vermilion"], linewidth=2, linestyle="--", label="B")
                ax.set_title("Temperature: A vs B"); ax.legend(frameon=False); ax.grid(True, linestyle="--", alpha=0.35)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            with cR:
                fig = plt.figure(figsize=(6.8, 4.0)); ax = fig.add_subplot(111)
                ax.plot(dfA["year"], dfA["flood_risk"], color=WONG_PALETTE["skyblue"], linewidth=2, label="Flood A")
                ax.plot(dfA["year"], dfA["drought_risk"], color=WONG_PALETTE["orange"], linewidth=2, label="Drought A")
                ax.plot(dfB["year"], dfB["flood_risk"], color=WONG_PALETTE["skyblue"], linewidth=2, linestyle="--", label="Flood B")
                ax.plot(dfB["year"], dfB["drought_risk"], color=WONG_PALETTE["orange"], linewidth=2, linestyle="--", label="Drought B")
                ax.set_ylim(0, 100); ax.set_title("Risks: A vs B"); ax.legend(frameon=False, fontsize=8); ax.grid(True, linestyle="--", alpha=0.35)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

# ----------------------------------------------------------------------------
# TAB 2 — London map
# ----------------------------------------------------------------------------
def risk_color_hex(value):
    if value < 25: return "#2ecc71"
    if value < 50: return "#f1c40f"
    if value < 75: return "#e67e22"
    return "#e74c3c"

with tab_map:
    st.subheader("London climate-risk hotspots")
    st.caption("Each marker shows how today's slider settings would play out in a real London neighbourhood. "
               "Riverside areas get higher flood weighting; low-green / high-urban areas get higher drought weighting.")
    risk_view = st.radio("Show on map:", ["Flood risk", "Drought risk", "Combined"],
                         horizontal=True, key="map_risk_view")
    show_thames = st.checkbox(
        "🌊 Show illustrative Thames flood corridor",
        value=False,
        help="Draws an indicative Thames-side high-flood-risk band. Not a substitute for the official EA Flood Map.",
    )

    m = folium.Map(location=[51.5074, -0.1278], zoom_start=11,
                   tiles="cartodbdark_matter", control_scale=True)

    map_css = """<style>
    .climate-marker { font-size: 28px; text-align: center; line-height: 1;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6));
        transform: translate(-50%, -50%); position: relative; cursor: pointer; }
    .climate-marker .ring { position: absolute; left: 50%; top: 50%;
        width: 36px; height: 36px; border-radius: 50%;
        transform: translate(-50%, -50%); pointer-events: none; }
    .pulse-flood   .ring { background: rgba(56,189,248,0.35); animation: pf 1.8s infinite cubic-bezier(0.4,0,0.6,1); }
    .pulse-drought .ring { background: rgba(251,146,60,0.35); animation: pd 1.8s infinite cubic-bezier(0.4,0,0.6,1); }
    .pulse-extreme .ring { background: rgba(248,113,113,0.40); animation: pe 1.4s infinite cubic-bezier(0.4,0,0.6,1); }
    @keyframes pf {0%{box-shadow:0 0 0 0 rgba(56,189,248,0.55);}70%{box-shadow:0 0 0 28px rgba(56,189,248,0);}100%{box-shadow:0 0 0 0 rgba(56,189,248,0);}}
    @keyframes pd {0%{box-shadow:0 0 0 0 rgba(251,146,60,0.55);}70%{box-shadow:0 0 0 28px rgba(251,146,60,0);}100%{box-shadow:0 0 0 0 rgba(251,146,60,0);}}
    @keyframes pe {0%{box-shadow:0 0 0 0 rgba(248,113,113,0.65);}70%{box-shadow:0 0 0 32px rgba(248,113,113,0);}100%{box-shadow:0 0 0 0 rgba(248,113,113,0);}}
    .climate-legend { position: fixed; bottom: 24px; left: 24px; z-index:9999;
        background: rgba(15,23,42,0.92); color: #e2e8f0; padding: 10px 14px;
        border-radius: 10px; font-size: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.08);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .leaflet-popup-content-wrapper { background: rgba(15,23,42,0.96); color: #e2e8f0;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
    .leaflet-popup-tip { background: rgba(15,23,42,0.96); }
    </style>"""
    m.get_root().html.add_child(folium.Element(map_css))

    # Optional Thames flood corridor (illustrative polyline)
    if show_thames:
        thames_corridor = [
            (51.485, -0.30), (51.488, -0.22), (51.490, -0.16), (51.498, -0.115),
            (51.500, -0.08), (51.505, -0.02), (51.499,  0.005), (51.494,  0.04),
            (51.485,  0.07),
        ]
        folium.PolyLine(thames_corridor, color="#38bdf8", weight=10, opacity=0.45,
                        tooltip="Illustrative Thames flood corridor (not the official EA flood map)"
                        ).add_to(m)

    heat_points = []
    for loc in LONDON_LOCATIONS:
        f_local, d_local = local_risk(loc, flood_val, drought_val,
                                      st.session_state["green_infra_pct"],
                                      st.session_state["urbanization_pct"])
        if risk_view == "Flood risk":   shown, kind = f_local, "flood"
        elif risk_view == "Drought risk": shown, kind = d_local, "drought"
        else:
            shown = (f_local + d_local) / 2
            kind = "flood" if f_local >= d_local else "drought"

        if kind == "flood":
            emoji = "💧" if shown < 50 else "🌊"
            pulse = "pulse-flood" if shown >= 45 else ""
        else:
            emoji = "🌵" if shown < 50 else "🔥"
            pulse = "pulse-drought" if shown >= 45 else ""
        if shown >= 75:
            pulse, emoji = "pulse-extreme", "⚠️"

        size = int(28 + (shown / 100.0) * 22)
        marker_html = f"""
        <div class="climate-marker {pulse}" style="font-size:{size}px;">
            <div class="ring"></div>
            <span style="position:relative;z-index:2;">{emoji}</span>
        </div>"""
        popup_html = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif; min-width:230px;">
          <div style="font-weight:700; font-size:14px; margin-bottom:6px;">{loc['name']}</div>
          <div style="font-size:11px; color:#94a3b8; margin-bottom:6px;">Borough: {loc.get('borough','—')}</div>
          <div style="display:flex; justify-content:space-between; padding:4px 0; border-top:1px solid rgba(255,255,255,0.1);">
            <span>🌊 Flood risk</span><b style="color:#7dd3fc;">{f_local:.0f}/100</b>
          </div>
          <div style="display:flex; justify-content:space-between; padding:4px 0; border-top:1px solid rgba(255,255,255,0.1);">
            <span>🌵 Drought risk</span><b style="color:#fbbf24;">{d_local:.0f}/100</b>
          </div>
          <div style="margin-top:8px; font-size:11px; color:#94a3b8;">
            River exposure {int(loc['river']*100)}% · green {int(loc['green']*100)}% · urban {int(loc['urban']*100)}%
          </div>
        </div>"""

        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            icon=folium.DivIcon(html=marker_html, icon_size=(size, size),
                                icon_anchor=(size//2, size//2)),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{loc['name']} — {risk_view}: {shown:.0f}/100",
        ).add_to(m)
        heat_points.append([loc["lat"], loc["lon"], shown / 100.0])

    HeatMap(heat_points, radius=42, blur=32, min_opacity=0.3).add_to(
        folium.FeatureGroup(name="🔥 Heat halo", show=False).add_to(m)
    )
    folium.LayerControl(collapsed=True).add_to(m)
    legend_html = """<div class="climate-legend">
      <b>🌍 Climate-risk legend</b><br>
      <span style="font-size:14px;">💧</span> mild &nbsp; <span style="font-size:14px;">🌊</span> high flood<br>
      <span style="font-size:14px;">🌵</span> mild &nbsp; <span style="font-size:14px;">🔥</span> high drought<br>
      <span style="font-size:14px;">⚠️</span> very high risk (pulses)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    map_key = (f"map_{risk_view}_{int(flood_val)}_{int(drought_val)}"
               f"_{st.session_state['green_infra_pct']}_{st.session_state['urbanization_pct']}"
               f"_{st.session_state['co2_ppm']}_{int(show_thames)}")
    st_folium(m, width=None, height=540, returned_objects=[], key=map_key)

    st.caption("Location risk profiles are illustrative for the festival demo, not an official hazard map. "
               "For authoritative London flood data see the [Environment Agency Flood Map for Planning](https://flood-map-for-planning.service.gov.uk/).")

# ----------------------------------------------------------------------------
# TAB 3 — Sensitivity (tornado)
# ----------------------------------------------------------------------------
with tab_sens:
    st.subheader("🔬 Parameter sensitivity (one-at-a-time, ±20%)")
    st.caption("How much would the end-of-horizon outputs change if you nudged each lever 20% in either direction? "
               "This is a tornado plot — the longest bars are the levers that matter most.")
    target_metric = st.radio("Sensitivity of:", ["Flood risk", "Drought risk", "Warming"],
                             horizontal=True, key="sens_target")

    sens = sensitivity_analysis(
        years=st.session_state["years"], co2_ppm=st.session_state["co2_ppm"],
        rainfall_change_pct=st.session_state["rainfall_change_pct"],
        green_infra_pct=st.session_state["green_infra_pct"],
        urbanization_pct=st.session_state["urbanization_pct"],
    )
    col_map = {"Flood risk": "delta_flood", "Drought risk": "delta_drought", "Warming": "delta_temp"}
    col = col_map[target_metric]
    pivot = sens.pivot(index="parameter", columns="direction", values=col).reindex(
        columns=["−20%", "+20%"]
    )
    # Order by total |effect|
    pivot["abs"] = pivot.abs().sum(axis=1)
    pivot = pivot.sort_values("abs", ascending=True).drop(columns="abs")
    pretty_names = {
        "co2_ppm": "CO₂ concentration",
        "rainfall_change_pct": "Rainfall change",
        "green_infra_pct": "Green infrastructure",
        "urbanization_pct": "Urbanization",
    }
    pivot.index = [pretty_names.get(p, p) for p in pivot.index]

    figs = plt.figure(figsize=(9.5, 4.4))
    ax = figs.add_subplot(111)
    y = np.arange(len(pivot))
    ax.barh(y, pivot["−20%"], color=WONG_PALETTE["skyblue"], label="−20%")
    ax.barh(y, pivot["+20%"], color=WONG_PALETTE["vermilion"], label="+20%")
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(pivot.index)
    unit = "°C" if target_metric == "Warming" else "points (0–100)"
    ax.set_xlabel(f"Δ {target_metric} ({unit})")
    ax.set_title(f"Sensitivity tornado — {target_metric}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    figs.tight_layout()
    st.pyplot(figs, clear_figure=True)

    with st.expander("Show raw sensitivity table"):
        st.dataframe(sens, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4 — AI assistant (streaming + offline brain)
# ----------------------------------------------------------------------------
def get_anthropic_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k: return k
    try: return (st.secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
    except Exception: return ""


def offline_explainer(temp, flood, drought, params):
    bits = []
    bits.append("warming stays modest" if temp < 1 else "warming reaches a noticeable level"
                if temp < 2 else "warming climbs into a dangerous range")
    bits.append("flood risk becomes severe" if flood > 70 else
                "flood risk is moderate but rising" if flood > 40 else
                "flood risk stays relatively contained")
    bits.append("drought stress is severe" if drought > 70 else
                "drought risk is moderate" if drought > 40 else
                "drought risk stays low")
    levers = []
    if params["green_infra_pct"] < 30: levers.append("Adding more green infrastructure (parks, green roofs, SuDS) is the biggest unused lever.")
    if params["urbanization_pct"] > 60: levers.append("Heavy urbanization makes runoff worse — permeable pavements and tree pits would help.")
    if params["co2_ppm"] > 500: levers.append("CO₂ is high — this drives the warming term and amplifies every other risk.")
    if not levers: levers.append("Settings are already in a fairly resilient zone — try pushing CO₂ down further.")
    return (f"Over {params['years']} years with CO₂ at {params['co2_ppm']} ppm, "
            f"green infrastructure at {params['green_infra_pct']}%, and "
            f"urbanization at {params['urbanization_pct']}%, the simulator predicts that "
            + ", ".join(bits) + ".\n\n**What you could try next:**\n- " + "\n- ".join(levers))


def offline_answer(question, temp, flood, drought, params):
    q = (question or "").lower().strip()
    if not q: return offline_explainer(temp, flood, drought, params)
    asks_where = any(w in q for w in ["where","which area","which location","which place",
                                       "which neighbourhood","which neighborhood","what area",
                                       "what part","highest","worst","most"])
    asks_flood = "flood" in q
    asks_drought = "drought" in q
    asks_why = any(w in q for w in ["why","reason","cause","because"])
    asks_reduce = any(w in q for w in ["reduce","lower","decrease","fix","improve",
                                        "how to","how can","how do","what should","solve"])
    asks_explain = any(w in q for w in ["explain","what does","what is","what's","tell me about","meaning","mean by"])

    local_data = []
    for loc in LONDON_LOCATIONS:
        fl, dr = local_risk(loc, flood, drought, params["green_infra_pct"], params["urbanization_pct"])
        local_data.append({"name": loc["name"], "flood": fl, "drought": dr})

    if asks_where and (asks_flood or asks_drought or "risk" in q):
        if asks_flood and not asks_drought:
            top = sorted(local_data, key=lambda x: -x["flood"])[:3]
            lines = ["**Areas with the highest flood risk in your scenario:**"]
            for i, item in enumerate(top, 1): lines.append(f"{i}. {item['name']} — {item['flood']:.0f} / 100")
            lines.append("\nRiverside zones (Westminster, Canary Wharf, Greenwich) and dense urban areas with little green cover get the worst flood scores.")
            return "\n".join(lines)
        if asks_drought and not asks_flood:
            top = sorted(local_data, key=lambda x: -x["drought"])[:3]
            lines = ["**Areas with the highest drought risk in your scenario:**"]
            for i, item in enumerate(top, 1): lines.append(f"{i}. {item['name']} — {item['drought']:.0f} / 100")
            lines.append("\nDrought hits dense, low-green areas hardest. Adding parks, green roofs and street trees would help most here.")
            return "\n".join(lines)
        top = sorted(local_data, key=lambda x: -((x["flood"] + x["drought"]) / 2))[:3]
        lines = ["**Areas with the highest combined climate risk:**"]
        for i, item in enumerate(top, 1): lines.append(f"{i}. {item['name']} — flood {item['flood']:.0f}, drought {item['drought']:.0f}")
        return "\n".join(lines)

    if asks_why and asks_flood:
        reasons = []
        if params["urbanization_pct"] > 50: reasons.append(f"high urbanization ({params['urbanization_pct']}%) — concrete and roads stop water soaking in")
        if params["green_infra_pct"] < 30: reasons.append(f"low green infrastructure ({params['green_infra_pct']}%) — not enough parks / SuDS to absorb runoff")
        if params["rainfall_change_pct"] > 5: reasons.append(f"rainfall is up by {params['rainfall_change_pct']}% — more water falling overall")
        if params["co2_ppm"] > 450: reasons.append(f"CO₂ at {params['co2_ppm']} ppm drives warming, which intensifies extreme rain events")
        if not reasons: return f"Flood risk is **{flood:.0f}/100** — fairly contained for your settings."
        return f"Flood risk is **{flood:.0f}/100**. Main drivers in your scenario:\n\n- " + "\n- ".join(reasons)

    if asks_why and asks_drought:
        reasons = []
        if params["co2_ppm"] > 450: reasons.append(f"CO₂ at {params['co2_ppm']} ppm drives warming, which boosts evaporation and dries soils")
        if params["rainfall_change_pct"] < 0: reasons.append(f"rainfall is down by {abs(params['rainfall_change_pct'])}% — less water available")
        if params["green_infra_pct"] < 30: reasons.append(f"low green infrastructure ({params['green_infra_pct']}%) — soils dry out faster without vegetation")
        if params["urbanization_pct"] > 50: reasons.append(f"heavy urbanization ({params['urbanization_pct']}%) creates urban heat islands that worsen drought stress")
        if not reasons: return f"Drought risk is **{drought:.0f}/100** — under control for your settings."
        return f"Drought risk is **{drought:.0f}/100**. Main drivers in your scenario:\n\n- " + "\n- ".join(reasons)

    if asks_reduce:
        if asks_flood and not asks_drought:
            return (f"To lower flood risk (currently **{flood:.0f}/100**):\n\n"
                    "- Push **green infrastructure** up — most effective lever (60–80%).\n"
                    "- Reduce **urbanization** — replace concrete with permeable surfaces.\n"
                    "- Lower **CO₂** — less warming means less intense rainfall.")
        if asks_drought and not asks_flood:
            return (f"To lower drought risk (currently **{drought:.0f}/100**):\n\n"
                    "- Lower **CO₂** — slower warming means less evaporation.\n"
                    "- Add **green infrastructure** — soils retain moisture and shade reduces heat stress.\n"
                    "- Reduce dense urbanization to weaken the urban heat island.")
        return ("**Three strongest levers to reduce climate risk:**\n\n"
                f"1. Lower CO₂ (you're at {params['co2_ppm']} ppm — try 350–400).\n"
                f"2. Boost green infrastructure (you're at {params['green_infra_pct']}% — try 60+).\n"
                f"3. Reduce urbanization (you're at {params['urbanization_pct']}% — try 30–40).")

    if asks_explain:
        if "co2" in q or "co₂" in q or "carbon" in q:
            return (f"**CO₂ concentration** — atmospheric carbon dioxide in parts per million. "
                    f"Pre-industrial = 280 ppm, today ≈ 425 ppm. You've set **{params['co2_ppm']} ppm**.")
        if "green" in q or "infrastructure" in q or "suds" in q:
            return (f"**Green infrastructure** = parks, green roofs, street trees, SuDS, permeable pavements. "
                    f"You've set **{params['green_infra_pct']}%**.")
        if "urban" in q or "impervious" in q:
            return (f"**Urbanization / imperviousness** = how much of the city is buildings, roads and concrete. "
                    f"You're at **{params['urbanization_pct']}%**. Higher = worse flooding and worse drought.")
        if "flood" in q: return f"**Flood risk** is a 0–100 unitless index. You're at **{flood:.0f}/100**."
        if "drought" in q: return f"**Drought risk** is a 0–100 unitless index. You're at **{drought:.0f}/100**."
        if "warming" in q or "temperature" in q or "heat" in q:
            return f"**Warming (proxy °C)** is the projected end-of-horizon anomaly above pre-industrial. You're at **{temp:.2f} °C**."

    return (offline_explainer(temp, flood, drought, params)
            + "\n\n_Tip: try asking_ *\"which area has the highest flood risk?\"*, "
            "*\"why is drought so high?\"*, *\"how do I reduce flood risk?\"*, or *\"what does CO₂ mean?\"*.")


def ask_anthropic_stream(messages, system_prompt, placeholder):
    from anthropic import Anthropic
    api_key = get_anthropic_key()
    if not api_key: raise RuntimeError("No Anthropic API key set.")
    client = Anthropic(api_key=api_key)
    text_so_far = ""
    with client.messages.stream(model="claude-haiku-4-5-20251001",
                                max_tokens=600, system=system_prompt,
                                messages=messages) as stream:
        for chunk in stream.text_stream:
            text_so_far += chunk
            placeholder.markdown(text_so_far + " ▌")
    placeholder.markdown(text_so_far)
    return text_so_far


with tab_ai:
    online_now = bool(get_anthropic_key())
    chip = ('<span class="ai-chip online">● ONLINE · Claude Haiku 4.5</span>'
            if online_now else
            '<span class="ai-chip offline">● OFFLINE · built-in brain</span>')
    st.markdown(f"### 🤖 Ask the climate engineer {chip}", unsafe_allow_html=True)
    if online_now:
        st.caption("Live AI answers, streamed token-by-token, grounded in your current sliders.")
    else:
        st.caption("Running on the built-in explainer. Paste an Anthropic API key in the sidebar for streaming live AI.")
        with st.expander("🔑 How to enable streaming live AI (≈30 seconds)"):
            st.markdown(
                "1. Go to **https://console.anthropic.com/** and sign in.\n"
                "2. Click **Get API keys** → **Create Key**.\n"
                "3. Copy the key (starts with `sk-ant-...`).\n"
                "4. Paste it in the **🤖 AI assistant** field in the sidebar.\n\n"
                "For Streamlit Cloud, go to *Manage app → Settings → Secrets* and add:\n"
                "`ANTHROPIC_API_KEY = \"sk-ant-...\"`")

    current_params = {k: int(st.session_state[k]) for k in
                      ("years", "co2_ppm", "rainfall_change_pct", "green_infra_pct", "urbanization_pct")}

    with st.container(border=True):
        st.markdown("**📝 Auto-summary of your current scenario**")
        st.write(offline_explainer(temp_val, flood_val, drought_val, current_params))

    st.divider()
    st.markdown("**💬 Chat**")

    is_kids = st.session_state["mode"] == "Kids (simple)"
    is_researcher = st.session_state["researcher_mode"]
    if is_kids:
        tone = "Use a playful, simple tone for children aged 8-12. Use vivid metaphors."
    elif is_researcher:
        tone = ("Use precise, scientific language. Always cite which parameter is responsible for an effect. "
                "If asked for absolute risk, refuse and explain the model is a simplified 0-D index. "
                "When relevant, mention the IPCC AR6 SSP scenarios as anchors. Never overstate certainty.")
    else:
        tone = "Use clear, accessible language for a general adult audience."

    system_prompt = (
        "You are a careful climate engineering tutor at the Great Exhibition Road Festival "
        "in London. You help visitors understand a SIMPLIFIED 0-D simulator (NOT a GCM). "
        "Keep replies short (under ~120 words), concrete, and grounded in the user's CURRENT "
        "simulator settings provided below. Avoid alarmism. " + tone +
        f"\n\nCURRENT SCENARIO:\n"
        f"- horizon: {current_params['years']} years\n"
        f"- CO2: {current_params['co2_ppm']} ppm\n"
        f"- rainfall change: {current_params['rainfall_change_pct']}%\n"
        f"- green infrastructure: {current_params['green_infra_pct']}%\n"
        f"- urbanization: {current_params['urbanization_pct']}%\n"
        f"- end-of-horizon warming: {temp_val:.2f} °C\n"
        f"- end-of-horizon flood risk: {flood_val:.0f}/100\n"
        f"- end-of-horizon drought risk: {drought_val:.0f}/100\n"
    )

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question, e.g. 'Why is flood risk so high?'")
    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_🌍 Thinking..._")
            try:
                if get_anthropic_key():
                    answer = ask_anthropic_stream(
                        messages=[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state["chat_history"]],
                        system_prompt=system_prompt, placeholder=placeholder,
                    )
                else:
                    raise RuntimeError("no_api_key")
            except Exception as e:
                smart = offline_answer(user_q, temp_val, flood_val, drought_val, current_params)
                answer = smart if str(e) == "no_api_key" else f"_(Live AI hiccup: {e}.)_\n\n{smart}"
                placeholder.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    if st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()

# ----------------------------------------------------------------------------
# TAB 5 — Methods & references
# ----------------------------------------------------------------------------
with tab_methods:
    st.subheader("📚 Methods, equations & references")
    st.markdown(f"**Model version:** `v{MODEL_VERSION}`")

    st.markdown("#### Equations")
    st.markdown('<div class="eq">ΔT(t) = λ · ln(C / C₀) · (1 − exp(−t / τ))</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq">R(t) = (1 + δp) · (a + b·U) · (1 − c·G) · (1 + d·ΔT(t))</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq">F(t) = 100 · (1 − exp(−k_F · R(t)))</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq">E(t) = (1 + e·ΔT(t)) / (1 + δp) · (1 − f·G)</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq">D(t) = 100 · (1 − exp(−k_D · E(t)))</div>', unsafe_allow_html=True)
    st.caption("U = urbanization fraction; G = green-infrastructure fraction; δp = fractional rainfall change; "
               "ΔT(t) = warming above pre-industrial.")

    st.markdown("#### Default coefficients")
    st.dataframe(pd.DataFrame.from_dict(DEFAULT_PARAMS, orient="index", columns=["value"]),
                 use_container_width=True)

    st.markdown("#### Limitations")
    st.markdown(
        "- This is a **0-D educational simulator**, not a GCM.\n"
        "- Coefficients are illustrative — calibrated for demonstration behaviour, **not** validated "
        "against gauge or reanalysis data.\n"
        "- Risks are unitless 0–100 indices, **not** flood return periods, drought severity classes, "
        "or insurance loss probabilities.\n"
        "- Spatial effects are crude: location-level risk is a multiplicative modulation of the "
        "global simulator output by hand-tuned river/green/urban factors.\n"
        "- No representation of: ocean circulation, ice sheet feedbacks, regional precipitation "
        "patterns, climate tipping points, demographic change, or adaptation feedbacks.\n"
        "- **Do not use for policy, planning, insurance, or operational decisions.**"
    )

    st.markdown("#### References")
    for r in REFERENCES:
        st.markdown(f"""<div class="refs-card">
          <span class="refs-key">[{r['key']}]</span>{r['text']}
          <div class="refs-doi">{r['doi']}</div>
        </div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 6 — Export & share
# ----------------------------------------------------------------------------
with tab_export:
    st.subheader("📤 Export & share")
    st.caption("Reproducibility: every export below carries the model version and the full parameter manifest.")

    current_params = {
        "years": int(st.session_state["years"]),
        "co2_ppm": int(st.session_state["co2_ppm"]),
        "rainfall_change_pct": int(st.session_state["rainfall_change_pct"]),
        "green_infra_pct": int(st.session_state["green_infra_pct"]),
        "urbanization_pct": int(st.session_state["urbanization_pct"]),
    }

    # Shareable URL
    qs = "&".join(f"{k}={v}" for k, v in current_params.items())
    share_url = f"?{qs}"
    st.markdown("**🔗 Shareable scenario link**")
    st.code(share_url, language=None)
    st.caption("Append this to your deployed app URL. Anyone who opens it will see the exact same scenario.")

    st.markdown("---")

    cdl1, cdl2, cdl3 = st.columns(3)

    # CSV export
    with cdl1:
        st.markdown("**📊 Time series (CSV)**")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download scenario.csv", data=csv_bytes,
                           file_name=f"scenario_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv", use_container_width=True)

    # JSON manifest
    with cdl2:
        st.markdown("**🧾 Parameter manifest (JSON)**")
        manifest = {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now().isoformat(),
            "parameters": current_params,
            "default_coefficients": DEFAULT_PARAMS,
            "end_of_horizon": {
                "warming_C": round(temp_val, 3),
                "flood_risk": round(flood_val, 1),
                "drought_risk": round(drought_val, 1),
            },
            "share_url_query": qs,
        }
        json_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        st.download_button("Download manifest.json", data=json_bytes,
                           file_name=f"manifest_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", use_container_width=True)

    # PDF report
    with cdl3:
        st.markdown("**📄 One-page PDF report**")
        if st.button("Generate PDF", use_container_width=True):
            try:
                pdf_bytes = build_report(df, mc, current_params,
                                         scenario_label=st.session_state.get("scenario_pick", "Custom scenario"))
                st.session_state["_pdf_bytes"] = pdf_bytes
                st.success("PDF ready below ⬇")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        if st.session_state.get("_pdf_bytes"):
            st.download_button("Download report.pdf", data=st.session_state["_pdf_bytes"],
                               file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf", use_container_width=True)

    st.markdown("---")
    st.markdown("**🎓 Cite this app**")
    st.code(
        f'Mohammad, A. ({datetime.now().year}). London Climate Engineering Simulator (v{MODEL_VERSION}). '
        f'Imperial College London. https://climate-simulator.streamlit.app/',
        language=None,
    )

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption(
    f"Developed by Ashikujjaman Mohammad · MSc Environmental Engineering · Imperial College London · "
    f"Model v{MODEL_VERSION} · 2026"
)
