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
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
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
/* === HIDE STREAMLIT CHROME (Share / Star / GitHub / 3-dot menu / deploy bar) ===
   Critical: do NOT hide the header itself or any "header" buttons globally —
   the sidebar collapse/expand toggle lives there. Hide only the right-side
   action chrome. */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stAppDeployButton, .stDeployButton { display: none !important; }
.viewerBadge_container__1QSob, .viewerBadge_link__qRIco { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* Header itself stays in the DOM (so the sidebar collapse toggle can render
   inside it) but it's transparent and slim so it doesn't draw attention. */
header[data-testid="stHeader"] {
  background: transparent !important;
  height: 32px !important;
  border: none !important;
  box-shadow: none !important;
}
header[data-testid="stHeader"] > * { background: transparent !important; }

/* === SIDEBAR — visible by default, but collapsible on click ====
   Apply width + visibility ONLY when sidebar is expanded (or has
   no aria-expanded attribute set yet, i.e. fresh page load).
   When the user clicks the collapse arrow, Streamlit sets
   aria-expanded="false" — we don't override that case, so the
   collapse animation runs naturally. */
html body section[data-testid="stSidebar"][aria-expanded="true"],
html body section[data-testid="stSidebar"]:not([aria-expanded]) {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: 21rem !important;
  min-width: 21rem !important;
  transform: translateX(0) !important;
}
/* Children must also be visible when expanded */
html body section[data-testid="stSidebar"][aria-expanded="true"] > div,
html body section[data-testid="stSidebar"]:not([aria-expanded]) > div {
  display: block !important;
  visibility: visible !important;
}
/* Always keep the collapse/expand toggle reachable in BOTH states */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[aria-label="Open sidebar"],
[aria-label="Close sidebar"] {
  display: flex !important;
  visibility: visible !important;
  z-index: 99999 !important;
}

.block-container { padding-top: 0.5rem !important; }

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

/* Slim header replacing the bulky hero */
.slim-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 14px 20px;
    margin-bottom: 12px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(46,134,222,0.12), rgba(238,90,36,0.10));
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    flex-wrap: wrap;
    animation: fadeInUp 0.7s cubic-bezier(0.2,0.8,0.2,1);
}
.slim-icons {
    font-size: 22px;
    letter-spacing: 6px;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.4));
}
.slim-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #fbbf24;
    background: rgba(251,191,36,0.10);
    border: 1px solid rgba(251,191,36,0.35);
    border-radius: 999px;
    padding: 4px 12px;
}
.slim-tagline { font-size: 13px; color: #cbd5e1; flex: 1; min-width: 200px; }
.slim-tagline .ts  { color: #7dd3fc; font-weight: 600; }
.slim-tagline .ts2 { color: #fbbf24; font-weight: 600; }
.slim-tagline .ts3 { color: #f87171; font-weight: 700; }
.slim-tagline .sep { color: rgba(255,255,255,0.25); margin: 0 8px; }

/* ================================================================
   COMPACT METRIC STRIP — replaces tall cards so globe fits above-fold
   ================================================================ */
.metric-strip {
    display: flex;
    gap: 10px;
    margin: 4px 0 12px;
    flex-wrap: wrap;
}
.metric-pill {
    flex: 1;
    min-width: 180px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 14px rgba(0,0,0,0.22);
    backdrop-filter: blur(6px);
    animation: fadeInUp 0.55s cubic-bezier(0.2,0.8,0.2,1) both;
}
.metric-pill:nth-of-type(2) { animation-delay: 0.08s; }
.metric-pill:nth-of-type(3) { animation-delay: 0.16s; }
.metric-pill .mp-icon { font-size: 22px; line-height: 1; opacity: 0.95; }
.metric-pill .mp-body { display: flex; flex-direction: column; line-height: 1.05; }
.metric-pill .mp-label { font-size: 10px; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; }
.metric-pill .mp-value { font-size: 22px; font-weight: 800; margin-top: 2px; }
.metric-pill .mp-trend { font-size: 11px; color: #94a3b8; }

/* ================================================================
   PRESET CHIPS — top-of-sidebar one-click scenarios
   ================================================================ */
.preset-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
    margin: 4px 0 10px;
}
section[data-testid="stSidebar"] .preset-row .stButton button {
    width: 100%;
    padding: 8px 4px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: 999px !important;
    border: 1px solid rgba(125,211,252,0.25) !important;
    background: rgba(125,211,252,0.06) !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .preset-row .stButton button:hover {
    background: linear-gradient(135deg, rgba(125,211,252,0.18), rgba(251,191,36,0.18)) !important;
    border-color: rgba(251,191,36,0.4) !important;
}

/* ================================================================
   LOCATION SPLASH CARD — appears next to globe when a marker clicked
   ================================================================ */
.loc-splash {
    border-radius: 14px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(46,134,222,0.16), rgba(238,90,36,0.10));
    border: 1px solid rgba(125,211,252,0.30);
    box-shadow: 0 6px 22px rgba(0,0,0,0.30);
    margin-bottom: 10px;
    animation: splashIn 0.5s cubic-bezier(0.2,0.8,0.2,1);
}
@keyframes splashIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.loc-splash .ls-eyebrow { font-size: 10px; letter-spacing: 0.18em; color: #fbbf24; text-transform: uppercase; }
.loc-splash .ls-name { font-size: 20px; font-weight: 800; margin: 2px 0 4px; color: #e2e8f0; }
.loc-splash .ls-borough { font-size: 12px; color: #94a3b8; }
.loc-splash .ls-row {
    display: flex;
    gap: 14px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.loc-splash .ls-stat { flex: 1; }
.loc-splash .ls-stat .lsv { font-size: 18px; font-weight: 800; }
.loc-splash .ls-stat .lsl { font-size: 10px; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;}

/* ================================================================
   SIDEBAR — expander polish + sparkline tightness
   ================================================================ */
section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] details summary {
    background: rgba(125,211,252,0.04);
    border-radius: 10px;
    padding: 6px 10px;
    font-weight: 700;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] details[open] summary {
    border-color: rgba(125,211,252,0.35);
}
.spark-row { margin: -4px 0 6px; padding: 0 4px; }

/* Tighten block container — globe fits above the fold */
.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SPLASH SCREEN — game-style intro
# ============================================================================
if not st.session_state.get("splash_shown", False):
    # Hide sidebar + header for a true full-screen takeover.
    # Scope with body:has(.splash) so the rules only apply while the splash
    # element is in the DOM — they vanish the moment splash is dismissed.
    st.markdown("""
    <style>
    body:has(.splash) section[data-testid="stSidebar"] { display: none !important; }
    body:has(.splash) header[data-testid="stHeader"]   { display: none !important; }
    body:has(.splash) div[data-testid="stToolbar"]     { display: none !important; }
    /* Welcome page fills the viewport but keeps Streamlit's natural
       horizontal padding so the card is FRAMED with side margins —
       not stretched edge-to-edge. */
    body:has(.splash) .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        /* DO NOT override max-width or left/right padding —
           Streamlit's defaults give the card its left/right breathing room */
        min-height: 100vh !important;
    }
    body:has(.splash) [data-testid="stMain"],
    body:has(.splash) [data-testid="stAppViewContainer"] {
        min-height: 100vh !important;
    }

    .splash {
        position: relative;
        /* Card sized so card + button + credit together fill the viewport.
           ~82vh card + 2vh top + ~14vh button/credit = ~98vh total, no empty bottom. */
        min-height: 82vh;
        padding: 48px 28px 36px;
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
        margin-bottom: 24px;
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
        margin-bottom: 8px;
        animation: fadeInUp 0.8s ease 1.3s both;
    }
    .splash-tagline .ts { color: #7dd3fc; }
    .splash-tagline .ts2 { color: #fbbf24; }
    .splash-tagline .ts3 { color: #f87171; font-weight: 700; }
    .splash-tagline .sep { color: rgba(255,255,255,0.25); margin: 0 10px; }

    .splash-prompt {
        font-size: 10px; color: #64748b; letter-spacing: 0.18em; text-transform: uppercase;
        margin-top: 8px;
        animation: fadeIn 0.8s ease 1.6s both, blinkSoft 2s ease-in-out infinite 2s;
    }
    @keyframes blinkSoft { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }

    /* Style the Streamlit "Enter the Lab" button when on splash */
    .splash + div .stButton button {
        background: linear-gradient(135deg, #7dd3fc 0%, #fbbf24 50%, #f87171 100%) !important;
        background-size: 200% 100% !important;
        color: #0b1220 !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        letter-spacing: 0.1em !important;
        padding: 11px 28px !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 32px rgba(125,211,252,0.4), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
        animation: shimmer 5s linear infinite, btnPulse 2.4s ease-in-out infinite, fadeInUp 0.9s ease 1.5s both !important;
        transition: transform 0.18s ease !important;
        margin-top: 10px !important;
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
        margin-top: 12px;
        font-size: 10px;
        color: #475569;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        animation: fadeIn 1s ease 2s both;
        line-height: 1.4;
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

    st.markdown(f"""
    <div style="text-align:center;">
      <div class="splash-prompt">▼  Click to begin  ▼</div>
      <div class="splash-credit">
        Developed by Ashikujjaman Mohammad · MSc Environmental Engineering · Imperial College London · Model v{MODEL_VERSION} · 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================================
# MAIN APP — only reached after splash dismissed
# ============================================================================
# Hero removed — the splash already covers the title. We go straight to metrics.

# Force sidebar open on every page load. Streamlit's `initial_sidebar_state`
# only applies once per session; if the browser remembers a collapsed state
# (or auto-collapsed on a narrow viewport at any point) it stays collapsed.
# This injected JS finds the "open sidebar" toggle and clicks it if visible.
components.html("""
<script>
(function() {
  const tryExpand = (attempt = 0) => {
    if (attempt > 30) return;  // give up after ~9s
    const doc = window.parent.document;
    const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
    // If sidebar is collapsed, the "open" toggle is rendered separately
    const openBtn = doc.querySelector(
      '[data-testid="stSidebarCollapsedControl"] button, ' +
      '[data-testid="collapsedControl"] button, ' +
      'button[kind="header"][aria-label*="ide"], ' +
      'button[aria-label="Open sidebar"]'
    );
    // Sidebar present and visibly open? Done.
    if (sidebar && sidebar.getBoundingClientRect().width > 50) return;
    if (openBtn) { openBtn.click(); return; }
    setTimeout(() => tryExpand(attempt + 1), 300);
  };
  tryExpand();
})();
</script>
""", height=0)

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
# Sidebar — presets first, sliders grouped, advanced collapsed
# ============================================================================

# Preset chips: one-click scenarios for casual users (no slider fiddling)
PRESET_CHIPS = {
    "🌱 Net-zero": {
        "co2_ppm": 380, "rainfall_change_pct": 5,
        "green_infra_pct": 70, "urbanization_pct": 35,
    },
    "🔥 Worst-case": {
        "co2_ppm": 750, "rainfall_change_pct": 25,
        "green_infra_pct": 10, "urbanization_pct": 75,
    },
    "🌳 Adaptation": {
        "co2_ppm": 500, "rainfall_change_pct": 12,
        "green_infra_pct": 80, "urbanization_pct": 50,
    },
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert a #RRGGBB hex string to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _spark(values, color="#7dd3fc", height=28):
    """Tiny inline sparkline showing how a parameter behaves over the horizon."""
    fig = go.Figure(go.Scatter(
        y=list(values), mode="lines",
        line=dict(color=color, width=1.6, shape="spline"),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.18),
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


with st.sidebar:
    st.header("Controls")

    # ---- 1. PRESET CHIPS — fast scenarios at the very top ---------
    st.markdown("**🎛️ Quick presets**")
    st.markdown('<div class="preset-row">', unsafe_allow_html=True)
    pcol1, pcol2, pcol3 = st.columns(3)
    preset_names = list(PRESET_CHIPS.keys())
    for col, name in zip([pcol1, pcol2, pcol3], preset_names):
        with col:
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                for k, v in PRESET_CHIPS[name].items():
                    st.session_state[k] = v
                st.session_state["challenge_won"] = False
                st.session_state.pop("selected_location_idx", None)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- 2. CLIMATE INPUTS — expander group --------------------
    with st.expander("🌡️ Climate inputs", expanded=True):
        st.radio("Mode", ["Standard", "Kids (simple)"],
                 key="mode", horizontal=True, label_visibility="collapsed")
        st.slider("⏱️ Horizon (years)", 20, 120, step=5, key="years")
        st.slider("💨 CO₂ (ppm)", 280, 900, step=10, key="co2_ppm")
        # CO2 sparkline — projected ramp up under current settings
        _co2_now = int(st.session_state["co2_ppm"])
        _co2_path = np.linspace(420, _co2_now, 40)
        st.markdown('<div class="spark-row">', unsafe_allow_html=True)
        st.plotly_chart(_spark(_co2_path, "#fbbf24"), use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True},
                        key="spark_co2")
        st.markdown('</div>', unsafe_allow_html=True)
        st.slider("☔ Rainfall change (%)", -30, 50, step=1, key="rainfall_change_pct")

    # ---- 3. LONDON / INFRASTRUCTURE inputs ---------------------
    with st.expander("🏙️ London infrastructure", expanded=True):
        if st.session_state["mode"] == "Kids (simple)":
            st.slider("🌳 Green solutions (%)", 0, 100, step=5, key="green_infra_pct")
            st.session_state["urbanization_pct"] = 45
            st.caption("Kids mode hides urbanization for faster exploration.")
        else:
            st.slider("🌳 Green infrastructure (%)", 0, 100, step=5, key="green_infra_pct")
            # Quick visual cue for green vs urban balance
            _g = int(st.session_state["green_infra_pct"])
            _u = int(st.session_state.get("urbanization_pct", 40))
            st.markdown(
                f"<div style='display:flex;height:6px;border-radius:99px;overflow:hidden;"
                f"margin:-4px 0 6px;background:rgba(255,255,255,0.06);'>"
                f"<div style='flex:{_g};background:linear-gradient(90deg,#10b981,#34d399);'></div>"
                f"<div style='flex:{_u};background:linear-gradient(90deg,#fb923c,#ef4444);'></div>"
                f"<div style='flex:{max(0, 100 - _g - _u)};background:transparent;'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.slider("🏗️ Urbanization (%)", 0, 100, step=5, key="urbanization_pct")

    # ---- 4. SCENARIO LIBRARY (compact) -------------------------
    with st.expander("📚 Scenario library", expanded=False):
        sl_keys = list(SCENARIO_LIBRARY.keys())
        chosen = st.selectbox(
            "IPCC-anchored or London-policy scenario:",
            ["— custom —"] + sl_keys, index=0, key="scenario_pick",
        )
        sa1, sa2 = st.columns([1, 1])
        with sa1:
            if st.button("Apply", use_container_width=True,
                         disabled=(chosen == "— custom —"), key="apply_lib"):
                for k, v in SCENARIO_LIBRARY[chosen].items():
                    st.session_state[k] = v
                st.session_state["challenge_won"] = False
                st.rerun()
        with sa2:
            if st.button("🔄 Reset", use_container_width=True, key="reset_lib"):
                for k in list(DEFAULTS.keys()):
                    st.session_state.pop(k, None)
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.pop("selected_location_idx", None)
                st.rerun()

    # ---- 5. ADVANCED — display + challenge + compare + AI key ---
    with st.expander("🔧 Advanced", expanded=False):
        st.markdown("**🔬 Display options**")
        st.toggle("5–95% uncertainty bands", key="show_uncertainty")
        st.toggle("IPCC SSP reference scenarios", key="show_ssp")
        st.toggle("HadCRUT5 historical splice", key="show_history")
        st.toggle("Researcher mode", key="researcher_mode",
                  help="Shows additional diagnostics in the Charts tab.")

        st.divider()
        st.markdown("**🎯 Challenge mode**")
        st.toggle("Enable challenge", key="challenge_on")
        diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty_choice")
        target_flood = int(DIFFICULTY_TARGETS[diff]["target_flood"])
        target_drought = int(DIFFICULTY_TARGETS[diff]["target_drought"])
        st.caption(f"Targets: Flood ≤ {target_flood} · Drought ≤ {target_drought}")
        if st.button("🏆 Reset calibration"):
            st.session_state["challenge_won"] = False
            st.rerun()

        st.divider()
        st.markdown("**🧪 Scenario comparison**")
        st.toggle("Enable A/B comparison", key="compare_on")

        st.divider()
        st.markdown("**🤖 AI assistant key**")
        api_key_input = st.text_input(
            "Anthropic API key (optional)", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Leave blank to use the offline brain.",
        )
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input

    # Make targets visible outside the expander too
    if not st.session_state.get("challenge_on", False):
        target_flood = int(DIFFICULTY_TARGETS[
            st.session_state.get("difficulty_choice", "Medium")
        ]["target_flood"])
        target_drought = int(DIFFICULTY_TARGETS[
            st.session_state.get("difficulty_choice", "Medium")
        ]["target_drought"])

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

# --- Compact metric strip — small footprint so the globe fits above the fold ---
_temp_label = 'Paris-aligned' if temp_val < 2 else 'Above Paris target'
_flood_label = ('Low' if flood_val < 25 else 'Moderate' if flood_val < 50
                else 'High' if flood_val < 75 else 'Very high')
_drought_label = ('Low' if drought_val < 25 else 'Moderate' if drought_val < 50
                  else 'High' if drought_val < 75 else 'Very high')

st.markdown(f"""
<div class="metric-strip">
  <div class="metric-pill">
    <div class="mp-icon">🌡️</div>
    <div class="mp-body">
      <div class="mp-label">End-of-horizon warming</div>
      <div class="mp-value {severity_class(temp_val, 'temp')}">{temp_val:.2f} °C</div>
      <div class="mp-trend">{_temp_label}</div>
    </div>
  </div>
  <div class="metric-pill">
    <div class="mp-icon">🌊</div>
    <div class="mp-body">
      <div class="mp-label">Flood risk</div>
      <div class="mp-value {severity_class(flood_val)}">{flood_val:.0f}<span style="font-size:13px;color:#94a3b8;"> / 100</span></div>
      <div class="mp-trend">{_flood_label}</div>
    </div>
  </div>
  <div class="metric-pill">
    <div class="mp-icon">🌵</div>
    <div class="mp-body">
      <div class="mp-label">Drought risk</div>
      <div class="mp-value {severity_class(drought_val)}">{drought_val:.0f}<span style="font-size:13px;color:#94a3b8;"> / 100</span></div>
      <div class="mp-trend">{_drought_label}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Tabs
# ============================================================================
tab_charts, tab_sens, tab_ai, tab_methods, tab_export = st.tabs(
    ["📈 Charts & live map", "🔬 Sensitivity", "🤖 AI assistant",
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

    # ============================================================
    # SECTION 1 — ROTATING 3D GLOBE + LOCATION SPLASH (above-the-fold)
    # ============================================================

    # Read selected location FIRST so we can recenter the globe before render
    sel_idx = st.session_state.get("selected_location_idx")

    # Compute per-location risks (responds to current sliders)
    globe_view = st.session_state.get("globe_risk_view", "Combined")
    globe_data = []
    for loc in LONDON_LOCATIONS:
        fl, dr = local_risk(loc, flood_val, drought_val,
                            st.session_state["green_infra_pct"],
                            st.session_state["urbanization_pct"])
        if globe_view == "Flood risk":
            shown = fl
        elif globe_view == "Drought risk":
            shown = dr
        else:
            shown = (fl + dr) / 2
        globe_data.append({**loc, "flood": fl, "drought": dr, "shown": shown})

    # ---- Layout: Globe (left, 2/3) + Splash card (right, 1/3) ----
    g_left, g_right = st.columns([2, 1], gap="medium")

    with g_right:
        st.markdown("### 🌍 3D London globe")
        globe_view = st.radio(
            "Risk shown",
            ["Combined", "Flood risk", "Drought risk"],
            key="globe_risk_view",
            horizontal=True,
            label_visibility="collapsed",
        )
        st.caption("🌐 Globe rotates automatically. Click a marker (or use the dropdown below) to zoom in and see its rain/drought signs.")

    # ============================================================
    # GLOBE — rendered as raw Plotly.js inside an iframe so that
    # auto-rotation actually fires (st.plotly_chart doesn't autoplay
    # animations and JS injection across iframes is unreliable).
    # Selection happens via the dropdown in the right column below.
    # ============================================================
    sel_loc_payload = None
    if sel_idx is not None and 0 <= sel_idx < len(globe_data):
        s = globe_data[sel_idx]
        sel_loc_payload = {
            "name": s["name"], "lat": s["lat"], "lon": s["lon"],
            "flood": float(s["flood"]), "drought": float(s["drought"]),
            "borough": s.get("borough", "—"),
        }

    globe_payload = {
        "points": [{
            "name": d["name"], "lat": d["lat"], "lon": d["lon"],
            "shown": float(d["shown"]),
            "flood": float(d["flood"]), "drought": float(d["drought"]),
            "borough": d.get("borough", "—"),
            "river": int(d["river"] * 100), "green": int(d["green"] * 100),
            "urban": int(d["urban"] * 100),
        } for d in globe_data],
        "view": globe_view,
        "selected": sel_loc_payload,
        # City-wide values used in the OSM annotation when zoomed in
        "city_temp": float(temp_val),
        "city_flood": float(flood_val),
        "city_drought": float(drought_val),
    }

    globe_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
  #globe { width: 100%; height: 580px; background: transparent; }
  .modebar { display: none !important; }
</style>
</head>
<body>
<div id="globe"></div>
<script>
const PAYLOAD = __PAYLOAD__;
const points = PAYLOAD.points;
const sel = PAYLOAD.selected;
const view = PAYLOAD.view;

const COLORSCALE = [
  [0.0, '#10b981'], [0.25, '#34d399'],
  [0.5, '#fbbf24'], [0.75, '#fb923c'], [1.0, '#ef4444']
];

function buildTraces() {
  const traces = [];

  // Pulsing rings behind high-risk markers
  for (const d of points) {
    if (d.shown >= 60) {
      traces.push({
        type: 'scattergeo',
        lon: [d.lon], lat: [d.lat],
        mode: 'markers',
        marker: {
          size: Math.round(34 + d.shown / 3),
          color: d.shown >= 75 ? 'rgba(248,113,113,0.20)' : 'rgba(56,189,248,0.20)',
          line: {width: 0}
        },
        hoverinfo: 'skip', showlegend: false
      });
    }
  }

  // Main markers — clickable
  traces.push({
    type: 'scattergeo',
    lon: points.map(d => d.lon),
    lat: points.map(d => d.lat),
    text: points.map(d => d.name),
    customdata: points.map(d => [d.flood, d.drought, d.borough, d.river, d.green, d.urban]),
    mode: 'markers',
    marker: {
      size: points.map(d => Math.max(12, 10 + d.shown / 3)),
      color: points.map(d => d.shown),
      colorscale: COLORSCALE,
      cmin: 0, cmax: 100,
      showscale: true,
      colorbar: {
        title: {text: view, font: {color: '#cbd5e1', size: 11}},
        tickfont: {color: '#cbd5e1', size: 10},
        thickness: 10, len: 0.6, x: 1.02,
        bgcolor: 'rgba(0,0,0,0)'
      },
      line: {width: 1.5, color: 'rgba(255,255,255,0.7)'},
      opacity: 0.95
    },
    hovertemplate:
      '<b>%{text}</b><br>' +
      '🏙️ Borough: %{customdata[2]}<br>' +
      '🌊 Flood risk: <b>%{customdata[0]:.0f}</b>/100<br>' +
      '🌵 Drought risk: <b>%{customdata[1]:.0f}</b>/100<br>' +
      '<i>River %{customdata[3]}% · green %{customdata[4]}% · urban %{customdata[5]}%</i>' +
      '<extra></extra>',
    name: 'London hotspots'
  });

  // Visual signs when a location is selected
  if (sel) {
    // Outer aura — sized for scale=12 so it visibly haloes the borough
    traces.push({type: 'scattergeo', lon: [sel.lon], lat: [sel.lat], mode: 'markers',
      marker: {size: 80, color: 'rgba(251,191,36,0.18)', line: {width: 0}},
      hoverinfo: 'skip', showlegend: false});
    // Mid ring
    traces.push({type: 'scattergeo', lon: [sel.lon], lat: [sel.lat], mode: 'markers',
      marker: {size: 50, color: 'rgba(251,191,36,0.32)', line: {width: 0}},
      hoverinfo: 'skip', showlegend: false});
    // Solid ring + name
    traces.push({type: 'scattergeo', lon: [sel.lon], lat: [sel.lat],
      mode: 'markers+text',
      marker: {size: 28, color: 'rgba(0,0,0,0)', line: {width: 3, color: '#fbbf24'}},
      text: ['📍 ' + sel.name],
      textposition: 'top center',
      textfont: {size: 14, color: '#fbbf24',
                 family: '-apple-system, BlinkMacSystemFont, Inter, sans-serif'},
      hoverinfo: 'skip', showlegend: false});

    // ===== ICON ROW — sits clearly above/below the borough at scale=12 =====
    // At scale=12 the visible window is ~15° wide, so a 0.6° lat offset puts
    // icons ~30-40px above/below the pin: visible but still hugging the spot.
    const ROW_OFFSET_LAT = 0.55;   // ~60km above/below the pin
    const ICON_STEP_LON  = 0.45;   // horizontal spacing between icons

    // Icon set scales BOTH count and size with severity, capped at 3 icons.
    let floodIcons, floodSize;
    if (sel.flood < 30)       { floodIcons = ['💧'];           floodSize = 26; }
    else if (sel.flood < 60)  { floodIcons = ['💧', '🌊'];     floodSize = 28; }
    else if (sel.flood < 85)  { floodIcons = ['💧','🌊','☔']; floodSize = 32; }
    else                      { floodIcons = ['🌊','☔','🌊']; floodSize = 36; }

    let droughtIcons, droughtSize;
    if (sel.drought < 30)      { droughtIcons = ['🌱'];             droughtSize = 26; }
    else if (sel.drought < 60) { droughtIcons = ['☀️','🌵'];        droughtSize = 28; }
    else if (sel.drought < 85) { droughtIcons = ['🔥','☀️','🌵'];   droughtSize = 32; }
    else                       { droughtIcons = ['🔥','🥵','🔥'];   droughtSize = 36; }

    // Place flood icons in a row ABOVE the pin
    const flood_n = floodIcons.length;
    for (let i = 0; i < flood_n; i++) {
      const dLon = sel.lon + (i - (flood_n - 1) / 2) * ICON_STEP_LON;
      traces.push({type: 'scattergeo', lon: [dLon], lat: [sel.lat + ROW_OFFSET_LAT],
        mode: 'text', text: [floodIcons[i]],
        textfont: {size: floodSize, color: '#56B4E9'},
        hoverinfo: 'skip', showlegend: false});
    }

    // Place drought icons in a row BELOW the pin
    const drought_n = droughtIcons.length;
    for (let i = 0; i < drought_n; i++) {
      const dLon = sel.lon + (i - (drought_n - 1) / 2) * ICON_STEP_LON;
      traces.push({type: 'scattergeo', lon: [dLon], lat: [sel.lat - ROW_OFFSET_LAT],
        mode: 'text', text: [droughtIcons[i]],
        textfont: {size: droughtSize, color: '#E69F00'},
        hoverinfo: 'skip', showlegend: false});
    }
  }

  return traces;
}

const config = {
  displaylogo: false,
  displayModeBar: false,
  responsive: true,
  scrollZoom: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

// =============================================================
// HYBRID VIEW:
//   - No selection: spinning 3D orthographic globe (world view)
//   - Selection:    Google-Maps-style street view (OpenStreetMap)
//                   with temp / flood / drought labels overlaid
// =============================================================
const gd = document.getElementById('globe');

// ---------- MAPBOX (street-level) traces for the zoomed view ----------
function buildMapboxTraces() {
  const out = [];
  if (!sel) return out;

  // Layered glow pin
  out.push({type:'scattermapbox', lon:[sel.lon], lat:[sel.lat],
    mode:'markers',
    marker:{size:90, color:'rgba(251,191,36,0.18)'},
    hoverinfo:'skip', showlegend:false});
  out.push({type:'scattermapbox', lon:[sel.lon], lat:[sel.lat],
    mode:'markers',
    marker:{size:60, color:'rgba(251,191,36,0.32)'},
    hoverinfo:'skip', showlegend:false});
  out.push({type:'scattermapbox', lon:[sel.lon], lat:[sel.lat],
    mode:'markers+text',
    marker:{size:28, color:'#fbbf24'},
    text:['📍 '+sel.name],
    textposition:'top right',
    textfont:{size:14, color:'#fbbf24',
              family:'-apple-system, BlinkMacSystemFont, Inter, sans-serif'},
    hoverinfo:'skip', showlegend:false});

  // Visual signs near the pin — flood ABOVE, drought BELOW
  const ROW_OFFSET_LAT = 0.0035;
  const ICON_STEP_LON  = 0.0030;

  let fIcons, fSize;
  if (sel.flood < 30)       { fIcons=['💧'];           fSize=30; }
  else if (sel.flood < 60)  { fIcons=['💧','🌊'];      fSize=32; }
  else if (sel.flood < 85)  { fIcons=['💧','🌊','☔']; fSize=36; }
  else                      { fIcons=['🌊','☔','🌊']; fSize=40; }

  let dIcons, dSize;
  if (sel.drought < 30)     { dIcons=['🌱'];            dSize=30; }
  else if (sel.drought < 60){ dIcons=['☀️','🌵'];       dSize=32; }
  else if (sel.drought < 85){ dIcons=['🔥','☀️','🌵'];  dSize=36; }
  else                      { dIcons=['🔥','🥵','🔥'];  dSize=40; }

  for (let i=0;i<fIcons.length;i++) {
    const dLon = sel.lon + (i - (fIcons.length-1)/2) * ICON_STEP_LON;
    out.push({type:'scattermapbox', lon:[dLon], lat:[sel.lat+ROW_OFFSET_LAT],
      mode:'text', text:[fIcons[i]],
      textfont:{size:fSize, color:'#1e3a8a'},
      hoverinfo:'skip', showlegend:false});
  }
  for (let i=0;i<dIcons.length;i++) {
    const dLon = sel.lon + (i - (dIcons.length-1)/2) * ICON_STEP_LON;
    out.push({type:'scattermapbox', lon:[dLon], lat:[sel.lat-ROW_OFFSET_LAT],
      mode:'text', text:[dIcons[i]],
      textfont:{size:dSize, color:'#7c2d12'},
      hoverinfo:'skip', showlegend:false});
  }
  return out;
}

if (sel) {
  // ============================================================
  // STREET-LEVEL VIEW — OpenStreetMap (Google-Maps-style detail)
  // with temp/flood/drought header labels overlaid on the map.
  // ============================================================
  const tempLabel    = (PAYLOAD.city_temp ?? 0).toFixed(2) + '°C';
  const floodLabel   = Math.round(sel.flood) + '/100';
  const droughtLabel = Math.round(sel.drought) + '/100';

  const mapLayout = {
    mapbox: {
      style: 'open-street-map',
      center: {lat: sel.lat, lon: sel.lon},
      zoom: 14
    },
    margin: {t: 0, b: 0, l: 0, r: 0},
    height: 580,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    annotations: [
      // Top-left: location name
      {x:0.01, y:0.99, xref:'paper', yref:'paper',
       xanchor:'left', yanchor:'top',
       text: "<b style='color:#0b1220;font-size:15px'>📍 " + sel.name + "</b>"
             + "<br><span style='color:#475569;font-size:11px'>" + sel.borough + "</span>",
       showarrow:false,
       font:{family:'-apple-system, BlinkMacSystemFont, Inter, sans-serif'},
       bgcolor:'rgba(255,255,255,0.94)',
       bordercolor:'rgba(251,191,36,0.7)',
       borderwidth:2, borderpad:10, align:'left'},
      // Top-right: live values strip (temp / flood / drought)
      {x:0.99, y:0.99, xref:'paper', yref:'paper',
       xanchor:'right', yanchor:'top',
       text: "<b style='color:#dc2626;font-size:13px'>🌡️ " + tempLabel + "</b>"
             + "<br><b style='color:#1e40af;font-size:13px'>🌊 Flood " + floodLabel + "</b>"
             + "<br><b style='color:#b45309;font-size:13px'>🌵 Drought " + droughtLabel + "</b>",
       showarrow:false,
       font:{family:'-apple-system, BlinkMacSystemFont, Inter, sans-serif'},
       bgcolor:'rgba(255,255,255,0.94)',
       bordercolor:'rgba(15,23,42,0.25)',
       borderwidth:1, borderpad:10, align:'right'}
    ]
  };
  Plotly.newPlot(gd, buildMapboxTraces(), mapLayout, config);

} else {
  // ============================================================
  // WORLD VIEW — auto-rotating 3D orthographic globe
  // ============================================================
  const globeLayout = {
    geo: {
      resolution: 50,
      projection: {
        type: 'orthographic',
        rotation: {lon: -0.13, lat: 20, roll: 0},
        scale: 0.95  // zoomed out so the entire globe is comfortably in frame
      },
      showland: true,        landcolor:    'rgb(75, 95, 130)',
      showocean: true,       oceancolor:   'rgb(8, 14, 28)',
      showcountries: true,   countrycolor: 'rgba(255,255,255,0.35)',
      showcoastlines: true,  coastlinecolor:'rgba(125,211,252,0.75)',
      showlakes: true,       lakecolor:    'rgb(12, 22, 44)',
      showrivers: true,      rivercolor:   'rgba(56,189,248,0.7)',
      showsubunits: true,    subunitcolor: 'rgba(255,255,255,0.20)',
      bgcolor: 'rgba(0,0,0,0)'
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: '-apple-system, BlinkMacSystemFont, Inter, sans-serif', color: '#cbd5e1'},
    margin: {t: 10, b: 10, l: 0, r: 0},
    height: 580,
    hoverlabel: {bgcolor: 'rgba(15,23,42,0.96)',
                 bordercolor: 'rgba(125,211,252,0.4)',
                 font: {color: '#e2e8f0', size: 12}},
    showlegend: false
  };

  Plotly.newPlot(gd, buildTraces(), globeLayout, config).then(() => {
    let rotLon = -180;
    let paused = false;
    gd.addEventListener('mouseenter', () => { paused = true; });
    gd.addEventListener('mouseleave', () => { paused = false; });
    setInterval(() => {
      if (paused) return;
      rotLon = (rotLon + 1.6) % 360;
      const lon = rotLon > 180 ? rotLon - 360 : rotLon;
      Plotly.relayout(gd, {
        'geo.projection.rotation.lon': lon,
        'geo.projection.rotation.lat': 20
      });
    }, 50);
  });
}
</script>
</body>
</html>
"""

    # Inject payload as JSON literal; escape "</" to avoid </script> injection
    globe_html_filled = globe_html.replace(
        "__PAYLOAD__",
        json.dumps(globe_payload).replace("</", "<\\/")
    )

    with g_left:
        components.html(globe_html_filled, height=600, scrolling=False)


    # ---- Right column: location splash card ----
    sel_idx = st.session_state.get("selected_location_idx")

    with g_right:
        # Plain dropdown — uses placeholder text when nothing is picked,
        # so the spinning globe stays the default view on first load.
        loc_names = [loc["name"] for loc in LONDON_LOCATIONS]
        picked = st.selectbox(
            "📍 Pick a location to zoom in",
            loc_names,
            index=sel_idx if sel_idx is not None else None,
            placeholder="Choose a London borough…",
            key="loc_picker",
        )
        new_idx = loc_names.index(picked) if picked is not None else None

        # Force immediate rerun so the view switches without lag
        if new_idx != sel_idx:
            if new_idx is None:
                st.session_state.pop("selected_location_idx", None)
            else:
                st.session_state["selected_location_idx"] = new_idx
            st.rerun()

        if sel_idx is None:
            st.info("👆 Click a marker on the globe (or pick from the dropdown) to splash its 80-year trajectory into the charts below.")
        else:
            sel_loc = LONDON_LOCATIONS[sel_idx]
            sel_d = next((g for g in globe_data if g["name"] == sel_loc["name"]), None)
            if sel_d:
                st.markdown(f"""
                <div class="loc-splash">
                  <div class="ls-eyebrow">Selected location</div>
                  <div class="ls-name">📍 {sel_d['name']}</div>
                  <div class="ls-borough">🏙️ {sel_d.get('borough', '—')} ·
                    river {int(sel_d['river']*100)}% · green {int(sel_d['green']*100)}% · urban {int(sel_d['urban']*100)}%</div>
                  <div class="ls-row">
                    <div class="ls-stat">
                      <div class="lsv {severity_class(sel_d['flood'])}">{sel_d['flood']:.0f}</div>
                      <div class="lsl">🌊 Flood / 100</div>
                    </div>
                    <div class="ls-stat">
                      <div class="lsv {severity_class(sel_d['drought'])}">{sel_d['drought']:.0f}</div>
                      <div class="lsl">🌵 Drought / 100</div>
                    </div>
                    <div class="ls-stat">
                      <div class="lsv">{sel_d['flood'] - flood_val:+.0f}</div>
                      <div class="lsl">vs city avg</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔄 Back to city view", use_container_width=True, key="reset_loc"):
                    # Clear BOTH the index AND the dropdown's session state —
                    # otherwise the dropdown re-applies its previous value on rerun.
                    st.session_state.pop("selected_location_idx", None)
                    st.session_state.pop("loc_picker", None)
                    st.rerun()

    st.caption(
        "Location risk profiles are illustrative for the festival demo, not an official hazard map. "
        "For authoritative London flood data see the [Environment Agency Flood Map for Planning](https://flood-map-for-planning.service.gov.uk/)."
    )

    # ============================================================
    # SECTION 2 — TRAJECTORY CHARTS (city-wide OR selected location)
    # ============================================================

    # If a location is selected, build location-specific trajectories
    sel_idx = st.session_state.get("selected_location_idx")
    if sel_idx is not None:
        sel_loc = LONDON_LOCATIONS[sel_idx]
        # Modulate every year's flood/drought via local_risk
        local_flood, local_drought = [], []
        for _, row in df.iterrows():
            fl, dr = local_risk(sel_loc, float(row["flood_risk"]), float(row["drought_risk"]),
                                st.session_state["green_infra_pct"],
                                st.session_state["urbanization_pct"])
            local_flood.append(fl)
            local_drought.append(dr)
        local_label = f"📍 {sel_loc['name']} — splashed trajectory"
        risk_title = f"🌊 Flood & drought — {sel_loc['name']}"
        temp_title = "🌡️ Temperature trajectory (city-wide)"
    else:
        local_flood = local_drought = None
        local_label = ""
        risk_title = "🌊 Flood & drought risk (city-wide)"
        temp_title = "🌡️ Temperature trajectory"

    L, R = st.columns([1, 1])

    # ---------------- Temperature ----------------
    with L:
        st.subheader(temp_title)
        fig_t = go.Figure()

        if st.session_state["show_history"]:
            fig_t.add_trace(go.Scatter(
                x=[y for y, _ in HADCRUT_HISTORICAL],
                y=[t for _, t in HADCRUT_HISTORICAL],
                mode="lines+markers", name="HadCRUT5 (1850–2023)",
                line=dict(color="#94a3b8", width=1.6, shape="spline"),
                marker=dict(size=4, color="#94a3b8"),
                hovertemplate="<b>%{x}</b>: %{y:.2f} °C<extra>HadCRUT5</extra>",
            ))

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

        fig_t.add_trace(go.Scatter(
            x=df["year"], y=df["temp_anomaly_C"],
            mode="lines", name="Your scenario",
            line=dict(color="#7dd3fc", width=3, shape="spline"),
            hovertemplate="<b>%{x}</b>: <b>%{y:.2f} °C</b><extra>Your scenario</extra>",
        ))

        if st.session_state["show_ssp"]:
            for name, info in SSP_SCENARIOS.items():
                fig_t.add_hline(
                    y=info["warming_2100"],
                    line=dict(color=info["color"], width=1, dash="dot"),
                    annotation=dict(text=name, font=dict(size=9, color=info["color"]),
                                    bgcolor="rgba(15,23,42,0.6)", borderpad=2),
                    annotation_position="right",
                )

        fig_t.update_layout(**_plotly_layout("Projected warming above pre-industrial", "°C", height=360))
        st.plotly_chart(fig_t, use_container_width=True,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # ---------------- Risks (city OR location) ----------------
    with R:
        st.subheader(risk_title)
        fig_r = go.Figure()

        if mc is not None and sel_idx is None:
            # Bands only on the city-wide view (Monte Carlo is on city aggregates)
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["flood_p95"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["flood_p05"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(86,180,233,0.16)", name="Flood 5–95% band",
                hoverinfo="skip"))
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["drought_p95"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_r.add_trace(go.Scatter(x=mc["year"], y=mc["drought_p05"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(230,159,0,0.16)", name="Drought 5–95% band",
                hoverinfo="skip"))

        # City baseline — drawn faint when a location is selected
        baseline_op = 0.35 if sel_idx is not None else 1.0
        fig_r.add_trace(go.Scatter(
            x=df["year"], y=df["flood_risk"], mode="lines",
            name="🌊 Flood (city)" if sel_idx is not None else "🌊 Flood",
            line=dict(color="#56B4E9", width=2 if sel_idx is not None else 3,
                      shape="spline", dash="dot" if sel_idx is not None else "solid"),
            opacity=baseline_op,
            hovertemplate="<b>%{x}</b>: <b>%{y:.0f}</b>/100<extra>City flood</extra>",
        ))
        fig_r.add_trace(go.Scatter(
            x=df["year"], y=df["drought_risk"], mode="lines",
            name="🌵 Drought (city)" if sel_idx is not None else "🌵 Drought",
            line=dict(color="#E69F00", width=2 if sel_idx is not None else 3,
                      shape="spline", dash="dot" if sel_idx is not None else "solid"),
            opacity=baseline_op,
            hovertemplate="<b>%{x}</b>: <b>%{y:.0f}</b>/100<extra>City drought</extra>",
        ))

        # SPLASHED location lines
        if sel_idx is not None:
            sel_name = LONDON_LOCATIONS[sel_idx]["name"]
            fig_r.add_trace(go.Scatter(
                x=df["year"], y=local_flood, mode="lines",
                name=f"🌊 {sel_name}",
                line=dict(color="#38bdf8", width=4, shape="spline"),
                hovertemplate=("<b>%{x}</b>: <b>%{y:.0f}</b>/100"
                               f"<extra>{sel_name} flood</extra>"),
            ))
            fig_r.add_trace(go.Scatter(
                x=df["year"], y=local_drought, mode="lines",
                name=f"🌵 {sel_name}",
                line=dict(color="#fb923c", width=4, shape="spline"),
                hovertemplate=("<b>%{x}</b>: <b>%{y:.0f}</b>/100"
                               f"<extra>{sel_name} drought</extra>"),
            ))

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

        layout = _plotly_layout("Risk trajectories (0–100 unitless index)", "Risk index", height=360)
        layout["yaxis"]["range"] = [0, 100]
        fig_r.update_layout(**layout)
        st.plotly_chart(fig_r, use_container_width=True,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    if sel_idx is not None:
        st.caption(f"💡 Charts above show {LONDON_LOCATIONS[sel_idx]['name']}'s trajectory "
                   "(solid bold) overlaid on the city baseline (dotted). "
                   "Click '🔄 Back to city view' beside the globe to return to city-wide.")

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
# TAB 2 — Sensitivity (tornado)
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

# Footer removed — credit now lives on the splash welcome page.
