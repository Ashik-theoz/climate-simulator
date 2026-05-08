import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Map deps
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Can Engineering Reverse the Climate Clock?", layout="wide")

# ----------------------------
# Model
# ----------------------------
def simulate(years=80, co2_ppm=450, rainfall_change_pct=10, green_infra_pct=20, urbanization_pct=40):
    t = np.arange(years + 1)
    temp_anom = 1.2 * np.log(co2_ppm / 280)
    temp_series = temp_anom * (1 - np.exp(-t / 25))
    rainfall_factor = 1 + rainfall_change_pct / 100.0
    impervious = urbanization_pct / 100.0
    green = green_infra_pct / 100.0
    runoff_index = (rainfall_factor * (0.6 + 1.2 * impervious) * (1 - 0.55 * green))
    runoff_series = runoff_index * (1 + 0.08 * temp_series)
    flood_risk = 100 * (1 - np.exp(-0.9 * runoff_series))
    evap = (1 + 0.18 * temp_series)
    drought_index = (evap / rainfall_factor) * (1 - 0.15 * green)
    drought_risk = 100 * (1 - np.exp(-0.8 * drought_index))
    df = pd.DataFrame(
        {
            "year": 2025 + t,
            "temp_anomaly_C": temp_series,
            "flood_risk": flood_risk,
            "drought_risk": drought_risk,
        }
    )
    return df

# ----------------------------
# Defaults + session state init
# ----------------------------
DEFAULTS = {
    "mode": "Standard",
    "years": 80,
    "co2_ppm": 450,
    "rainfall_change_pct": 10,
    "green_infra_pct": 20,
    "urbanization_pct": 40,
    "challenge_on": False,
    "challenge_won": False,
    "difficulty_choice": "Medium",
    "compare_on": False,
    "scenario_A": None,
    "scenario_B": None,
    "chat_history": [],
}
DIFFICULTY_TARGETS = {
    "Easy": {"target_flood": 55, "target_drought": 55},
    "Medium": {"target_flood": 40, "target_drought": 40},
    "Hard": {"target_flood": 30, "target_drought": 30},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def snapshot_current(df: pd.DataFrame):
    params = {
        "mode": st.session_state["mode"],
        "years": int(st.session_state["years"]),
        "co2_ppm": int(st.session_state["co2_ppm"]),
        "rainfall_change_pct": int(st.session_state["rainfall_change_pct"]),
        "green_infra_pct": int(st.session_state["green_infra_pct"]),
        "urbanization_pct": int(st.session_state["urbanization_pct"]),
    }
    return {"params": params, "df": df.copy()}


def pretty_params(p: dict) -> str:
    return (
        f"Years={p['years']}, CO₂={p['co2_ppm']} ppm, Rain={p['rainfall_change_pct']}%, "
        f"Green={p['green_infra_pct']}%, Urban={p['urbanization_pct']}%"
    )

# ----------------------------
# Custom theme + hero header
# ----------------------------
st.markdown("""
<style>
/* App-wide cinematic theme */
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(46, 134, 222, 0.18), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(238, 90, 36, 0.18), transparent 60%),
        linear-gradient(180deg, #0b1220 0%, #0e1726 60%, #0b1220 100%);
}

/* Hero */
.hero-wrap {
    padding: 24px 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(46,134,222,0.18), rgba(238,90,36,0.18));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.05;
    background: linear-gradient(90deg, #7dd3fc 0%, #fbbf24 50%, #f87171 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.hero-sub {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 6px;
    opacity: 0.92;
}
.hero-icons {
    position: absolute;
    right: 24px;
    top: 18px;
    font-size: 36px;
    opacity: 0.95;
    letter-spacing: 6px;
    filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4));
}
.hero-icons .ic { display:inline-block; animation: float 4s ease-in-out infinite; }
.hero-icons .ic:nth-child(2){ animation-delay: 0.6s; }
.hero-icons .ic:nth-child(3){ animation-delay: 1.2s; }
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-6px); }
}

/* Custom metric cards */
.metric-card {
    border-radius: 14px;
    padding: 16px 18px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(6px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.25);
}
.metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 32px; font-weight: 800; margin-top: 4px; }
.metric-trend { font-size: 12px; color: #94a3b8; margin-top: 2px; }
.severity-low    { color: #34d399; }
.severity-mod    { color: #fbbf24; }
.severity-high   { color: #fb923c; }
.severity-vhigh  { color: #f87171; }

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1726 0%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Tab labels */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(46,134,222,0.25), rgba(238,90,36,0.25));
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.2s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(46,134,222,0.25);
}

/* AI status chip */
.ai-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-left: 8px;
}
.ai-chip.online  { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.ai-chip.offline { background: rgba(251, 146, 60, 0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
  <div class="hero-icons"><span class="ic">🌍</span><span class="ic">💧</span><span class="ic">🔥</span></div>
  <div class="hero-title">Can Engineering Reverse the Climate Clock?</div>
  <div class="hero-sub">An interactive London-scale climate simulator — pull the engineering levers and watch the future of flood, drought and heat respond in real time.</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Reset to Default"):
        for key in list(DEFAULTS.keys()):
            st.session_state.pop(key, None)
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.subheader("Quick scenarios")
    c1, c2, c3 = st.columns(3)
    if c1.button("🏢 Business"):
        st.session_state.update(
            {
                "mode": "Standard",
                "years": 80,
                "co2_ppm": 650,
                "rainfall_change_pct": 10,
                "green_infra_pct": 10,
                "urbanization_pct": 65,
                "challenge_won": False,
            }
        )
        st.rerun()
    if c2.button("🌿 Green"):
        st.session_state.update(
            {
                "mode": "Standard",
                "years": 80,
                "co2_ppm": 380,
                "rainfall_change_pct": 5,
                "green_infra_pct": 70,
                "urbanization_pct": 30,
                "challenge_won": False,
            }
        )
        st.rerun()
    if c3.button("🏙️ Urban"):
        st.session_state.update(
            {
                "mode": "Standard",
                "years": 80,
                "co2_ppm": 520,
                "rainfall_change_pct": 15,
                "green_infra_pct": 15,
                "urbanization_pct": 85,
                "challenge_won": False,
            }
        )
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
    st.caption("Optional: paste an Anthropic API key to enable chat. Without a key, a built-in explainer still works.")
    api_key_input = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Stored only in this session. Leave blank to use the offline explainer.",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

# ----------------------------
# Run simulation (current)
# ----------------------------
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

# Sidebar comparison buttons (after df is computed)
with st.sidebar:
    if st.session_state.get("compare_on", False):
        b1, b2, b3 = st.columns(3)
        if b1.button("Save A"):
            st.session_state["scenario_A"] = snapshot_current(df)
            st.rerun()
        if b2.button("Save B"):
            st.session_state["scenario_B"] = snapshot_current(df)
            st.rerun()
        if b3.button("Clear"):
            st.session_state["scenario_A"] = None
            st.session_state["scenario_B"] = None
            st.rerun()
        if st.session_state.get("scenario_A"):
            st.caption("A: " + pretty_params(st.session_state["scenario_A"]["params"]))
        if st.session_state.get("scenario_B"):
            st.caption("B: " + pretty_params(st.session_state["scenario_B"]["params"]))

# ----------------------------
# Challenge status
# ----------------------------
if st.session_state.get("challenge_on", False):
    flood_ok = flood_val <= target_flood
    drought_ok = drought_val <= target_drought
    colA, colB = st.columns(2)
    with colA:
        if flood_ok:
            st.success("🌊 Flood OK")
        else:
            st.error("🌊 Flood too high")
    with colB:
        if drought_ok:
            st.success("🌵 Drought OK")
        else:
            st.error("🌵 Drought too high")
    if flood_ok and drought_ok:
        if not st.session_state.get("challenge_won", False):
            st.balloons()
            st.session_state["challenge_won"] = True
    else:
        st.session_state["challenge_won"] = False

# ----------------------------
# Top metrics — custom styled cards
# ----------------------------
def severity_class(value, kind="risk"):
    if kind == "temp":
        if value < 1.0:  return "severity-low"
        if value < 2.0:  return "severity-mod"
        if value < 3.0:  return "severity-high"
        return "severity-vhigh"
    if value < 25:  return "severity-low"
    if value < 50:  return "severity-mod"
    if value < 75:  return "severity-high"
    return "severity-vhigh"

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌡️ End-of-horizon warming</div>
      <div class="metric-value {severity_class(temp_val,'temp')}">{temp_val:.2f} °C</div>
      <div class="metric-trend">{'Paris-aligned' if temp_val < 2 else 'Above Paris target'}</div>
    </div>
    """, unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌊 Flood risk</div>
      <div class="metric-value {severity_class(flood_val)}">{flood_val:.0f} <span style="font-size:16px;color:#94a3b8;">/ 100</span></div>
      <div class="metric-trend">{'Low' if flood_val<25 else 'Moderate' if flood_val<50 else 'High' if flood_val<75 else 'Very high'}</div>
    </div>
    """, unsafe_allow_html=True)
with mc3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌵 Drought risk</div>
      <div class="metric-value {severity_class(drought_val)}">{drought_val:.0f} <span style="font-size:16px;color:#94a3b8;">/ 100</span></div>
      <div class="metric-trend">{'Low' if drought_val<25 else 'Moderate' if drought_val<50 else 'High' if drought_val<75 else 'Very high'}</div>
    </div>
    """, unsafe_allow_html=True)
st.write("")  # breathing room

# ----------------------------
# Tabs: Charts | Map | AI
# ----------------------------
tab_charts, tab_map, tab_ai = st.tabs(["📈 Charts", "🗺️ London map", "🤖 AI assistant"])

# ============================
# TAB 1 — Charts
# ============================
with tab_charts:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Temperature (proxy)")
        fig = plt.figure(figsize=(6.8, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(df["year"], df["temp_anomaly_C"], linewidth=2)
        ax.set_title("Projected warming over time")
        ax.set_xlabel("Year")
        ax.set_ylabel("°C")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right:
        st.subheader("Risk proxies")
        fig = plt.figure(figsize=(6.8, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(df["year"], df["flood_risk"], linewidth=2, label="Flood risk")
        ax.plot(df["year"], df["drought_risk"], linewidth=2, label="Drought risk")
        ax.set_title("Flood and drought risk trajectory")
        ax.set_xlabel("Year")
        ax.set_ylabel("Risk (0–100)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax.minorticks_on()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if st.session_state.get("challenge_on", False):
            ax.axhline(target_flood, linestyle="--", alpha=0.6)
            ax.axhline(target_drought, linestyle="--", alpha=0.6)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Scenario comparison
    if st.session_state.get("compare_on", False):
        A = st.session_state.get("scenario_A")
        B = st.session_state.get("scenario_B")
        st.divider()
        st.subheader("🧪 Scenario comparison")
        if not A or not B:
            st.info("Save two scenarios (A and B) from the sidebar to compare them here.")
        else:
            dfA = A["df"]
            dfB = B["df"]
            tA, fA, dA = float(dfA["temp_anomaly_C"].iloc[-1]), float(dfA["flood_risk"].iloc[-1]), float(dfA["drought_risk"].iloc[-1])
            tB, fB, dB = float(dfB["temp_anomaly_C"].iloc[-1]), float(dfB["flood_risk"].iloc[-1]), float(dfB["drought_risk"].iloc[-1])
            m1, m2, m3 = st.columns(3)
            m1.metric("Δ Warming (A → B)", f"{(tB - tA):+.2f} °C")
            m2.metric("Δ Flood risk (A → B)", f"{(fB - fA):+.0f}")
            m3.metric("Δ Drought risk (A → B)", f"{(dB - dA):+.0f}")
            cL, cR = st.columns(2)
            with cL:
                st.markdown("**Temperature: A vs B**")
                fig = plt.figure(figsize=(6.8, 4.2))
                ax = fig.add_subplot(111)
                ax.plot(dfA["year"], dfA["temp_anomaly_C"], linewidth=2, label="Scenario A")
                ax.plot(dfB["year"], dfB["temp_anomaly_C"], linewidth=2, linestyle="--", label="Scenario B")
                ax.set_xlabel("Year")
                ax.set_ylabel("°C")
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(frameon=False, loc="upper left")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            with cR:
                st.markdown("**Risks: A vs B**")
                fig = plt.figure(figsize=(6.8, 4.2))
                ax = fig.add_subplot(111)
                ax.plot(dfA["year"], dfA["flood_risk"], linewidth=2, label="Flood (A)")
                ax.plot(dfA["year"], dfA["drought_risk"], linewidth=2, label="Drought (A)")
                ax.plot(dfB["year"], dfB["flood_risk"], linewidth=2, linestyle="--", label="Flood (B)")
                ax.plot(dfB["year"], dfB["drought_risk"], linewidth=2, linestyle="--", label="Drought (B)")
                ax.set_xlabel("Year")
                ax.set_ylabel("Risk (0–100)")
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(frameon=False, loc="upper left")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
            with st.expander("Show Scenario A & B parameters"):
                st.write("**Scenario A**:", A["params"])
                st.write("**Scenario B**:", B["params"])

# ============================
# TAB 2 — London map
# ============================

# A small set of London locations with characteristics that
# modulate flood vs drought risk locally:
#   river_factor   – proximity / vulnerability to Thames flooding (0..1)
#   green_factor   – existing green coverage (0..1)
#   urban_factor   – built-up / impervious surface (0..1)
LONDON_LOCATIONS = [
    {"name": "South Kensington (Exhibition Road)", "lat": 51.4988, "lon": -0.1749, "river": 0.25, "green": 0.55, "urban": 0.70},
    {"name": "Imperial College London",            "lat": 51.4988, "lon": -0.1749, "river": 0.25, "green": 0.50, "urban": 0.75},
    {"name": "Hyde Park",                          "lat": 51.5073, "lon": -0.1657, "river": 0.20, "green": 0.95, "urban": 0.10},
    {"name": "Westminster",                        "lat": 51.4995, "lon": -0.1248, "river": 0.85, "green": 0.30, "urban": 0.85},
    {"name": "Canary Wharf",                       "lat": 51.5054, "lon": -0.0235, "river": 0.95, "green": 0.20, "urban": 0.95},
    {"name": "Greenwich",                          "lat": 51.4826, "lon":  0.0077, "river": 0.90, "green": 0.55, "urban": 0.55},
    {"name": "Hackney",                            "lat": 51.5450, "lon": -0.0553, "river": 0.40, "green": 0.45, "urban": 0.75},
    {"name": "Richmond",                           "lat": 51.4613, "lon": -0.3037, "river": 0.65, "green": 0.80, "urban": 0.40},
    {"name": "Croydon",                            "lat": 51.3762, "lon": -0.0982, "river": 0.20, "green": 0.40, "urban": 0.80},
    {"name": "Heathrow Area",                      "lat": 51.4700, "lon": -0.4543, "river": 0.30, "green": 0.35, "urban": 0.85},
]


def local_risks(loc, flood_global, drought_global, user_green_pct, user_urban_pct):
    """Modulate the global risks by each location's geography."""
    river = loc["river"]
    green = loc["green"]
    urban = loc["urban"]

    # Combine user-controlled green/urban with location's intrinsic profile
    combined_green = 0.5 * green + 0.5 * (user_green_pct / 100.0)
    combined_urban = 0.5 * urban + 0.5 * (user_urban_pct / 100.0)

    # Flood: amplified by river proximity & impervious surface, dampened by green
    flood_local = flood_global * (0.7 + 0.6 * river) * (0.7 + 0.6 * combined_urban) * (1 - 0.35 * combined_green)
    # Drought: amplified in dense urban areas, dampened by green
    drought_local = drought_global * (0.8 + 0.4 * combined_urban) * (1 - 0.30 * combined_green)

    return float(np.clip(flood_local, 0, 100)), float(np.clip(drought_local, 0, 100))


def risk_color(value):
    """0=green, 50=amber, 100=red."""
    if value < 25:
        return "#2ecc71"   # green
    if value < 50:
        return "#f1c40f"   # yellow
    if value < 75:
        return "#e67e22"   # orange
    return "#e74c3c"       # red


with tab_map:
    st.subheader("London climate-risk hotspots")
    st.caption(
        "Each marker shows how today's slider settings would play out in a real London neighbourhood. "
        "Riverside areas get higher flood weighting; low-green / high-urban areas get higher drought weighting. "
        "Click a marker for details."
    )

    risk_view = st.radio(
        "Show on map:",
        ["Flood risk", "Drought risk", "Combined"],
        horizontal=True,
        key="map_risk_view",
    )

    # Build cinematic dark-mode map
    m = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=11,
        tiles="cartodbdark_matter",
        control_scale=True,
    )

    # Inject CSS into the map iframe for animated markers
    map_css = """
    <style>
    .climate-marker {
        font-size: 28px;
        text-align: center;
        line-height: 1;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6));
        transform: translate(-50%, -50%);
        position: relative;
        cursor: pointer;
    }
    .climate-marker .ring {
        position: absolute;
        left: 50%; top: 50%;
        width: 36px; height: 36px;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
    }
    .pulse-flood .ring {
        background: rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 0 0 rgba(56,189,248,0.7);
        animation: pulse-flood 1.8s infinite cubic-bezier(0.4,0,0.6,1);
    }
    .pulse-drought .ring {
        background: rgba(251, 146, 60, 0.35);
        box-shadow: 0 0 0 0 rgba(251,146,60,0.7);
        animation: pulse-drought 1.8s infinite cubic-bezier(0.4,0,0.6,1);
    }
    .pulse-extreme .ring {
        background: rgba(248, 113, 113, 0.4);
        box-shadow: 0 0 0 0 rgba(248,113,113,0.8);
        animation: pulse-extreme 1.4s infinite cubic-bezier(0.4,0,0.6,1);
    }
    @keyframes pulse-flood {
      0%   { box-shadow: 0 0 0 0    rgba(56,189,248,0.55); }
      70%  { box-shadow: 0 0 0 28px rgba(56,189,248,0);    }
      100% { box-shadow: 0 0 0 0    rgba(56,189,248,0);    }
    }
    @keyframes pulse-drought {
      0%   { box-shadow: 0 0 0 0    rgba(251,146,60,0.55); }
      70%  { box-shadow: 0 0 0 28px rgba(251,146,60,0);    }
      100% { box-shadow: 0 0 0 0    rgba(251,146,60,0);    }
    }
    @keyframes pulse-extreme {
      0%   { box-shadow: 0 0 0 0    rgba(248,113,113,0.65); }
      70%  { box-shadow: 0 0 0 32px rgba(248,113,113,0);    }
      100% { box-shadow: 0 0 0 0    rgba(248,113,113,0);    }
    }
    .climate-legend {
        position: fixed; bottom: 24px; left: 24px; z-index:9999;
        background: rgba(15,23,42,0.92); color: #e2e8f0;
        padding: 10px 14px; border-radius: 10px; font-size: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.08);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .climate-legend b { color: #f8fafc; }
    .leaflet-popup-content-wrapper {
        background: rgba(15,23,42,0.96); color: #e2e8f0;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .leaflet-popup-tip { background: rgba(15,23,42,0.96); }
    </style>
    """
    m.get_root().html.add_child(folium.Element(map_css))

    heat_points = []
    for loc in LONDON_LOCATIONS:
        f_local, d_local = local_risks(
            loc,
            flood_val,
            drought_val,
            st.session_state["green_infra_pct"],
            st.session_state["urbanization_pct"],
        )
        if risk_view == "Flood risk":
            shown = f_local
            kind = "flood"
        elif risk_view == "Drought risk":
            shown = d_local
            kind = "drought"
        else:
            shown = (f_local + d_local) / 2
            # whichever is dominant decides the icon flavour
            kind = "flood" if f_local >= d_local else "drought"

        # Choose emoji + animation
        if kind == "flood":
            emoji = "💧" if shown < 50 else "🌊"
            pulse_class = "pulse-flood" if shown >= 45 else ""
        else:
            emoji = "🌵" if shown < 50 else "🔥"
            pulse_class = "pulse-drought" if shown >= 45 else ""
        if shown >= 75:
            pulse_class = "pulse-extreme"
            emoji = "⚠️"

        size = int(28 + (shown / 100.0) * 22)  # 28..50 px

        marker_html = f"""
        <div class="climate-marker {pulse_class}" style="font-size:{size}px;">
            <div class="ring"></div>
            <span style="position:relative; z-index:2;">{emoji}</span>
        </div>
        """

        popup_html = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif; min-width:220px;">
          <div style="font-weight:700; font-size:14px; margin-bottom:6px;">{loc['name']}</div>
          <div style="display:flex; justify-content:space-between; padding:4px 0; border-top:1px solid rgba(255,255,255,0.1);">
            <span>🌊 Flood risk</span><b style="color:#7dd3fc;">{f_local:.0f}/100</b>
          </div>
          <div style="display:flex; justify-content:space-between; padding:4px 0; border-top:1px solid rgba(255,255,255,0.1);">
            <span>🌵 Drought risk</span><b style="color:#fbbf24;">{d_local:.0f}/100</b>
          </div>
          <div style="margin-top:8px; font-size:11px; color:#94a3b8;">
            River exposure {int(loc['river']*100)}% · green {int(loc['green']*100)}% · urban {int(loc['urban']*100)}%
          </div>
        </div>
        """

        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(size, size),
                icon_anchor=(size // 2, size // 2),
            ),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{loc['name']} — {risk_view}: {shown:.0f}/100",
        ).add_to(m)

        heat_points.append([loc["lat"], loc["lon"], shown / 100.0])

    # Heat halo layer (off by default — visitors can toggle)
    HeatMap(heat_points, radius=42, blur=32, min_opacity=0.3).add_to(
        folium.FeatureGroup(name="🔥 Heat halo", show=False).add_to(m)
    )
    folium.LayerControl(collapsed=True).add_to(m)

    # Cinematic legend
    legend_html = """
    <div class="climate-legend">
      <b>🌍 Climate-risk legend</b><br>
      <span style="font-size:14px;">💧</span> mild &nbsp;
      <span style="font-size:14px;">🌊</span> high flood<br>
      <span style="font-size:14px;">🌵</span> mild &nbsp;
      <span style="font-size:14px;">🔥</span> high drought<br>
      <span style="font-size:14px;">⚠️</span> very high risk (pulses)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Dynamic key forces the iframe to remount whenever inputs change,
    # otherwise st_folium caches the previous map and the radio button looks broken.
    map_key = (
        f"map_{risk_view}"
        f"_{int(flood_val)}_{int(drought_val)}"
        f"_{st.session_state['green_infra_pct']}"
        f"_{st.session_state['urbanization_pct']}"
        f"_{st.session_state['co2_ppm']}"
    )
    st_folium(m, width=None, height=520, returned_objects=[], key=map_key)

    st.caption(
        "Note: location risk profiles are illustrative for the festival demo, not an official hazard map."
    )

# ============================
# TAB 3 — AI assistant
# ============================

def offline_explainer(temp, flood, drought, params):
    """Rule-based fallback that always works (no API needed)."""
    bits = []
    if temp < 1.0:
        bits.append("warming stays modest")
    elif temp < 2.0:
        bits.append("warming reaches a noticeable level")
    else:
        bits.append("warming climbs into a dangerous range")

    if flood > 70:
        bits.append("flood risk becomes severe — surface water and river flooding are very likely")
    elif flood > 40:
        bits.append("flood risk is moderate but rising")
    else:
        bits.append("flood risk stays relatively contained")

    if drought > 70:
        bits.append("drought stress is severe — water supply and vegetation suffer")
    elif drought > 40:
        bits.append("drought risk is moderate")
    else:
        bits.append("drought risk stays low")

    levers = []
    if params["green_infra_pct"] < 30:
        levers.append("Adding more green infrastructure (parks, green roofs, SuDS) is the single biggest lever you haven't pulled.")
    if params["urbanization_pct"] > 60:
        levers.append("Heavy urbanization makes runoff worse — permeable pavements and tree pits would help.")
    if params["co2_ppm"] > 500:
        levers.append("CO₂ is high — this drives the warming term and amplifies every other risk.")
    if not levers:
        levers.append("Your settings are already in a fairly resilient zone — try pushing CO₂ down further to see what fully decarbonised looks like.")

    summary = (
        f"Over {params['years']} years with CO₂ at {params['co2_ppm']} ppm, "
        f"green infrastructure at {params['green_infra_pct']}%, and "
        f"urbanization at {params['urbanization_pct']}%, the simulator predicts that "
        + ", ".join(bits) + ".\n\n"
        + "**What you could try next:**\n- " + "\n- ".join(levers)
    )
    return summary


def offline_answer(question, temp, flood, drought, params):
    """
    Smarter rule-based Q&A for when no Anthropic API key is set.
    Detects the user's intent (where / why / how / what does X mean) and
    answers using the simulator state and the London location data.
    """
    q = (question or "").lower().strip()
    if not q:
        return offline_explainer(temp, flood, drought, params)

    asks_where   = any(w in q for w in ["where", "which area", "which location", "which place",
                                         "which neighbourhood", "which neighborhood",
                                         "what area", "what part", "highest", "worst", "most"])
    asks_flood   = "flood" in q
    asks_drought = "drought" in q
    asks_why     = any(w in q for w in ["why", "reason", "cause", "because"])
    asks_reduce  = any(w in q for w in ["reduce", "lower", "decrease", "fix", "improve",
                                         "how to", "how can", "how do", "what should", "solve"])
    asks_explain = any(w in q for w in ["explain", "what does", "what is", "what's", "tell me about",
                                         "meaning", "mean by"])

    # Compute per-location risks for any "where"-style answers
    local_data = []
    for loc in LONDON_LOCATIONS:
        f_local, d_local = local_risks(
            loc, flood, drought,
            params["green_infra_pct"], params["urbanization_pct"],
        )
        local_data.append({"name": loc["name"], "flood": f_local, "drought": d_local})

    # ---- "Where / which area is risk highest?" ----
    if asks_where and (asks_flood or asks_drought or "risk" in q):
        if asks_flood and not asks_drought:
            top = sorted(local_data, key=lambda x: -x["flood"])[:3]
            lines = ["**Areas with the highest flood risk in your scenario:**"]
            for i, item in enumerate(top, 1):
                lines.append(f"{i}. {item['name']} — {item['flood']:.0f} / 100")
            lines.append("")
            lines.append("Riverside zones (Westminster, Canary Wharf, Greenwich) and dense urban areas with little green cover get the worst flood scores. Try sliding **green infrastructure** up to see how much it helps.")
            return "\n".join(lines)
        if asks_drought and not asks_flood:
            top = sorted(local_data, key=lambda x: -x["drought"])[:3]
            lines = ["**Areas with the highest drought risk in your scenario:**"]
            for i, item in enumerate(top, 1):
                lines.append(f"{i}. {item['name']} — {item['drought']:.0f} / 100")
            lines.append("")
            lines.append("Drought hits dense, low-green areas hardest. Adding parks, green roofs and street trees would help most here.")
            return "\n".join(lines)
        # Combined
        top = sorted(local_data, key=lambda x: -((x["flood"] + x["drought"]) / 2))[:3]
        lines = ["**Areas with the highest combined climate risk:**"]
        for i, item in enumerate(top, 1):
            lines.append(f"{i}. {item['name']} — flood {item['flood']:.0f}, drought {item['drought']:.0f}")
        return "\n".join(lines)

    # ---- "Why is X risk so high?" ----
    if asks_why and asks_flood:
        reasons = []
        if params["urbanization_pct"] > 50:
            reasons.append(f"high urbanization ({params['urbanization_pct']}%) — concrete and roads stop water soaking into the ground, so it runs off as flood water")
        if params["green_infra_pct"] < 30:
            reasons.append(f"low green infrastructure ({params['green_infra_pct']}%) — not enough parks / SuDS to absorb runoff")
        if params["rainfall_change_pct"] > 5:
            reasons.append(f"rainfall is up by {params['rainfall_change_pct']}% — more water falling overall")
        if params["co2_ppm"] > 450:
            reasons.append(f"CO₂ at {params['co2_ppm']} ppm drives warming, which intensifies extreme rain events")
        if not reasons:
            return f"Flood risk is **{flood:.0f}/100** — fairly contained for your settings. The main drivers in this model are rainfall, impervious surfaces and warming."
        return f"Flood risk is **{flood:.0f}/100**. Main drivers in your scenario:\n\n- " + "\n- ".join(reasons)

    if asks_why and asks_drought:
        reasons = []
        if params["co2_ppm"] > 450:
            reasons.append(f"CO₂ at {params['co2_ppm']} ppm drives warming, which boosts evaporation and dries soils")
        if params["rainfall_change_pct"] < 0:
            reasons.append(f"rainfall is down by {abs(params['rainfall_change_pct'])}% — less water available")
        if params["green_infra_pct"] < 30:
            reasons.append(f"low green infrastructure ({params['green_infra_pct']}%) — soils dry out faster without vegetation")
        if params["urbanization_pct"] > 50:
            reasons.append(f"heavy urbanization ({params['urbanization_pct']}%) creates urban heat islands that worsen drought stress")
        if not reasons:
            return f"Drought risk is **{drought:.0f}/100** — under control for your settings."
        return f"Drought risk is **{drought:.0f}/100**. Main drivers in your scenario:\n\n- " + "\n- ".join(reasons)

    # ---- "How do I reduce X?" ----
    if asks_reduce:
        if asks_flood and not asks_drought:
            return (
                f"To lower flood risk (currently **{flood:.0f}/100**):\n\n"
                f"- Push **green infrastructure** up — it's the most effective lever (try 60–80%).\n"
                f"- Reduce **urbanization / imperviousness** — replace concrete with permeable surfaces.\n"
                f"- Lower **CO₂** — less warming means less intense rainfall.\n"
                f"- These also indirectly reduce drought."
            )
        if asks_drought and not asks_flood:
            return (
                f"To lower drought risk (currently **{drought:.0f}/100**):\n\n"
                f"- Lower **CO₂** — slower warming means less evaporation.\n"
                f"- Add **green infrastructure** — soils retain moisture and shade reduces heat stress.\n"
                f"- Reduce dense urbanization to weaken the urban heat island."
            )
        return (
            "**Three strongest levers to reduce climate risk:**\n\n"
            f"1. Lower CO₂ (you're at {params['co2_ppm']} ppm — try 350–400).\n"
            f"2. Boost green infrastructure (you're at {params['green_infra_pct']}% — try 60+).\n"
            f"3. Reduce urbanization (you're at {params['urbanization_pct']}% — try 30–40)."
        )

    # ---- "What does X mean?" ----
    if asks_explain:
        if "co2" in q or "co₂" in q or "carbon" in q:
            return (f"**CO₂ concentration** is how much carbon dioxide is in the atmosphere, in parts per million. "
                    f"Pre-industrial = 280 ppm, today ≈ 425 ppm. You've set it to **{params['co2_ppm']} ppm**. "
                    "Higher CO₂ means more warming, which amplifies floods and droughts.")
        if "green" in q or "infrastructure" in q or "suds" in q:
            return (f"**Green infrastructure** = parks, green roofs, street trees, sustainable drainage (SuDS), permeable pavements. "
                    "It absorbs rainfall, cools the city, and reduces flood and drought risk. "
                    f"You've set it to **{params['green_infra_pct']}%**.")
        if "urban" in q or "impervious" in q:
            return (f"**Urbanization / imperviousness** = how much of the city is covered by buildings, roads and concrete that water can't soak into. "
                    f"You're at **{params['urbanization_pct']}%**. Higher = worse flooding and worse drought.")
        if "flood" in q:
            return f"**Flood risk** is a 0–100 score for how likely flooding becomes given your CO₂, rainfall, urbanization and green-infrastructure settings. You're at **{flood:.0f}/100**."
        if "drought" in q:
            return f"**Drought risk** is a 0–100 score for how vulnerable your scenario is to water shortage. You're at **{drought:.0f}/100**."
        if "warming" in q or "temperature" in q or "heat" in q:
            return f"**Warming (proxy °C)** is the projected temperature rise above pre-industrial levels by the end of your simulation horizon. You're at **{temp:.2f} °C**. The Paris Agreement target is well below 2 °C."

    # Default: full summary + tip
    summary = offline_explainer(temp, flood, drought, params)
    return (
        summary
        + "\n\n_Tip: try asking things like_ "
        "*\"which area has the highest flood risk?\"*, "
        "*\"why is drought so high?\"*, "
        "*\"how do I reduce flood risk?\"*, or "
        "*\"what does CO₂ mean?\"*."
    )


def get_anthropic_key():
    """Resolve API key: sidebar/env first, then Streamlit secrets."""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    try:
        return (st.secrets.get("ANTHROPIC_API_KEY", "") or "").strip()
    except Exception:
        return ""


def ask_anthropic_stream(messages, system_prompt, placeholder):
    """Stream the response token-by-token into a Streamlit placeholder."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("The `anthropic` package is not installed.")

    api_key = get_anthropic_key()
    if not api_key:
        raise RuntimeError("No Anthropic API key set.")

    client = Anthropic(api_key=api_key)
    text_so_far = ""
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_prompt,
        messages=messages,
    ) as stream:
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
        st.caption("Running on the built-in explainer. Paste an Anthropic API key in the sidebar for streaming live AI answers.")
        with st.expander("🔑 How to get a free API key (≈30 seconds)"):
            st.markdown(
                "1. Go to **https://console.anthropic.com/** and sign in.\n"
                "2. Click **Get API keys** → **Create Key**. Anthropic gives free credits at signup, plenty for a festival booth.\n"
                "3. Copy the key (starts with `sk-ant-...`).\n"
                "4. Paste it in the **🤖 AI assistant** field in the sidebar.\n\n"
                "For a permanent setup on Streamlit Cloud, go to *Manage app → Settings → Secrets* and add: \n"
                "`ANTHROPIC_API_KEY = \"sk-ant-...\"` — then the key works for everyone visiting your booth."
            )

    current_params = {
        "years": int(st.session_state["years"]),
        "co2_ppm": int(st.session_state["co2_ppm"]),
        "rainfall_change_pct": int(st.session_state["rainfall_change_pct"]),
        "green_infra_pct": int(st.session_state["green_infra_pct"]),
        "urbanization_pct": int(st.session_state["urbanization_pct"]),
    }

    # ---- Auto explainer (always available) ----
    with st.container(border=True):
        st.markdown("**📝 Auto-summary of your current scenario**")
        st.write(offline_explainer(temp_val, flood_val, drought_val, current_params))

    st.divider()

    # ---- Chat ----
    st.markdown("**💬 Chat with the AI assistant**")

    use_kids_tone = st.session_state["mode"] == "Kids (simple)"
    system_prompt = (
        "You are a friendly climate engineering tutor at the Great Exhibition Road Festival in London. "
        "You help visitors understand a simple climate simulator. Keep replies short (under ~120 words), "
        "concrete, and grounded in the user's CURRENT simulator settings provided below. "
        "Avoid alarmism; explain trade-offs clearly. "
        + ("Use a playful, simple tone suitable for children aged 8-12. " if use_kids_tone else "Use clear, accessible language for a general adult audience. ")
        + f"\n\nCURRENT SCENARIO:\n"
        + f"- horizon: {current_params['years']} years\n"
        + f"- CO2: {current_params['co2_ppm']} ppm\n"
        + f"- rainfall change: {current_params['rainfall_change_pct']}%\n"
        + f"- green infrastructure: {current_params['green_infra_pct']}%\n"
        + f"- urbanization: {current_params['urbanization_pct']}%\n"
        + f"- end-of-horizon warming: {temp_val:.2f} °C\n"
        + f"- end-of-horizon flood risk: {flood_val:.0f}/100\n"
        + f"- end-of-horizon drought risk: {drought_val:.0f}/100\n"
    )

    # render chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question, e.g. 'Why is flood risk so high?'")
    if user_q:
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_🌍 Thinking..._")
            try:
                if get_anthropic_key():
                    answer = ask_anthropic_stream(
                        messages=[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state["chat_history"]],
                        system_prompt=system_prompt,
                        placeholder=placeholder,
                    )
                else:
                    raise RuntimeError("no_api_key")
            except Exception as e:
                # Graceful fallback — smart Q&A grounded in the user's actual question
                smart = offline_answer(user_q, temp_val, flood_val, drought_val, current_params)
                if str(e) == "no_api_key":
                    answer = smart  # offline brain — chip already tells the user
                else:
                    answer = f"_(Live AI hiccup: {e}. Using built-in answer.)_\n\n" + smart
                placeholder.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    cclear1, cclear2 = st.columns([1, 6])
    if cclear1.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()

# ----------------------------
# Explanation + Data
# ----------------------------
st.divider()
st.subheader("What's going on here?")
st.write(
    "This is a fast, educational simulator (not a full climate model). "
    "It's designed for interactive exploration: CO₂ affects warming, rainfall affects water stress, "
    "urbanization increases runoff, and green infrastructure reduces runoff and slightly improves resilience."
)
with st.expander("Show data table"):
    st.dataframe(df, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Developed by Ashikujjaman Mohammad | MSc Environmental Engineering | "
    "Imperial College London | 2026"
)
