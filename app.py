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
# UI Header
# ----------------------------
st.title("Can Engineering Reverse the Climate Clock?")
st.caption(
    "Interactive environmental simulation — adjust engineering choices and see future impacts instantly."
)

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
# Top metrics
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("End-of-horizon warming (proxy °C)", f"{temp_val:.2f}")
col2.metric("End-of-horizon flood risk (0–100)", f"{flood_val:.0f}")
col3.metric("End-of-horizon drought risk (0–100)", f"{drought_val:.0f}")

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

    # Build map
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="cartodbpositron")

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
        elif risk_view == "Drought risk":
            shown = d_local
        else:
            shown = (f_local + d_local) / 2

        color = risk_color(shown)
        radius = 8 + (shown / 100.0) * 14  # 8..22 px

        popup_html = (
            f"<b>{loc['name']}</b><br>"
            f"Flood risk: <b>{f_local:.0f}</b> / 100<br>"
            f"Drought risk: <b>{d_local:.0f}</b> / 100<br>"
            f"<i>River exposure: {int(loc['river']*100)}%, "
            f"green: {int(loc['green']*100)}%, "
            f"urban: {int(loc['urban']*100)}%</i>"
        )

        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{loc['name']} — {risk_view}: {shown:.0f}",
        ).add_to(m)

        heat_points.append([loc["lat"], loc["lon"], shown / 100.0])

    # Optional heat layer
    HeatMap(heat_points, radius=35, blur=25, min_opacity=0.25).add_to(
        folium.FeatureGroup(name="Heat layer", show=False).add_to(m)
    )
    folium.LayerControl().add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                background: rgba(255,255,255,0.92); padding: 8px 12px;
                border-radius: 6px; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <b>Risk legend</b><br>
      <span style="color:#2ecc71;">●</span> 0–25 low&nbsp;
      <span style="color:#f1c40f;">●</span> 25–50 moderate<br>
      <span style="color:#e67e22;">●</span> 50–75 high&nbsp;
      <span style="color:#e74c3c;">●</span> 75–100 very high
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


def ask_anthropic(messages, system_prompt):
    """Call Anthropic API. Returns string or raises."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("The `anthropic` package is not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No Anthropic API key set.")

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_prompt,
        messages=messages,
    )
    # Concatenate text blocks
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


with tab_ai:
    st.subheader("🤖 Ask the climate engineer")
    st.caption(
        "Ask anything about your current scenario. Works offline with a built-in explainer; "
        "if you've added an Anthropic API key in the sidebar, you'll get a richer chat."
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
            placeholder.markdown("_Thinking..._")
            try:
                # Prefer Anthropic API if key present
                api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
                if api_key_set:
                    answer = ask_anthropic(
                        messages=[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state["chat_history"]],
                        system_prompt=system_prompt,
                    )
                else:
                    raise RuntimeError("no_api_key")
            except Exception as e:
                # graceful fallback — use the smart Q&A on the actual question
                smart = offline_answer(user_q, temp_val, flood_val, drought_val, current_params)
                if str(e) == "no_api_key":
                    answer = "_(Offline mode — answering from the simulator's built-in knowledge.)_\n\n" + smart
                else:
                    answer = (
                        f"_(AI service unavailable: {e}. Falling back to built-in answer.)_\n\n" + smart
                    )
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
