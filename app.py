import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
    # main
    "mode": "Standard",
    "years": 80,
    "co2_ppm": 450,
    "rainfall_change_pct": 10,
    "green_infra_pct": 20,
    "urbanization_pct": 40,
    # challenge
    "challenge_on": False,
    "challenge_won": False,
    "difficulty_choice": "Medium",  # NOTE: widget key is difficulty_choice (not difficulty)
    # scenario compare
    "compare_on": False,
    "scenario_A": None,
    "scenario_B": None,
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
    """Store params + dataframe snapshot for scenario comparison."""
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
# Sidebar UI (ALL controls live here)
# ----------------------------
with st.sidebar:
    st.header("Controls")

    # Reset everything (avoid Streamlit key errors by clearing widget keys first)
    if st.button("🔄 Reset to Default"):
        for key in [
            "mode",
            "years",
            "co2_ppm",
            "rainfall_change_pct",
            "green_infra_pct",
            "urbanization_pct",
            "challenge_on",
            "challenge_won",
            "difficulty_choice",
            "compare_on",
            "scenario_A",
            "scenario_B",
        ]:
            st.session_state.pop(key, None)

        # re-seed defaults
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

    # Mode (Standard / Kids)
    st.radio("Mode", ["Standard", "Kids (simple)"], key="mode")

    # Main sliders (always present)
    st.slider("Simulation horizon (years)", 20, 120, step=5, key="years")
    st.slider("CO₂ concentration (ppm)", 280, 900, step=10, key="co2_ppm")
    st.slider("Rainfall change (%)", -30, 50, step=1, key="rainfall_change_pct")

    # Kids vs Standard control depth
    if st.session_state["mode"] == "Kids (simple)":
        st.slider("Green solutions (%)", 0, 100, step=5, key="green_infra_pct")
        # fixed in kids mode (safe because urbanization slider is NOT instantiated in kids mode)
        st.session_state["urbanization_pct"] = 45
        st.info("Kids mode uses fewer controls for faster exploration.")
    else:
        st.slider("Green infrastructure (%)", 0, 100, step=5, key="green_infra_pct")
        st.slider("Urbanization / imperviousness (%)", 0, 100, step=5, key="urbanization_pct")

    st.divider()

    # ----------------------------
    # Challenge mode (Sidebar)
    # ----------------------------
    st.subheader("🎯 Challenge mode")
    st.toggle("Enable challenge", key="challenge_on")

    if st.button("🏆 Reset challenge calibration"):
        # only reset the celebration state + difficulty widget (targets re-sync automatically)
        st.session_state["challenge_won"] = False
        st.session_state.pop("difficulty_choice", None)
        st.rerun()

    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="difficulty_choice")
    target_flood = int(DIFFICULTY_TARGETS[diff]["target_flood"])
    target_drought = int(DIFFICULTY_TARGETS[diff]["target_drought"])

    st.caption(f"Targets: Flood ≤ {target_flood} | Drought ≤ {target_drought}")

    st.divider()

    # ----------------------------
    # Scenario comparison mode (Sidebar)
    # ----------------------------
    st.subheader("🧪 Scenario comparison")
    st.toggle("Enable comparison", key="compare_on")

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

# End-of-horizon values
flood_val = float(df["flood_risk"].iloc[-1])
drought_val = float(df["drought_risk"].iloc[-1])
temp_val = float(df["temp_anomaly_C"].iloc[-1])

# ----------------------------
# Comparison buttons now that df exists
# (these MUST run after df is computed)
# ----------------------------
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
# Challenge status box (compact, above graphs)
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

    # Balloons only once per NEW win
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
# Charts (professional-ish Matplotlib without forcing colors)
# ----------------------------
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

    # Target threshold lines (ONLY when challenge is on)
    if st.session_state.get("challenge_on", False):
        ax.axhline(target_flood, linestyle="--", alpha=0.6)
        ax.axhline(target_drought, linestyle="--", alpha=0.6)

    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Scenario Comparison (MAIN PAGE)
# ----------------------------
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

        # End metrics
        tA, fA, dA = float(dfA["temp_anomaly_C"].iloc[-1]), float(dfA["flood_risk"].iloc[-1]), float(dfA["drought_risk"].iloc[-1])
        tB, fB, dB = float(dfB["temp_anomaly_C"].iloc[-1]), float(dfB["flood_risk"].iloc[-1]), float(dfB["drought_risk"].iloc[-1])

        m1, m2, m3 = st.columns(3)
        m1.metric("Δ Warming (A → B)", f"{(tB - tA):+.2f} °C", help="Positive means Scenario B is warmer at end-of-horizon.")
        m2.metric("Δ Flood risk (A → B)", f"{(fB - fA):+.0f}", help="Positive means Scenario B has higher flood risk.")
        m3.metric("Δ Drought risk (A → B)", f"{(dB - dA):+.0f}", help="Positive means Scenario B has higher drought risk.")

        # Plots
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
            ax.minorticks_on()
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
            ax.minorticks_on()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="upper left")

            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        with st.expander("Show Scenario A & B parameters"):
            st.write("**Scenario A**:", A["params"])
            st.write("**Scenario B**:", B["params"])

# ----------------------------
# Explanation + Data
# ----------------------------
st.divider()
st.subheader("What’s going on here?")
st.write(
    "This is a fast, educational simulator (not a full climate model). "
    "It’s designed for interactive exploration: CO₂ affects warming, rainfall affects water stress, "
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
    
