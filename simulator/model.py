"""Climate engineering simulator — core model.

This is a deliberately simple 0-D educational model. It is **not** a GCM-grade
climate model and should not be used for policy or insurance decisions. Its
purpose is to illustrate, in real time, how engineering choices (urbanization,
green infrastructure) interact with emissions and rainfall trajectories to
shape local flood and drought risk.

Equations
---------
Temperature anomaly relative to pre-industrial:
    ΔT(t) = λ · ln(C/C₀) · (1 − exp(−t/τ))
Logarithmic forcing follows the form of Myhre et al. (1998); the τ-lag is a
first-order representation of ocean thermal inertia.

Runoff index (unitless):
    R(t) = (1 + δp) · (a + b·U) · (1 − c·G) · (1 + d·ΔT(t))
where U = urbanization fraction, G = green-infrastructure fraction,
δp = fractional rainfall change.

Flood risk (0-100):
    F(t) = 100 · (1 − exp(−k_F · R(t)))

Evaporation index (unitless):
    E(t) = (1 + e·ΔT(t)) / (1 + δp) · (1 − f·G)

Drought risk (0-100):
    D(t) = 100 · (1 − exp(−k_D · E(t)))

All coefficients are illustrative — calibrated to produce demonstration
behaviour, not validated against gauge or remote-sensing observations.

References
----------
- Myhre, G., et al. (1998). New estimates of radiative forcing due to well
  mixed greenhouse gases. Geophysical Research Letters, 25(14), 2715-2718.
  doi:10.1029/98GL01908
- IPCC (2021). Climate Change 2021: The Physical Science Basis. Working
  Group I contribution to AR6. Cambridge University Press.
- Pachauri, R.K. & Meyer, L.A. (2014). IPCC AR5 Synthesis Report.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

MODEL_VERSION: str = "0.3.0"

# Default model coefficients (illustrative, not calibrated to observations)
DEFAULT_PARAMS: Dict[str, float] = {
    "lambda": 1.2,        # K, sensitivity coefficient on ln(C/C0)
    "tau": 25.0,          # yr, e-folding time of temperature response
    "C0": 280.0,          # ppm, pre-industrial CO2
    "a": 0.6,             # baseline runoff coefficient
    "b": 1.2,             # urban fraction → runoff multiplier
    "c": 0.55,            # green fraction → runoff dampening
    "d": 0.08,            # warming → runoff amplification
    "k_flood": 0.9,       # flood risk saturation rate
    "e": 0.18,            # warming → evaporation
    "f": 0.15,            # green fraction → evaporation dampening
    "k_drought": 0.8,     # drought risk saturation rate
}


def temperature_anomaly(t: np.ndarray, co2_ppm: float,
                        params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Logarithmic CO₂ forcing with first-order temperature lag."""
    p = params or DEFAULT_PARAMS
    eq = p["lambda"] * np.log(max(co2_ppm, 1e-6) / p["C0"])
    return eq * (1 - np.exp(-t / p["tau"]))


def simulate(years: int = 80,
             co2_ppm: float = 450,
             rainfall_change_pct: float = 10,
             green_infra_pct: float = 20,
             urbanization_pct: float = 40,
             params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Run the simulator for `years` years starting in 2025.

    Returns a DataFrame with columns:
        year, temp_anomaly_C, flood_risk, drought_risk, runoff_index, evap_index
    """
    p = params or DEFAULT_PARAMS
    t = np.arange(int(years) + 1)
    temp_series = temperature_anomaly(t, co2_ppm, p)

    rainfall_factor = 1 + rainfall_change_pct / 100.0
    impervious = max(0.0, min(1.0, urbanization_pct / 100.0))
    green = max(0.0, min(1.0, green_infra_pct / 100.0))

    runoff_index = rainfall_factor * (p["a"] + p["b"] * impervious) \
        * (1 - p["c"] * green) * (1 + p["d"] * temp_series)
    flood_risk = 100 * (1 - np.exp(-p["k_flood"] * runoff_index))

    evap_index = (1 + p["e"] * temp_series) / rainfall_factor * (1 - p["f"] * green)
    drought_risk = 100 * (1 - np.exp(-p["k_drought"] * evap_index))

    return pd.DataFrame({
        "year": 2025 + t,
        "temp_anomaly_C": temp_series,
        "flood_risk": np.clip(flood_risk, 0, 100),
        "drought_risk": np.clip(drought_risk, 0, 100),
        "runoff_index": runoff_index,
        "evap_index": evap_index,
    })


def monte_carlo(years: int = 80,
                co2_ppm: float = 450,
                rainfall_change_pct: float = 10,
                green_infra_pct: float = 20,
                urbanization_pct: float = 40,
                n_samples: int = 300,
                co2_sigma_ppm: float = 25,
                rainfall_sigma_pct: float = 4,
                green_sigma_pct: float = 5,
                urban_sigma_pct: float = 5,
                seed: int = 42) -> pd.DataFrame:
    """Monte Carlo uncertainty propagation.

    Each input is perturbed by an independent Gaussian and the simulator is
    re-run `n_samples` times. Returns p05 / p50 / p95 envelopes across years.
    """
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(int(n_samples)):
        c = max(280, co2_ppm + rng.normal(0, co2_sigma_ppm))
        r = rainfall_change_pct + rng.normal(0, rainfall_sigma_pct)
        g = float(np.clip(green_infra_pct + rng.normal(0, green_sigma_pct), 0, 100))
        u = float(np.clip(urbanization_pct + rng.normal(0, urban_sigma_pct), 0, 100))
        runs.append(simulate(years=years, co2_ppm=c, rainfall_change_pct=r,
                             green_infra_pct=g, urbanization_pct=u))

    temp = np.stack([d["temp_anomaly_C"].values for d in runs])
    flood = np.stack([d["flood_risk"].values for d in runs])
    drought = np.stack([d["drought_risk"].values for d in runs])

    return pd.DataFrame({
        "year": runs[0]["year"].values,
        "temp_p05": np.percentile(temp, 5, axis=0),
        "temp_p50": np.percentile(temp, 50, axis=0),
        "temp_p95": np.percentile(temp, 95, axis=0),
        "flood_p05": np.percentile(flood, 5, axis=0),
        "flood_p50": np.percentile(flood, 50, axis=0),
        "flood_p95": np.percentile(flood, 95, axis=0),
        "drought_p05": np.percentile(drought, 5, axis=0),
        "drought_p50": np.percentile(drought, 50, axis=0),
        "drought_p95": np.percentile(drought, 95, axis=0),
    })


def sensitivity_analysis(years: int = 80,
                         co2_ppm: float = 450,
                         rainfall_change_pct: float = 10,
                         green_infra_pct: float = 20,
                         urbanization_pct: float = 40,
                         delta_pct: float = 0.20) -> pd.DataFrame:
    """One-at-a-time sensitivity (Δ ±delta_pct) on flood / drought / temperature.

    Returns a tidy DataFrame ready for a tornado plot:
        parameter, direction, delta_temp, delta_flood, delta_drought
    """
    base = simulate(years=years, co2_ppm=co2_ppm,
                    rainfall_change_pct=rainfall_change_pct,
                    green_infra_pct=green_infra_pct,
                    urbanization_pct=urbanization_pct)
    base_t = base["temp_anomaly_C"].iloc[-1]
    base_f = base["flood_risk"].iloc[-1]
    base_d = base["drought_risk"].iloc[-1]

    knobs = {
        "co2_ppm": co2_ppm,
        "rainfall_change_pct": rainfall_change_pct,
        "green_infra_pct": green_infra_pct,
        "urbanization_pct": urbanization_pct,
    }
    rows = []
    for key, val in knobs.items():
        for direction, delta in [("−20%", -delta_pct), ("+20%", +delta_pct)]:
            new = val * (1 + delta) if val != 0 else val + (delta * 10)
            if "pct" in key and "co2" not in key:
                new = float(np.clip(new, 0, 100))
            elif key == "co2_ppm":
                new = max(280, new)
            kw = {**knobs, key: new}
            df = simulate(years=years, **kw)
            rows.append({
                "parameter": key,
                "direction": direction,
                "delta_temp": df["temp_anomaly_C"].iloc[-1] - base_t,
                "delta_flood": df["flood_risk"].iloc[-1] - base_f,
                "delta_drought": df["drought_risk"].iloc[-1] - base_d,
            })
    return pd.DataFrame(rows)


def local_risk(loc: Dict[str, float],
               flood_global: float, drought_global: float,
               user_green_pct: float, user_urban_pct: float):
    """Modulate global risks for a specific London location."""
    river = loc.get("river", 0.3)
    green = loc.get("green", 0.4)
    urban = loc.get("urban", 0.6)
    combined_green = 0.5 * green + 0.5 * (user_green_pct / 100.0)
    combined_urban = 0.5 * urban + 0.5 * (user_urban_pct / 100.0)
    flood_local = flood_global * (0.7 + 0.6 * river) * (0.7 + 0.6 * combined_urban) * (1 - 0.35 * combined_green)
    drought_local = drought_global * (0.8 + 0.4 * combined_urban) * (1 - 0.30 * combined_green)
    return float(np.clip(flood_local, 0, 100)), float(np.clip(drought_local, 0, 100))
