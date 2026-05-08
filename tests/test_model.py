"""Sanity tests for the simulator core."""

import numpy as np
import pytest

from simulator.model import (
    simulate, monte_carlo, sensitivity_analysis,
    temperature_anomaly, local_risk, MODEL_VERSION,
)


def test_simulate_basic_shape():
    df = simulate(years=80)
    assert len(df) == 81
    for col in ("year", "temp_anomaly_C", "flood_risk", "drought_risk"):
        assert col in df.columns


def test_higher_co2_means_more_warming():
    a = simulate(co2_ppm=400)["temp_anomaly_C"].iloc[-1]
    b = simulate(co2_ppm=600)["temp_anomaly_C"].iloc[-1]
    assert b > a


def test_more_green_means_less_flood():
    a = simulate(green_infra_pct=10)["flood_risk"].iloc[-1]
    b = simulate(green_infra_pct=70)["flood_risk"].iloc[-1]
    assert b < a


def test_more_urban_means_more_flood():
    a = simulate(urbanization_pct=20)["flood_risk"].iloc[-1]
    b = simulate(urbanization_pct=80)["flood_risk"].iloc[-1]
    assert b > a


def test_temperature_starts_at_zero():
    t = np.array([0.0])
    assert abs(temperature_anomaly(t, 450)[0]) < 1e-12


def test_risks_are_bounded():
    df = simulate()
    assert (df["flood_risk"] >= 0).all() and (df["flood_risk"] <= 100).all()
    assert (df["drought_risk"] >= 0).all() and (df["drought_risk"] <= 100).all()


def test_pre_industrial_co2_zero_warming():
    df = simulate(co2_ppm=280)
    assert abs(df["temp_anomaly_C"].iloc[-1]) < 1e-9


def test_monte_carlo_envelope_ordered():
    mc = monte_carlo(n_samples=100)
    assert (mc["temp_p95"] >= mc["temp_p50"]).all()
    assert (mc["temp_p50"] >= mc["temp_p05"]).all()


def test_sensitivity_returns_all_knobs():
    s = sensitivity_analysis()
    assert set(s["parameter"].unique()) == {
        "co2_ppm", "rainfall_change_pct", "green_infra_pct", "urbanization_pct",
    }
    # Each knob exercised in both directions
    assert (s.groupby("parameter").size() == 2).all()


def test_local_risk_river_amplifies_flood():
    riverside = {"river": 0.95, "green": 0.20, "urban": 0.95}
    inland   = {"river": 0.10, "green": 0.50, "urban": 0.50}
    f1, _ = local_risk(riverside, 60, 60, 20, 60)
    f2, _ = local_risk(inland,    60, 60, 20, 60)
    assert f1 > f2


def test_model_version_string():
    assert isinstance(MODEL_VERSION, str)
    assert len(MODEL_VERSION.split(".")) == 3
