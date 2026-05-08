"""Climate engineering simulator — pure-Python core.

Importable as:

    from simulator.model import simulate, monte_carlo, sensitivity_analysis
    from simulator.data import LONDON_LOCATIONS, SSP_SCENARIOS, HADCRUT_HISTORICAL

The Streamlit UI in app.py is a thin presentation layer over this package.
"""

from simulator.model import (
    simulate,
    monte_carlo,
    sensitivity_analysis,
    temperature_anomaly,
    MODEL_VERSION,
    DEFAULT_PARAMS,
)
from simulator.data import (
    LONDON_LOCATIONS,
    SSP_SCENARIOS,
    HADCRUT_HISTORICAL,
    REFERENCES,
)

__all__ = [
    "simulate",
    "monte_carlo",
    "sensitivity_analysis",
    "temperature_anomaly",
    "MODEL_VERSION",
    "DEFAULT_PARAMS",
    "LONDON_LOCATIONS",
    "SSP_SCENARIOS",
    "HADCRUT_HISTORICAL",
    "REFERENCES",
]
