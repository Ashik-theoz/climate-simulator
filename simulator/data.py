"""Reference datasets bundled with the simulator.

Sources are noted on each dataset. Anything marked *illustrative* is
hand-tuned for demonstration and should not be cited as an authoritative
hazard layer.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# London locations — characteristic profiles for the map
# Numbers below are approximate, blending London Datastore borough-level
# green-cover figures with hand-curated river-proximity scores.
# ---------------------------------------------------------------------------
LONDON_LOCATIONS: List[Dict] = [
    {"name": "South Kensington (Exhibition Road)", "lat": 51.4988, "lon": -0.1749,
     "river": 0.25, "green": 0.55, "urban": 0.70, "borough": "Kensington and Chelsea"},
    {"name": "Imperial College London",            "lat": 51.4988, "lon": -0.1749,
     "river": 0.25, "green": 0.50, "urban": 0.75, "borough": "Kensington and Chelsea"},
    {"name": "Hyde Park",                          "lat": 51.5073, "lon": -0.1657,
     "river": 0.20, "green": 0.95, "urban": 0.10, "borough": "Westminster"},
    {"name": "Westminster",                        "lat": 51.4995, "lon": -0.1248,
     "river": 0.85, "green": 0.30, "urban": 0.85, "borough": "Westminster"},
    {"name": "Canary Wharf",                       "lat": 51.5054, "lon": -0.0235,
     "river": 0.95, "green": 0.20, "urban": 0.95, "borough": "Tower Hamlets"},
    {"name": "Greenwich",                          "lat": 51.4826, "lon":  0.0077,
     "river": 0.90, "green": 0.55, "urban": 0.55, "borough": "Greenwich"},
    {"name": "Hackney",                            "lat": 51.5450, "lon": -0.0553,
     "river": 0.40, "green": 0.45, "urban": 0.75, "borough": "Hackney"},
    {"name": "Richmond",                           "lat": 51.4613, "lon": -0.3037,
     "river": 0.65, "green": 0.80, "urban": 0.40, "borough": "Richmond upon Thames"},
    {"name": "Croydon",                            "lat": 51.3762, "lon": -0.0982,
     "river": 0.20, "green": 0.40, "urban": 0.80, "borough": "Croydon"},
    {"name": "Heathrow Area",                      "lat": 51.4700, "lon": -0.4543,
     "river": 0.30, "green": 0.35, "urban": 0.85, "borough": "Hillingdon"},
    {"name": "Stratford / Olympic Park",           "lat": 51.5417, "lon": -0.0036,
     "river": 0.55, "green": 0.65, "urban": 0.65, "borough": "Newham"},
    {"name": "Lambeth (South Bank)",               "lat": 51.4946, "lon": -0.1115,
     "river": 0.90, "green": 0.30, "urban": 0.90, "borough": "Lambeth"},
]


# ---------------------------------------------------------------------------
# HadCRUT5-style historical global mean surface temperature anomaly
# Annual values are decadal anchors approximating Met Office HadCRUT5
# (relative to 1961-1990 baseline; here re-baselined to roughly 1850-1900).
# Source: Morice, C. P., et al. (2021). An updated assessment of near-surface
# temperature change from 1850. JGR-Atmospheres, 126(3), e2019JD032361.
# Numbers are decade anchors and should be treated as illustrative.
# ---------------------------------------------------------------------------
HADCRUT_HISTORICAL: List[Tuple[int, float]] = [
    (1850, -0.41),
    (1880, -0.30),
    (1900, -0.10),
    (1920, -0.18),
    (1940,  0.04),
    (1960, -0.04),
    (1980,  0.16),
    (2000,  0.49),
    (2010,  0.84),
    (2015,  0.94),
    (2020,  1.05),
    (2023,  1.45),
]


# ---------------------------------------------------------------------------
# IPCC AR6 SSP reference scenarios — best-estimate end-of-century warming
# Source: IPCC AR6 WG1 SPM (2021), Table SPM.1.
# Used as horizontal/dotted reference lines on temperature plots.
# ---------------------------------------------------------------------------
SSP_SCENARIOS: Dict[str, Dict] = {
    "SSP1-1.9": {"warming_2100": 1.4, "color": "#10b981",
                 "label": "Paris-aligned (≈1.5°C)",
                 "description": "Strong mitigation, net-zero ~2050."},
    "SSP1-2.6": {"warming_2100": 1.8, "color": "#22d3ee",
                 "label": "Strong mitigation",
                 "description": "Sustainability pathway, net-zero ~2070."},
    "SSP2-4.5": {"warming_2100": 2.7, "color": "#fbbf24",
                 "label": "Middle of the road",
                 "description": "Current-policy continuation."},
    "SSP3-7.0": {"warming_2100": 3.6, "color": "#fb923c",
                 "label": "Regional rivalry",
                 "description": "Fragmented climate policy."},
    "SSP5-8.5": {"warming_2100": 4.4, "color": "#ef4444",
                 "label": "Fossil-fuelled growth",
                 "description": "High-emissions baseline."},
}


# ---------------------------------------------------------------------------
# Pre-canned, research-anchored quick scenarios
# ---------------------------------------------------------------------------
SCENARIO_LIBRARY: Dict[str, Dict] = {
    "🌱 Paris-aligned (SSP1-2.6)": dict(co2_ppm=420, rainfall_change_pct=4,
                                        green_infra_pct=70, urbanization_pct=40),
    "🛣️ Current trajectory (SSP2-4.5)": dict(co2_ppm=520, rainfall_change_pct=8,
                                              green_infra_pct=30, urbanization_pct=60),
    "🏭 No mitigation (SSP5-8.5)": dict(co2_ppm=750, rainfall_change_pct=15,
                                         green_infra_pct=15, urbanization_pct=80),
    "🏗️ London 2050 net-zero plan": dict(co2_ppm=400, rainfall_change_pct=6,
                                          green_infra_pct=55, urbanization_pct=55),
    "🌊 Thames Estuary 2100 high-end": dict(co2_ppm=600, rainfall_change_pct=18,
                                             green_infra_pct=25, urbanization_pct=70),
    "🏙️ Heavily urbanized": dict(co2_ppm=520, rainfall_change_pct=12,
                                  green_infra_pct=15, urbanization_pct=85),
    "🌲 Green city": dict(co2_ppm=380, rainfall_change_pct=4,
                          green_infra_pct=80, urbanization_pct=30),
}


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
REFERENCES: List[Dict[str, str]] = [
    {"key": "Myhre1998",
     "text": "Myhre, G., Highwood, E. J., Shine, K. P., & Stordal, F. (1998). New estimates of radiative forcing due to well mixed greenhouse gases. Geophysical Research Letters, 25(14), 2715-2718.",
     "doi": "10.1029/98GL01908"},
    {"key": "IPCC_AR6_WG1",
     "text": "IPCC (2021). Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report. Cambridge University Press.",
     "doi": "10.1017/9781009157896"},
    {"key": "Morice2021",
     "text": "Morice, C. P., et al. (2021). An updated assessment of near-surface temperature change from 1850: The HadCRUT5 data set. Journal of Geophysical Research: Atmospheres, 126(3).",
     "doi": "10.1029/2019JD032361"},
    {"key": "EA_FloodMap",
     "text": "Environment Agency (2024). Flood Map for Planning (Rivers and Sea). UK Government Open Data.",
     "doi": "https://environment.data.gov.uk/"},
    {"key": "LondonDatastore_Greenspace",
     "text": "Greater London Authority (2023). London Tree Canopy Cover & Greenspace. London Datastore.",
     "doi": "https://data.london.gov.uk/"},
    {"key": "ThamesEstuary2100",
     "text": "Environment Agency (2023). Thames Estuary 2100: 10-Year Review. UK Government.",
     "doi": "https://www.gov.uk/government/publications/thames-estuary-2100-te2100"},
    {"key": "WongPalette",
     "text": "Wong, B. (2011). Color blindness. Nature Methods, 8(6), 441.",
     "doi": "10.1038/nmeth.1618"},
]


# Color-blind-safe palette (Wong 2011) for plots
WONG_PALETTE: Dict[str, str] = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "skyblue": "#56B4E9",
    "vermilion": "#D55E00",
    "magenta": "#CC79A7",
    "black":   "#000000",
}
