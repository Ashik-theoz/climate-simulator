# London Climate Engineering Simulator

> Can engineering reverse the climate clock? An interactive London-scale
> simulator for the Great Exhibition Road Festival (Imperial College London).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)

**Live demo:** https://climate-simulator.streamlit.app/

---

## What it does

A real-time 0-D educational climate simulator. Visitors pull four engineering
levers — CO₂ concentration, rainfall change, urbanization, and green
infrastructure — and the app projects warming, flood risk and drought risk
across a configurable horizon (20–120 years). Outputs are plotted with
optional 5–95% Monte Carlo uncertainty bands and anchored against the IPCC
AR6 SSP reference scenarios. A live London map shows neighbourhood-level risk
hotspots with animated indicators, and a built-in AI assistant explains the
physics on demand.

The model is **not** a GCM. It is intentionally simple, designed for
intuition-building and outreach. Methodology, equations, coefficients and
limitations are documented inside the app's "📚 Methods" tab and in
[`simulator/model.py`](simulator/model.py).

## Features

- **Interactive simulation** with sliders and pre-canned IPCC-anchored scenarios.
- **Monte Carlo uncertainty bands** (5–95%) on temperature, flood and drought.
- **IPCC SSP overlays** (SSP1-1.9 through SSP5-8.5) on the temperature plot.
- **HadCRUT5-style historical anomaly** spliced onto projection plots.
- **Sensitivity tornado** showing which lever matters most.
- **Animated London map** with climate-emoji markers that pulse on high risk.
- **Streaming AI assistant** (Anthropic Claude Haiku 4.5) with a smart offline fallback.
- **One-page PDF report** download.
- **Shareable scenario URLs** that encode every slider in the query string.
- **CSV / JSON export** with a model-version manifest for reproducibility.
- **Color-blind-safe palette** (Wong 2011) on all plots.

## Run locally

```bash
git clone https://github.com/Ashik-theoz/climate-simulator.git
cd climate-simulator
pip install -r requirements.txt
streamlit run app.py
```

## Optional: enable streaming AI

The chat tab uses Anthropic Claude Haiku 4.5 when an API key is present.
Without one, a smart rule-based fallback handles "where", "why" and
"how to reduce" questions out of the box.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

For Streamlit Cloud, paste the same line into *Manage app → Settings → Secrets*.

## Test

```bash
pip install pytest
pytest
```

## Project layout

```
.
├── app.py                  # Streamlit UI (presentation only)
├── simulator/
│   ├── __init__.py
│   ├── model.py            # equations, Monte Carlo, sensitivity
│   ├── data.py             # London locations, SSPs, HadCRUT, references
│   └── report.py           # PDF generator
├── tests/
│   └── test_model.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Cite

```bibtex
@software{mohammad_2026_climate_simulator,
  author       = {Mohammad, Ashikujjaman},
  title        = {London Climate Engineering Simulator},
  version      = {0.3.0},
  year         = 2026,
  publisher    = {Imperial College London},
  url          = {https://climate-simulator.streamlit.app/}
}
```

A [`CITATION.cff`](CITATION.cff) is included so GitHub auto-renders this.

## Limitations

This is a **0-D educational simulator**. It does not capture spatial
dynamics, ocean circulation, regional precipitation patterns, ice-sheet
feedbacks, or climate tipping points. Outputs should be interpreted as
illustrative trajectories of *relative* behaviour under engineering choices,
not as quantitative forecasts of absolute risk. Do not use for policy,
planning, insurance or operational decisions. See the in-app *Methods* tab
for full caveats and references.

## License

[MIT](LICENSE) © 2026 Ashikujjaman Mohammad
