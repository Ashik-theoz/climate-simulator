"""One-page PDF report generator for a simulator scenario."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

from simulator.model import MODEL_VERSION


def _fig_to_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_report(df: pd.DataFrame,
                 mc: Optional[pd.DataFrame],
                 params: Dict,
                 scenario_label: str = "Custom scenario",
                 author: str = "Ashikujjaman Mohammad",
                 institution: str = "Imperial College London") -> bytes:
    """Render a one-page A4 PDF report and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.6*cm, rightMargin=1.6*cm,
                            topMargin=1.4*cm, bottomMargin=1.4*cm)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#0e3b66"),
                        fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#0e3b66"),
                        fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8,
                           leading=10, textColor=colors.HexColor("#555555"))

    story = []

    # Title block
    story.append(Paragraph(f"London Climate Engineering Simulator", h1))
    story.append(Paragraph(
        f"<b>Scenario:</b> {scenario_label} &nbsp;·&nbsp; "
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; "
        f"<b>Model:</b> v{MODEL_VERSION}", small))
    story.append(Spacer(1, 0.3*cm))

    # Headline metrics
    last = df.iloc[-1]
    headline_data = [
        ["End-of-horizon warming", f"{last['temp_anomaly_C']:.2f} °C"],
        ["End-of-horizon flood risk", f"{last['flood_risk']:.0f} / 100"],
        ["End-of-horizon drought risk", f"{last['drought_risk']:.0f} / 100"],
        ["Simulation horizon", f"{params['years']} years (2025–{2025+params['years']})"],
    ]
    table = Table(headline_data, colWidths=[6*cm, 5*cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f3f6fb"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde3ec")),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#dde3ec")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    # Charts
    story.append(Paragraph("Projections", h2))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax = axes[0]
    if mc is not None:
        ax.fill_between(mc["year"], mc["temp_p05"], mc["temp_p95"], alpha=0.18,
                        color="#0072B2", label="5–95% band")
    ax.plot(df["year"], df["temp_anomaly_C"], color="#0072B2", linewidth=2)
    ax.set_title("Warming (°C)", fontsize=10)
    ax.set_xlabel("Year"); ax.set_ylabel("°C")
    ax.grid(True, linestyle="--", alpha=0.35); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    if mc is not None:
        ax.fill_between(mc["year"], mc["flood_p05"], mc["flood_p95"], alpha=0.15,
                        color="#0072B2", label="Flood band")
        ax.fill_between(mc["year"], mc["drought_p05"], mc["drought_p95"], alpha=0.15,
                        color="#E69F00", label="Drought band")
    ax.plot(df["year"], df["flood_risk"], color="#0072B2", linewidth=2, label="Flood")
    ax.plot(df["year"], df["drought_risk"], color="#E69F00", linewidth=2, label="Drought")
    ax.set_title("Risk indices (0–100)", fontsize=10)
    ax.set_xlabel("Year"); ax.set_ylabel("Risk")
    ax.set_ylim(0, 100); ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    img_buf = _fig_to_bytes(fig)
    story.append(Image(img_buf, width=17*cm, height=6*cm))

    # Parameters
    story.append(Paragraph("Parameter manifest", h2))
    pm = [
        ["CO₂ concentration", f"{params['co2_ppm']} ppm"],
        ["Rainfall change", f"{params['rainfall_change_pct']:+d}%"],
        ["Green infrastructure", f"{params['green_infra_pct']}%"],
        ["Urbanization", f"{params['urbanization_pct']}%"],
    ]
    pt = Table(pm, colWidths=[6*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f3f6fb"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.3, colors.HexColor("#dde3ec")),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#dde3ec")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pt)

    # Methodology summary + caveats
    story.append(Paragraph("Methodology (summary)", h2))
    story.append(Paragraph(
        "Temperature follows a logarithmic CO₂ forcing with a first-order lag (τ = 25 yr); "
        "flood and drought are unitless 0–100 indices derived from runoff and evaporation proxies "
        "modulated by green-infrastructure and urbanization fractions. Model coefficients are "
        "illustrative and have not been calibrated against observed gauge or reanalysis data. "
        "Uncertainty bands shown above (when present) are 5–95% Gaussian Monte Carlo intervals "
        "with default σ = 25 ppm CO₂, ±4% rainfall, ±5% green/urban fractions.",
        body))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Limitations", h2))
    story.append(Paragraph(
        "This is a 0-D educational simulator, <b>not</b> a GCM. It does not capture spatial "
        "dynamics, ocean circulation, regional precipitation patterns, ice-sheet feedbacks, or "
        "climate tipping points. Outputs should be interpreted as illustrative trajectories of "
        "<i>relative</i> behaviour under engineering choices, not as quantitative forecasts of "
        "absolute risk.",
        body))

    # Citation
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"<b>Cite as:</b> {author} ({datetime.now().year}). London Climate Engineering Simulator "
        f"(v{MODEL_VERSION}). {institution}.",
        small))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
