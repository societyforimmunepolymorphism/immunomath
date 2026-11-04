"""Plotly figures built from a saved analysis record.

Every function takes the plain ``dict`` produced by
:meth:`~immunomath.analysis.AnalysisResult.to_dict` (or read back with
:func:`~immunomath.reporting.export.read_json`), never a live cohort, so figures
can be regenerated from saved results without re-running any inference.

Plotly is an optional dependency: ``pip install 'immunomath[reporting]'``.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["loss_histogram_figure", "reliability_figure", "risk_coverage_figure"]


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError("Plotting requires plotly: pip install 'immunomath[reporting]'.") from exc
    return go


def reliability_figure(calibration: dict[str, Any], title: str | None = None) -> Any:
    """Reliability diagram: observed rate against mean predicted probability.

    The dashed diagonal is perfect calibration. Points are non-empty bins with
    Wilson intervals as error bars; marker area scales with bin count, and the
    hover text states the count so sparse bins are not over-read.
    """
    go = _go()
    bins = [b for b in calibration["bins"] if b["count"] > 0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="perfect calibration",
            hoverinfo="skip",
        )
    )
    if bins:
        counts = [b["count"] for b in bins]
        biggest = max(counts)
        fig.add_trace(
            go.Scatter(
                x=[b["mean_predicted"] for b in bins],
                y=[b["observed_rate"] for b in bins],
                mode="markers+lines",
                name="observed",
                marker={"size": [8 + 22 * math.sqrt(c / biggest) for c in counts]},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [b["ci_high"] - b["observed_rate"] for b in bins],
                    "arrayminus": [b["observed_rate"] - b["ci_low"] for b in bins],
                },
                customdata=[[b["count"], b["lower"], b["upper"]] for b in bins],
                hovertemplate=(
                    "predicted %{x:.3f}<br>observed %{y:.3f}<br>"
                    "n = %{customdata[0]}<br>bin [%{customdata[1]:.3f}, "
                    "%{customdata[2]:.3f}]<extra></extra>"
                ),
            )
        )
    level = round(100 * calibration["ci_confidence"])
    fig.update_layout(
        title=title
        or f"Reliability: {calibration['label']} ({calibration['binning']}, "
        f"{calibration['n_bins_requested']} bins; bars = {level}% Wilson)",
        xaxis_title="mean predicted probability",
        yaxis_title="observed frequency",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
    )
    return fig


def risk_coverage_figure(risk_coverage: dict[str, Any]) -> Any:
    """Two-panel risk-coverage figure: error rate and mean log loss vs coverage.

    Operating points whose mean log loss is infinite cannot be drawn; they are
    listed in the panel title rather than silently omitted.
    """
    go = _go()
    from plotly.subplots import make_subplots

    points = risk_coverage["points"]
    n_inf = sum(1 for p in points if math.isinf(p["mean_log_loss"]))
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Error rate vs coverage",
            f"Mean log loss vs coverage ({n_inf} infinite point(s) not drawn)",
        ),
    )
    cov = [p["coverage"] for p in points]
    fig.add_trace(
        go.Scatter(
            x=cov, y=[p["error_rate"] for p in points], mode="lines+markers", name="error rate"
        ),
        row=1,
        col=1,
    )
    finite = [p for p in points if math.isfinite(p["mean_log_loss"])]
    fig.add_trace(
        go.Scatter(
            x=[p["coverage"] for p in finite],
            y=[p["mean_log_loss"] for p in finite],
            mode="lines+markers",
            name="mean log loss (nats)",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="coverage", range=[0, 1])
    fig.update_yaxes(title_text="error rate", row=1, col=1)
    fig.update_yaxes(title_text="mean log loss (nats)", row=1, col=2)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def loss_histogram_figure(subjects: dict[str, list[Any]], n_bins: int = 40) -> Any:
    """Histogram of finite per-subject log losses.

    Infinite losses cannot be placed on an axis; their number is stated in the
    title so the tail is never hidden. Undefined subjects are excluded and noted.
    """
    go = _go()
    scores = subjects["log_score"]
    classes = subjects["score_class"]
    finite = [s for s, c in zip(scores, classes, strict=True) if c == "finite"]
    n_inf = sum(1 for c in classes if c == "infinite")
    n_undef = sum(1 for c in classes if c == "undefined")
    fig = go.Figure(go.Histogram(x=finite, nbinsx=n_bins, name="finite losses"))
    fig.update_layout(
        title=(
            f"Distribution of per-subject log loss "
            f"({len(finite)} finite; {n_inf} infinite not drawn; {n_undef} undefined excluded)"
        ),
        xaxis_title="negative log score (nats)",
        yaxis_title="subjects",
        template="plotly_white",
    )
    return fig
