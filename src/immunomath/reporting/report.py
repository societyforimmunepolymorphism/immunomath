"""Self-contained HTML reports rendered from a saved analysis record.

:func:`render_report` takes the plain ``dict`` from
:meth:`~immunomath.analysis.AnalysisResult.to_dict` (or
:func:`~immunomath.reporting.export.read_json`) and writes one HTML file. It
needs no cohort and no inference output, so a report can be regenerated
whenever the saved JSON is available.

Every report states its analysis metadata, denominators, exclusions, methods,
assumptions and automatically generated warnings. All dynamic text is
HTML-escaped.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from .plots import loss_histogram_figure, reliability_figure, risk_coverage_figure

__all__ = ["collect_warnings", "render_report"]

#: Bins with fewer subjects than this get a wide-interval warning.
SMALL_BIN = 10
#: Cohorts smaller than this get a small-sample warning.
SMALL_COHORT = 30


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, bool) or x is None:
        return "—" if x is None else str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "+∞" if x > 0 else "−∞"
        return f"{x:.{digits}g}"
    return str(x)


def _esc(x: Any) -> str:
    return html.escape(_fmt(x) if not isinstance(x, str) else x)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def collect_warnings(record: dict[str, Any]) -> list[str]:
    """Return human-readable warnings implied by an analysis record.

    Warnings are generated, not hand-written, so they cannot be forgotten:
    infinite losses, excluded subjects, an unset seed, small cohorts, sparse
    calibration bins, an applied probability floor, and empty event tables.
    """
    out: list[str] = []
    d = record["denominators"]
    params = record["metadata"]["parameters"]
    summary = record["summary"]

    if params.get("seed") is None:
        out.append("No random seed was set: bootstrap intervals are not reproducible.")
    if d["n_infinite"] > 0:
        out.append(
            f"{d['n_infinite']} subject(s) assigned the truth exactly zero probability. "
            "Their log loss is +∞, so the primary mean log score is +∞ and any "
            "confidence bound reaching them is +∞. They are reported, never dropped."
        )
    if d["n_undefined"] > 0:
        out.append(
            f"{d['n_undefined']} of {d['n_total']} subject(s) were excluded because the "
            "truth probability is not identifiable (open-world residual mass). The mean "
            f"applies only to the {d['n_scored']} identifiable subjects."
        )
    if 0 < d["n_scored"] < SMALL_COHORT:
        out.append(f"Only {d['n_scored']} identifiable subjects: all intervals are wide.")
    if params.get("probability_floor") is not None:
        out.append(
            f"A probability floor of {params['probability_floor']} was applied to the log "
            "score. This is a numerical experiment, not the unmodified proper score."
        )
    ci = summary.get("mean_log_score_ci")
    if ci and ci.get("n_infinite_resamples", 0) > 0:
        out.append(
            f"{ci['n_infinite_resamples']} of {ci['n_resamples']} bootstrap resamples "
            "contained an infinite loss; the affected bound is +∞, not a finite estimate."
        )

    cals = {
        "top-label": record["calibration"],
        **{f"event '{k}'": v for k, v in record["event_calibration"].items()},
    }
    for name, cal in cals.items():
        sparse = [b for b in cal["bins"] if 0 < b["count"] < SMALL_BIN]
        if sparse:
            out.append(
                f"Calibration ({name}): {len(sparse)} bin(s) hold fewer than {SMALL_BIN} "
                "subjects; their observed rates and intervals are unreliable."
            )
        if cal["n_evaluated"] == 0:
            out.append(f"Calibration ({name}): no subject could be evaluated.")
        elif cal["n_undefined"] > 0:
            out.append(
                f"Calibration ({name}): {cal['n_undefined']} subject(s) excluded as "
                "not identifiable."
            )
    return out


_CSS = """
body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}
h1{margin-bottom:0}h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.6rem 0}th,td{border:1px solid #ccc;padding:.25rem .6rem;
text-align:right}th{background:#f4f4f4}td:first-child,th:first-child{text-align:left}
.warn{background:#fff4e5;border-left:4px solid #e08a00;padding:.5rem 1rem}
.note{color:#555;font-size:.92rem}code{background:#f4f4f4;padding:0 .2rem}
"""


def render_report(
    record: dict[str, Any],
    path: str | Path,
    *,
    plotlyjs: Literal["inline", "cdn"] = "inline",
) -> Path:
    """Write a self-contained HTML report.

    Parameters
    ----------
    record : dict
        An analysis record (``AnalysisResult.to_dict()`` or ``read_json(...)``).
    path : str or Path
        Output file.
    plotlyjs : {"inline", "cdn"}, optional
        ``"inline"`` (default) embeds the Plotly library (about 4 MB) so the file
        works offline and is genuinely self-contained. ``"cdn"`` links to it and
        needs network access when the file is opened.

    Returns
    -------
    Path
        The written path.
    """
    if record.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version {record.get('schema_version')!r}.")
    meta = record["metadata"]
    params = meta["parameters"]
    d = record["denominators"]
    s = record["summary"]

    first_fig = [True]

    def fig_html(fig: Any) -> str:
        js: Any = plotlyjs if first_fig[0] else False
        first_fig[0] = False
        return str(fig.to_html(full_html=False, include_plotlyjs=js))

    parts: list[str] = []
    parts.append(f"<h1>{html.escape(str(meta['label']))}</h1>")
    parts.append(
        f"<p class='note'>ImmunoMath {html.escape(str(meta['immunomath_version']))} — "
        "<em>Metrognome: keeping probabilistic genetics honest.</em>"
        + (f" Created: {html.escape(str(meta['created']))}." if meta.get("created") else "")
        + "</p>"
    )

    warnings = collect_warnings(record)
    parts.append("<h2>Warnings</h2>")
    if warnings:
        parts.append(
            "<div class='warn'><ul>"
            + "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
            + "</ul></div>"
        )
    else:
        parts.append("<p>No automatic warnings were raised.</p>")

    parts.append("<h2>Denominators</h2>")
    parts.append(
        _table(
            ["quantity", "count"],
            [
                ["subjects in cohort", d["n_total"]],
                ["identifiable (finite + infinite) — denominator of the mean", d["n_scored"]],
                ["finite loss", d["n_finite"]],
                ["infinite loss (truth has zero probability)", d["n_infinite"]],
                ["undefined (truth probability not identifiable) — excluded", d["n_undefined"]],
            ],
        )
    )

    ci = s.get("mean_log_score_ci")
    ci_text = (
        f"{_fmt(ci['low'])} to {_fmt(ci['high'])} "
        f"({round(100 * ci['confidence'])}% {ci['method']}, {ci['n_resamples']} "
        f"subject-level resamples, seed {ci['seed']})"
        if ci
        else "—"
    )
    parts.append("<h2>Primary results</h2>")
    parts.append(
        _table(
            ["metric", "value"],
            [
                [
                    f"mean negative log score (nats; over {d['n_scored']} identifiable subjects)",
                    s["mean_log_score"],
                ],
                ["bootstrap interval for the mean", ci_text],
                ["mean Brier score (sum form, range [0, 2])", s["mean_brier"]],
                ["median log score (finite subjects)", s["median_log_score"]],
                ["std. dev. of log score (finite subjects)", s["std_log_score"]],
                ["maximum log score", s["max_log_score"]],
            ],
        )
    )
    parts.append(
        "<p class='note'>The mean negative log score is the primary, strictly proper "
        "summary. It is never trimmed or replaced by a robust statistic.</p>"
    )
    if s["loss_quantiles"]:
        parts.append(f"<h3>Loss quantiles ({html.escape(str(s['quantile_method']))})</h3>")
        parts.append(
            _table(["quantile", "log loss"], [[q, v] for q, v in s["loss_quantiles"].items()])
        )
    parts.append(fig_html(loss_histogram_figure(record["subjects"])))

    cat = record["catastrophic"]
    parts.append("<h2>Extreme-loss diagnostics</h2>")
    if cat is None:
        parts.append("<p>No identifiable subjects; not computed.</p>")
    else:
        parts.append(
            _table(
                ["P(truth) below", "subjects", "fraction"],
                [[t["threshold"], t["count"], t["fraction"]] for t in cat["threshold_counts"]]
                + [
                    [
                        "exactly 0 (separate category)",
                        cat["n_zero_probability"],
                        cat["fraction_zero_probability"],
                    ]
                ],
            )
        )
        parts.append(f"<h3>Worst {len(cat['worst_subjects'])} subjects</h3>")
        parts.append(_table(["subject", "log loss"], cat["worst_subjects"]))
        parts.append(
            f"<p>Share of total loss from the single worst subject: "
            f"{_esc(cat['top_contributor_fraction'])}.</p>"
        )
        if cat["trimmed_mean_log_score"] is not None:
            parts.append(
                f"<p class='warn'>Sensitivity only: the {_esc(cat['trimmed_fraction'])} "
                f"upper-tail trimmed mean is {_esc(cat['trimmed_mean_log_score'])}. It is "
                "descriptive, is not a proper scoring rule, and does not replace the "
                "primary mean above.</p>"
            )

    cal = record["calibration"]
    parts.append("<h2>Calibration of the top prediction</h2>")
    parts.append(
        f"<p>ECE (bin-count weighted) = <b>{_esc(cal['ece'])}</b>, MCE = "
        f"<b>{_esc(cal['mce'])}</b> over {cal['n_evaluated']} subjects "
        f"({cal['n_undefined']} excluded). Binning: {html.escape(cal['binning'])}, "
        f"{cal['n_bins_requested']} bins requested.</p>"
    )
    parts.append(fig_html(reliability_figure(cal)))
    parts.append(
        _table(
            ["bin", "lower", "upper", "n", "mean predicted", "observed", "CI low", "CI high"],
            [
                [
                    b["index"],
                    b["lower"],
                    b["upper"],
                    b["count"],
                    b["mean_predicted"],
                    b["observed_rate"],
                    b["ci_low"],
                    b["ci_high"],
                ]
                for b in cal["bins"]
            ],
        )
    )

    for name, ev in record["event_calibration"].items():
        parts.append(f"<h2>Event calibration: {html.escape(name)}</h2>")
        parts.append(
            f"<p>Brier score for the event = <b>{_esc(ev['brier'])}</b>, ECE = "
            f"<b>{_esc(ev['ece'])}</b>, MCE = <b>{_esc(ev['mce'])}</b> over "
            f"{ev['n_evaluated']} subjects ({ev['n_undefined']} excluded: event "
            "probability is identifiable only for complete distributions).</p>"
        )
        parts.append(fig_html(reliability_figure(ev)))
        parts.append(
            _table(
                ["bin", "n", "mean predicted", "observed", "CI low", "CI high"],
                [
                    [
                        b["index"],
                        b["count"],
                        b["mean_predicted"],
                        b["observed_rate"],
                        b["ci_low"],
                        b["ci_high"],
                    ]
                    for b in ev["bins"]
                ],
            )
        )

    rc = record["risk_coverage"]
    parts.append("<h2>Risk–coverage</h2>")
    parts.append(
        f"<p>Confidence score: <code>{html.escape(str(rc['score_source']))}</code>. "
        f"{rc['n_evaluable']} evaluable subjects; {rc['n_abstained']} abstained "
        f"(counted as uncovered); {rc['n_undefined']} excluded as undefined.</p>"
    )
    parts.append(fig_html(risk_coverage_figure(rc)))
    parts.append(
        _table(
            ["threshold", "retained", "coverage", "accuracy", "error rate", "mean log loss"],
            [
                [
                    p["threshold"],
                    p["n_retained"],
                    p["coverage"],
                    p["accuracy"],
                    p["error_rate"],
                    p["mean_log_loss"],
                ]
                for p in rc["points"]
            ],
        )
    )

    if record["undefined"]:
        parts.append("<h2>Excluded (undefined) subjects</h2>")
        parts.append(_table(["subject", "reason"], record["undefined"]))

    parts.append("<h2>Methods and assumptions</h2>")
    parts.append(
        "<ul>"
        + "".join(
            f"<li>{html.escape(t)}</li>"
            for t in [
                "Scores compare predicted distributions with independently known truth; no "
                "inference is performed by ImmunoMath.",
                "Negative log score uses natural logarithms (nats); zero probability on the "
                "truth gives +∞ with no clipping. Brier score is the multiclass sum form.",
                "A subject is 'undefined' when its truth may lie in unreported residual mass; "
                "it is excluded and counted, never treated as zero probability.",
                f"Bootstrap: {params.get('n_resamples')} subject-level resamples, "
                f"{params.get('bootstrap_method')} method, seed {params.get('seed')}.",
                f"Calibration: {params.get('binning')} binning, {params.get('n_bins')} bins; "
                "ECE and "
                "MCE are weighted by bin count; per-bin intervals are Wilson score intervals.",
                "ECE and MCE depend on the binning and are not proper scoring rules; a "
                "calibrated forecast can still be uninformative.",
                f"Risk–coverage confidence score: {params.get('confidence_score')}. Coverage is "
                "retained subjects divided by evaluable subjects.",
                "Event probabilities are identifiable only for distributions without "
                "residual mass.",
                "Intervals are unadjusted for multiple comparisons.",
            ]
        )
        + "</ul>"
    )
    parts.append("<h3>Recorded parameters</h3>")
    parts.append(_table(["parameter", "value"], [[k, str(v)] for k, v in sorted(params.items())]))

    doc = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(str(meta['label']))} — ImmunoMath report</title>"
        f"<style>{_CSS}</style></head><body>" + "\n".join(parts) + "</body></html>"
    )
    out = Path(path)
    out.write_text(doc, encoding="utf-8")
    return out
