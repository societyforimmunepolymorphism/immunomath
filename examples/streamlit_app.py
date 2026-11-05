r"""Interactive case generator and evaluator for ImmunoMath.

Run with:

    uv run --with streamlit streamlit run examples/streamlit_app.py

or, inside an activated virtual environment that has streamlit installed:

    streamlit run examples/streamlit_app.py

What this app does
------------------
It generates a reproducible *set of cases* just like
``examples/nls_across_precisions.py``: each case is a predicted probability
distribution drawn from a symmetric Dirichlet (concentration ``alpha``), with a
*truth* sampled from that same distribution (an honest, calibrated forecaster).

It then shows, side by side:

* **Inputs**  — each predicted :class:`immunomath.Distribution` as a bar chart,
  with the sampled truth highlighted, plus the Dirichlet concentration knob.
* **Outputs** — the ImmunoMath evaluation of those inputs: per-case proper
  scores and diagnostics, a whole-cohort analysis via
  :func:`immunomath.analyze_cohort`, and an **observation-weighted** summary
  that upweights poorly resolved cases via the TRS adapter
  (:func:`immunomath.trs.trs_weights`).

ImmunoMath itself performs no simulation. The sampling here is confined to this
script; the distributions are passed to ImmunoMath exactly as generated and are
never silently renormalized.

This example is intentionally dependency-light on top of ImmunoMath: it needs
``streamlit`` (not an ImmunoMath dependency) and ``numpy`` / ``plotly`` (which
ImmunoMath already brings in, plotly via the ``reporting`` extra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import immunomath as im
from immunomath.trs import trs_weights

# --------------------------------------------------------------------------- #
# Case generation (the only randomness in this app; fully seeded).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Case:
    """One generated case: the inputs plus the sampled truth."""

    subject_id: str
    truth: str
    distribution: im.Distribution
    probs: np.ndarray  # raw sampled vector, aligned with ``outcomes``
    outcomes: tuple[str, ...]


def _make_distribution(probs: np.ndarray, outcomes: tuple[str, ...]) -> im.Distribution:
    """Build a validated :class:`im.Distribution` from a probability vector.

    The sampled vector is rounded and its largest entry absorbs the rounding
    residual so it sums to one exactly, without relying on ImmunoMath to
    renormalize (which it never does silently).
    """
    rounded = np.round(probs, 12)
    rounded[int(np.argmax(rounded))] += 1.0 - rounded.sum()
    return im.Distribution({g: float(p) for g, p in zip(outcomes, rounded, strict=False)})


def generate_cases(
    *,
    seed: int,
    num_cases: int,
    num_outcomes: int,
    precision: float,
) -> list[Case]:
    """Generate a reproducible set of honest, calibrated cases.

    Mirrors ``examples/nls_across_precisions.py``: each predicted distribution
    is a symmetric-Dirichlet draw with concentration ``precision``; the truth is
    drawn from that same distribution.
    """
    rng = np.random.default_rng(seed)
    outcomes = tuple(f"G{i}" for i in range(num_outcomes))
    cases: list[Case] = []
    for i in range(num_cases):
        probs = rng.dirichlet(np.full(num_outcomes, precision))
        distribution = _make_distribution(probs, outcomes)
        truth_index = int(rng.choice(num_outcomes, p=probs))
        cases.append(
            Case(
                subject_id=f"S{i + 1}",
                truth=outcomes[truth_index],
                distribution=distribution,
                probs=probs,
                outcomes=outcomes,
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# Visualization helpers.
# --------------------------------------------------------------------------- #

_TRUTH_COLOR = "#2ca02c"  # green: the outcome that actually occurred
_OTHER_COLOR = "#8c9bb5"  # muted: the other candidates


def input_bar_chart(case: Case) -> go.Figure:
    """Return a bar chart of one prediction with the truth highlighted."""
    probs = [case.distribution.probability(g) for g in case.outcomes]
    colors = [_TRUTH_COLOR if g == case.truth else _OTHER_COLOR for g in case.outcomes]
    fig = go.Figure(
        go.Bar(
            x=list(case.outcomes),
            y=probs,
            marker_color=colors,
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{case.subject_id} — truth = {case.truth} (green)",
        yaxis_range=[0, 1],
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
        height=160,
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Output (evaluation) helpers.
# --------------------------------------------------------------------------- #


def trs_score(distribution: im.Distribution) -> float:
    r"""Return the TRS score: the sum of squared predicted probabilities.

    .. math::

        \mathrm{TRS}(P) = \sum_k p_k^2 .

    This is a per-subject measure of *sharpness* (the collision probability of
    the distribution), independent of the truth. It lies in ``(0, 1]``: ``1.0``
    for a point mass on a single outcome, and ``1 / n`` for a uniform
    distribution over ``n`` outcomes. It is computed here from the validated
    :class:`immunomath.Distribution`; ImmunoMath itself does not define it.
    """
    return sum(p * p for p in distribution.probabilities)


def trs_covariates(cases: list[Case]) -> list[float]:
    r"""Return a per-case normalized TRS covariate in ``(0, 1]``.

    The sharpness Σp² is a natural, **truth-independent** resolution proxy: a
    peaked (well-resolved) prediction scores near 1, a flat (poorly resolved)
    one near ``1/n``. Using it as the TRS keeps the weight a function of the
    prediction's *shape* only — never of the observed truth — which is exactly
    what conditional propriety requires.
    """
    return [trs_score(case.distribution) for case in cases]


def per_case_rows(cases: list[Case], weights: list[float] | None = None) -> list[dict[str, object]]:
    """Score each case with ImmunoMath and return display-ready rows."""
    rows: list[dict[str, object]] = []
    for i, case in enumerate(cases):
        nls = im.negative_log_score(case.truth, case.distribution)
        brier = im.brier_score(case.truth, case.distribution)
        top1 = im.top_k_accuracy(case.truth, case.distribution, k=1)
        rank = im.rank_of_truth(case.truth, case.distribution)
        row: dict[str, object] = {
            "subject": case.subject_id,
            "truth": case.truth,
            "P(truth)": round(case.distribution.probability(case.truth), 4),
            "neg log score": None if math.isinf(nls) else round(nls, 4),
            "brier": round(brier, 4),
            "TRS (Σp²)": round(trs_score(case.distribution), 4),
            "top-1 hit": round(top1, 3),
            "rank(min,max)": f"{rank.min_rank},{rank.max_rank}",
            "entropy": round(im.entropy(case.distribution), 4),
        }
        if weights is not None:
            row["weight"] = round(weights[i], 3)
        rows.append(row)
    return rows


def build_cohort(cases: list[Case]) -> im.Cohort:
    """Assemble an ImmunoMath :class:`Cohort` from the generated cases."""
    return im.Cohort(
        subject_ids=[c.subject_id for c in cases],
        truths=[c.truth for c in cases],
        distributions=[c.distribution for c in cases],
    )


# --------------------------------------------------------------------------- #
# Streamlit UI.
# --------------------------------------------------------------------------- #


def main() -> None:
    """Render the Streamlit case-explorer page."""
    st.set_page_config(page_title="ImmunoMath case explorer", layout="wide")

    with st.sidebar:
        st.header("Case generation")
        seed = st.number_input("Seed", min_value=0, value=1, step=1)
        num_cases = st.slider("Number of cases", 1, 24, 6)
        num_outcomes = st.slider("Candidates per case", 2, 12, 6)
        precision = st.select_slider(
            "Dirichlet concentration (alpha)",
            options=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            value=0.3,
        )
        st.markdown(
            "*Small alpha* → sparse, peaked, confident predictions.  \n"
            "*Large alpha* → flat, near-uniform, hedged predictions."
        )

        st.header("TRS weighting")
        trs_on = st.toggle("Enable observation weighting", value=True)
        lam = st.slider("λ (upweight strength)", 0.0, 10.0, 4.0, 0.5)
        gamma = st.slider("γ (curvature)", 0.1, 4.0, 2.0, 0.1)
        st.markdown(
            "Weight `w = 1 + λ(1 − r)^γ` with `r` = the case's sharpness Σp² "
            "(a truth-independent resolution proxy, higher = better resolved). "
            "`λ = 0` recovers ordinary scoring."
        )

    cases = generate_cases(
        seed=int(seed),
        num_cases=int(num_cases),
        num_outcomes=int(num_outcomes),
        precision=float(precision),
    )

    # --- Outputs first, full width -------------------------------------- #
    cohort = build_cohort(cases)
    result = im.analyze_cohort(cohort, label="streamlit explorer", seed=int(seed), n_resamples=500)
    record = result.to_dict()
    denom = record["denominators"]
    summ = record["summary"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean neg log score", f"{summ['mean_log_score']:.4f}")
    m2.metric("Mean Brier", f"{summ['mean_brier']:.4f}")
    m3.metric("Scored / total", f"{denom['n_scored']} / {denom['n_total']}")

    ci = summ.get("mean_log_score_ci")
    if ci and ci.get("low") is not None and ci.get("high") is not None:
        st.caption(
            f"Mean neg-log-score {ci['confidence']:.0%} CI: "
            f"[{ci['low']:.4f}, {ci['high']:.4f}] "
            f"({ci['method']} bootstrap, {ci['n_resamples']} resamples)."
        )

    # --- TRS-weighted scoring ------------------------------------------- #
    weights = None
    if trs_on:
        covariates = trs_covariates(cases)
        weights = list(trs_weights(covariates, lam=float(lam), gamma=float(gamma)))
        truths = [c.truth for c in cases]
        dists = [c.distribution for c in cases]
        w_log = im.weighted_mean_loss(truths, dists, weights, metric="log")
        w_brier = im.weighted_mean_loss(truths, dists, weights, metric="brier")

        w1, w2, w3 = st.columns(3)
        w1.metric(
            "Weighted neg log score",
            "∞" if math.isinf(w_log.weighted_mean) else f"{w_log.weighted_mean:.4f}",
            delta=(
                None
                if math.isinf(w_log.weighted_mean) or math.isinf(w_log.unweighted_mean)
                else f"{w_log.weighted_mean - w_log.unweighted_mean:+.4f} vs unweighted"
            ),
            delta_color="inverse",
        )
        w2.metric("Weighted Brier", f"{w_brier.weighted_mean:.4f}")
        w3.metric("Effective N (Kish)", f"{w_log.effective_sample_size:.1f} / {w_log.n_scored}")
        st.caption(
            f"TRS weighting w = 1 + {lam:g}·(1 − r)^{gamma:g}, r = sharpness Σp². "
            "Weighted means upweight poorly resolved cases; the unweighted means "
            "above are always reported alongside. Weights depend only on the "
            "prediction shape, never on the truth."
        )

    st.caption(
        "Proper scoring rules (lower is better). A blank neg-log-score means an "
        "infinite penalty. **Click a row to show that subject's distribution "
        "below.**"
    )
    selection = st.dataframe(
        per_case_rows(cases, weights),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="case_matrix",
    )

    # --- Inputs below, driven by the matrix selection ------------------- #
    selected_rows = selection.selection.rows if selection else []
    chosen = [cases[i] for i in selected_rows] if selected_rows else cases

    for case in chosen:
        st.plotly_chart(
            input_bar_chart(case),
            width="stretch",
            key=f"chart_{case.subject_id}",
        )

    with st.expander("Full analysis record (JSON)"):
        st.json(record)


if __name__ == "__main__":
    main()
