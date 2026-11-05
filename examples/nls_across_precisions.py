r"""Evaluate ``negative_log_score`` across a range of distribution precisions.

Run with:

    python examples/nls_across_precisions.py

This example builds a simulated sample of predicted probability distributions
whose *skewness* (how sharply the mass concentrates on a single outcome) is
swept across a range of precision levels, then reports how the negative
logarithmic score responds.

What it illustrates
-------------------
A proper scoring rule rewards *honest, well-calibrated* confidence, not
confidence for its own sake. Here every prediction is drawn so that the truth
is sampled *from the predicted distribution itself* — i.e. the forecaster is
perfectly calibrated. Under that honest regime, a sharper (more skewed, lower
Dirichlet concentration) distribution yields a lower expected negative log
score, because the model's confidence is justified by the data-generating
process.

The "precision" knob is a symmetric Dirichlet concentration :math:`\alpha`.
*Small* :math:`\alpha` produces sparse, peaked, low-entropy distributions
(high skewness); *large* :math:`\alpha` produces flat, near-uniform,
high-entropy distributions (low skewness).

Reproducibility
---------------
All randomness flows from a single seeded ``numpy`` generator, so repeated runs
produce identical output. ImmunoMath itself performs no simulation; the sampling
here is confined to this example script. The distributions are passed to
ImmunoMath exactly as generated — never silently renormalized — so each sampled
vector is rounded and repaired to sum to one before construction.

Requires ``numpy`` (already a dependency of ImmunoMath).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

import immunomath as im

# --- Simulation configuration (all reproducible) -------------------------- #
SEED = 1
NUM_SAMPLES = 100  # probability sets per precision level
NUM_OUTCOMES = 8  # size of the candidate set for each distribution
PRECISIONS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # Dirichlet concentration alpha


@dataclass(frozen=True)
class PrecisionResult:
    """Aggregated negative-log-score statistics for one precision level."""

    precision: float
    mean_nls: float
    median_nls: float
    mean_entropy: float
    mean_top1: float
    num_finite: int
    num_total: int


def _make_distribution(probs: np.ndarray) -> im.Distribution:
    """Build a validated :class:`im.Distribution` from a probability vector.

    The sampled vector is rounded for display-stable mass and its largest
    entry absorbs the rounding residual, so the result sums to one exactly
    without relying on ImmunoMath to renormalize (which it never does silently).
    """
    rounded = np.round(probs, 12)
    rounded[int(np.argmax(rounded))] += 1.0 - rounded.sum()
    return im.Distribution({f"G{i}": float(p) for i, p in enumerate(rounded)})


def simulate_precision_level(
    precision: float,
    rng: np.random.Generator,
    *,
    num_samples: int = NUM_SAMPLES,
    num_outcomes: int = NUM_OUTCOMES,
) -> PrecisionResult:
    """Simulate a sample of predictions at one precision and score them.

    For each sample we draw a probability vector from a symmetric Dirichlet
    with concentration ``precision``, sample a *truth* from that same vector
    (honest, calibrated forecaster), and record the negative log score plus
    descriptive context.
    """
    nls_values: list[float] = []
    entropies: list[float] = []
    top1_hits: list[float] = []
    num_finite = 0

    for _ in range(num_samples):
        probs = rng.dirichlet(np.full(num_outcomes, precision))
        distribution = _make_distribution(probs)

        # Honest forecaster: the truth is drawn from the predicted distribution.
        truth_index = int(rng.choice(num_outcomes, p=probs))
        truth = f"G{truth_index}"

        score = im.negative_log_score(truth, distribution)
        if np.isfinite(score):
            nls_values.append(score)
            num_finite += 1

        entropies.append(im.entropy(distribution))
        top1_hits.append(im.top_k_accuracy(truth, distribution, k=1))

    return PrecisionResult(
        precision=precision,
        mean_nls=statistics.fmean(nls_values) if nls_values else float("nan"),
        median_nls=statistics.median(nls_values) if nls_values else float("nan"),
        mean_entropy=statistics.fmean(entropies),
        mean_top1=statistics.fmean(top1_hits),
        num_finite=num_finite,
        num_total=num_samples,
    )


def main() -> None:
    """Sweep precision levels and print how the log score responds."""
    rng = np.random.default_rng(SEED)

    print(
        f"Negative log score across {len(PRECISIONS)} precision levels "
        f"(seed={SEED}, {NUM_SAMPLES} samples x {NUM_OUTCOMES} outcomes each)\n"
    )
    header = (
        f"{'alpha':>7} {'mean NLS':>10} {'median NLS':>11} "
        f"{'mean H':>8} {'mean top-1':>11} {'finite':>8}"
    )
    print(header)
    print("-" * len(header))

    for precision in PRECISIONS:
        result = simulate_precision_level(precision, rng)
        print(
            f"{result.precision:>7.2f} "
            f"{result.mean_nls:>10.4f} "
            f"{result.median_nls:>11.4f} "
            f"{result.mean_entropy:>8.4f} "
            f"{result.mean_top1:>11.4f} "
            f"{result.num_finite:>5}/{result.num_total}"
        )

    print(
        "\nalpha is the symmetric Dirichlet concentration (inverse precision):\n"
        "small alpha -> sparse, peaked, low-entropy predictions (high skewness);\n"
        "large alpha -> flat, near-uniform, high-entropy predictions (low skewness).\n"
        "Because the truth is sampled from each prediction (an honest, calibrated\n"
        "forecaster), the sharper small-alpha distributions earn a lower mean\n"
        "negative log score."
    )


if __name__ == "__main__":
    main()
