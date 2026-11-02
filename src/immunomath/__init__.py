"""ImmunoMath: rigorous, reproducible evaluation of probabilistic genetic inference.

ImmunoMath (framework name: **Metrognome**) is an *evaluation* library, not an
inference library. It compares predicted probability distributions against
independently known truth using proper scoring rules, calibration diagnostics,
and cohort statistics. It does not perform imputation, simulate typing, or
estimate haplotype frequencies; those responsibilities belong to separate
packages.

*Keeping probabilistic genetics honest.*

The statistical kernel is deliberately free of HLA nomenclature. Its
foundational abstraction is the :class:`Distribution`: a validated, finite,
discrete probability distribution over hashable outcomes. All scoring and
diagnostic functions operate on that abstraction.

This release delivers **Milestone 1 — the mathematical kernel**.
"""

from __future__ import annotations

from .diagnostics import (
    RankOfTruth,
    effective_number_of_outcomes,
    entropy,
    num_candidates,
    probability_of_truth,
    rank_of_truth,
    top_k_accuracy,
)
from .distributions import RESIDUAL, Distribution, Projection, Residual
from .scoring import (
    CohortScore,
    brier_score,
    mean_brier_score,
    mean_negative_log_score,
    negative_log_score,
)

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022  (grouped by role, not alphabetical, for readability)
    "__version__",
    # Core abstraction
    "Distribution",
    "Residual",
    "RESIDUAL",
    "Projection",
    # Proper scoring rules
    "negative_log_score",
    "mean_negative_log_score",
    "brier_score",
    "mean_brier_score",
    "CohortScore",
    # Descriptive diagnostics
    "probability_of_truth",
    "top_k_accuracy",
    "rank_of_truth",
    "RankOfTruth",
    "entropy",
    "effective_number_of_outcomes",
    "num_candidates",
]
