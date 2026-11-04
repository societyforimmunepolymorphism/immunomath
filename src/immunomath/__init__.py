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

This release delivers **Milestone 1 — the mathematical kernel** and
**Milestone 2 — cohort assessment** (cohort summaries with explicit
denominators, seeded bootstrap confidence intervals, loss-distribution
quantiles, and catastrophic-error diagnostics).
"""

from __future__ import annotations

from ._version import __version__
from .analysis import AnalysisResult, analyze_cohort
from .calibration import (
    DEFAULT_N_BINS,
    CalibrationBin,
    CalibrationResult,
    event_calibration,
    top_label_calibration,
)
from .catastrophic import (
    DEFAULT_PROBABILITY_THRESHOLDS,
    CatastrophicReport,
    ThresholdCount,
    catastrophic_error_report,
)
from .cohort import Cohort, CohortSummary
from .comparison import PairedComparison, compare_cohorts
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
from .quantiles import QuantileMethod, loss_quantile, loss_quantiles
from .risk_coverage import RiskCoveragePoint, RiskCoverageResult, risk_coverage
from .scoring import (
    CohortAccounting,
    CohortScore,
    ScoreClass,
    SubjectScore,
    brier_score,
    mean_brier_score,
    mean_negative_log_score,
    negative_log_score,
    score_cohort_accounting,
)
from .statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_RESAMPLES,
    BootstrapResult,
    bootstrap_ci,
    paired_bootstrap_ci,
    trimmed_mean,
)

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
    # Cohort accounting (finite / infinite / undefined)
    "score_cohort_accounting",
    "CohortAccounting",
    "SubjectScore",
    "ScoreClass",
    # Cohort container and summary
    "Cohort",
    "CohortSummary",
    # Calibration, risk-coverage and analysis record
    "top_label_calibration",
    "event_calibration",
    "CalibrationResult",
    "CalibrationBin",
    "DEFAULT_N_BINS",
    "risk_coverage",
    "RiskCoverageResult",
    "RiskCoveragePoint",
    "analyze_cohort",
    "AnalysisResult",
    # Comparative assessment (paired)
    "compare_cohorts",
    "PairedComparison",
    # Resampling & distributional statistics
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "BootstrapResult",
    "trimmed_mean",
    "loss_quantile",
    "loss_quantiles",
    "QuantileMethod",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_N_RESAMPLES",
    # Catastrophic-error diagnostics
    "catastrophic_error_report",
    "CatastrophicReport",
    "ThresholdCount",
    "DEFAULT_PROBABILITY_THRESHOLDS",
    # Descriptive diagnostics
    "probability_of_truth",
    "top_k_accuracy",
    "rank_of_truth",
    "RankOfTruth",
    "entropy",
    "effective_number_of_outcomes",
    "num_candidates",
]
