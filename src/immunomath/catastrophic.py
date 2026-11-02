"""Catastrophic-error analysis for cohorts.

Because the negative log score penalizes small probabilities assigned to the
truth without bound, a handful of subjects can dominate the cohort mean. This
module provides diagnostics to *find and describe* those subjects — never to
trim or exclude them from the primary proper-score summary.

The key outputs are:

*   The fraction of subjects whose true-outcome probability falls below each of
    a set of configurable thresholds (default ``0.01``, ``0.001``, ``0.0001``),
    with **exactly zero** probability reported as a separate category.
*   The worst individual losses and their contribution to the total loss.
*   Quantiles of the individual-loss distribution.
*   An optional 1% upper-tail trimmed mean as a **descriptive sensitivity**
    statistic — explicitly *not* a proper scoring rule and never a replacement
    for the primary untrimmed mean.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from .cohort import Cohort
from .quantiles import QuantileMethod, loss_quantile
from .scoring import ScoreClass, SubjectScore
from .statistics import trimmed_mean

__all__ = [
    "DEFAULT_PROBABILITY_THRESHOLDS",
    "CatastrophicReport",
    "ThresholdCount",
    "catastrophic_error_report",
]

#: Default P(truth) thresholds for the low-probability fractions.
DEFAULT_PROBABILITY_THRESHOLDS: tuple[float, ...] = (0.01, 0.001, 0.0001)


@dataclass(frozen=True)
class ThresholdCount:
    """Count and fraction of identifiable subjects below a probability threshold.

    Parameters
    ----------
    threshold : float
        The ``P(truth)`` threshold (strict ``<``).
    count : int
        Number of identifiable subjects with ``0 < P(truth) < threshold``.
        (Exactly-zero subjects are reported separately, not here.)
    fraction : float
        ``count / n_identifiable``.
    """

    threshold: float
    count: int
    fraction: float


@dataclass(frozen=True)
class CatastrophicReport:
    """Diagnostics describing the tail of the individual-loss distribution.

    Attributes
    ----------
    n_identifiable : int
        Denominator for all fractions (finite + infinite subjects).
    threshold_counts : tuple[ThresholdCount, ...]
        Low-probability fractions, one per configured threshold.
    n_zero_probability : int
        Subjects with **exactly** zero probability on the truth (infinite loss),
        reported as a separate category from the thresholds.
    fraction_zero_probability : float
        ``n_zero_probability / n_identifiable``.
    worst_subjects : tuple[tuple[Hashable, float], ...]
        The ``(subject_id, log_score)`` pairs with the largest losses, descending.
    loss_quantiles : dict[float, float]
        Quantiles of the identifiable individual-loss distribution.
    total_loss : float
        Sum of identifiable individual losses (``+inf`` if any are infinite).
    top_contributor_fraction : float
        Fraction of the (finite) total loss contributed by the single worst
        finite subject; ``nan`` if the total is infinite or zero.
    primary_mean_log_score : float
        The primary untrimmed mean over identifiable subjects (for reference).
    trimmed_mean_log_score : float or None
        Optional descriptive 1% upper-tail trimmed mean; ``None`` unless
        requested. **Not** a proper scoring rule.
    trimmed_fraction : float or None
        The upper-tail fraction trimmed, when applicable.
    """

    n_identifiable: int
    threshold_counts: tuple[ThresholdCount, ...]
    n_zero_probability: int
    fraction_zero_probability: float
    worst_subjects: tuple[tuple[Hashable, float], ...]
    loss_quantiles: dict[float, float]
    total_loss: float
    top_contributor_fraction: float
    primary_mean_log_score: float
    trimmed_mean_log_score: float | None
    trimmed_fraction: float | None


def catastrophic_error_report(
    cohort: Cohort,
    *,
    thresholds: Sequence[float] = DEFAULT_PROBABILITY_THRESHOLDS,
    n_worst: int = 10,
    quantile_levels: Sequence[float] = (0.5, 0.9, 0.95, 0.99, 1.0),
    quantile_method: QuantileMethod = QuantileMethod.LINEAR,
    include_trimmed_mean: bool = False,
    trimmed_fraction: float = 0.01,
    floor: float | None = None,
) -> CatastrophicReport:
    r"""Build a catastrophic-error report for a cohort.

    Parameters
    ----------
    cohort : Cohort
        The cohort to analyze. Undefined subjects (unidentifiable truth) are
        excluded from all counts and reported only through the cohort summary;
        they are **not** counted as zero-probability.
    thresholds : sequence of float, optional
        ``P(truth)`` thresholds for the low-probability fractions. Defaults to
        ``(0.01, 0.001, 0.0001)``. Configurable.
    n_worst : int, optional
        How many worst-loss subjects to list. Defaults to ``10``.
    quantile_levels : sequence of float, optional
        Quantile levels of the individual-loss distribution to report.
    quantile_method : QuantileMethod, optional
        Quantile convention (default Hyndman-Fan type 7 / linear).
    include_trimmed_mean : bool, optional
        If ``True``, also compute a descriptive upper-tail trimmed mean. Off by
        default. The trimmed mean is **not** a proper scoring rule.
    trimmed_fraction : float, optional
        Upper-tail fraction to trim when ``include_trimmed_mean`` is set.
        Defaults to ``0.01`` (1%).
    floor : float, optional
        Optional probability floor passed through to the scorer.

    Returns
    -------
    CatastrophicReport
        The assembled diagnostics.

    Raises
    ------
    ValueError
        If any threshold is outside ``(0, 1]`` or there are no identifiable
        subjects.

    Examples
    --------
    >>> from immunomath import Distribution, Cohort
    >>> from immunomath.catastrophic import catastrophic_error_report
    >>> cohort = Cohort(
    ...     ["S1", "S2", "S3"],
    ...     ["G1", "G2", "G2"],
    ...     [
    ...         Distribution({"G1": 0.9, "G2": 0.1}),
    ...         Distribution({"G1": 0.999, "G2": 0.001}),   # near-catastrophic
    ...         Distribution({"G1": 1.0, "G2": 0.0}),        # catastrophic (zero)
    ...     ],
    ... )
    >>> report = catastrophic_error_report(cohort)
    >>> report.n_zero_probability
    1
    >>> report.n_identifiable
    3
    """
    for t in thresholds:
        if not 0.0 < t <= 1.0:
            raise ValueError(f"Each threshold must lie in (0, 1], got {t!r}.")

    from .scoring import score_cohort_accounting

    accounting = score_cohort_accounting(
        list(cohort.truths), list(cohort.distributions), floor=floor
    )
    ids = cohort.subject_ids
    identifiable = [s for s in accounting.subjects if s.score_class is not ScoreClass.UNDEFINED]
    n_identifiable = len(identifiable)
    if n_identifiable == 0:
        raise ValueError("No identifiable subjects to analyze.")

    # Exactly-zero-probability subjects are the INFINITE class.
    n_zero = sum(1 for s in identifiable if s.score_class is ScoreClass.INFINITE)

    def loss_of(s: SubjectScore) -> float:
        return s.score if s.score is not None else math.inf

    threshold_counts: list[ThresholdCount] = []
    for t in thresholds:
        # Count subjects with 0 < P(truth) < t. P = exp(-log_score) for finite,
        # so P < t  <=>  loss > -ln t.
        cutoff_loss = -math.log(t)
        count = sum(
            1
            for s in identifiable
            if s.score_class is ScoreClass.FINITE and s.score is not None and s.score > cutoff_loss
        )
        # Zero-probability subjects also have P(truth) < t; include them in the
        # count so the fraction is a genuine "below threshold" measure, while
        # still breaking them out separately as n_zero_probability.
        count += n_zero
        threshold_counts.append(
            ThresholdCount(threshold=t, count=count, fraction=count / n_identifiable)
        )

    losses = [loss_of(s) for s in identifiable]
    worst = sorted(
        ((ids[s.index], loss_of(s)) for s in identifiable),
        key=lambda pair: pair[1],
        reverse=True,
    )[:n_worst]

    loss_quantiles = {q: loss_quantile(losses, q, method=quantile_method) for q in quantile_levels}

    total_loss = math.inf if n_zero > 0 else math.fsum(losses)
    if math.isinf(total_loss) or total_loss == 0.0:
        top_contributor_fraction = math.nan
    else:
        top_contributor_fraction = max(losses) / total_loss

    primary_mean = math.inf if n_zero > 0 else math.fsum(losses) / n_identifiable

    trimmed_value: float | None = None
    used_fraction: float | None = None
    if include_trimmed_mean:
        trimmed_value = trimmed_mean(losses, upper_fraction=trimmed_fraction)
        used_fraction = trimmed_fraction

    return CatastrophicReport(
        n_identifiable=n_identifiable,
        threshold_counts=tuple(threshold_counts),
        n_zero_probability=n_zero,
        fraction_zero_probability=n_zero / n_identifiable,
        worst_subjects=tuple(worst),
        loss_quantiles=loss_quantiles,
        total_loss=total_loss,
        top_contributor_fraction=top_contributor_fraction,
        primary_mean_log_score=primary_mean,
        trimmed_mean_log_score=trimmed_value,
        trimmed_fraction=used_fraction,
    )
