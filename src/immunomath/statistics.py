"""Resampling statistics for cohort assessment: bootstrap CIs and trimmed mean.

This module provides the resampling machinery of Milestone 2:

*   :func:`trimmed_mean` — an upper-tail trimmed mean offered strictly as a
    *descriptive sensitivity* statistic. It is **not** a proper scoring rule and
    must never replace the primary (untrimmed) mean.
*   :func:`bootstrap_ci` — seeded, reproducible subject-level bootstrap
    confidence intervals (percentile by default, BCa optional).
*   :func:`paired_bootstrap_ci` — paired subject-level resampling for comparing
    two measurements on the same individuals.

Quantiles of a loss distribution live in :mod:`immunomath.quantiles`; this
module imports them rather than re-implementing them, so the Hyndman-Fan
convention and the explicit infinite-value handling are defined in exactly one
place.

Infinite values are first-class here. A cohort mean of log losses can
legitimately be ``+inf`` (a subject with zero probability on the truth). These
routines never silently drop such values or emit an unexplained ``NaN``; where a
statistic is genuinely infinite, that is reported explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_N_RESAMPLES",
    "BootstrapResult",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "trimmed_mean",
]

#: Default number of bootstrap resamples.
DEFAULT_N_RESAMPLES: int = 2000
#: Default two-sided confidence level.
DEFAULT_CONFIDENCE: float = 0.95


def trimmed_mean(values: Sequence[float], *, upper_fraction: float = 0.01) -> float:
    r"""Return an **upper-tail** trimmed mean (a descriptive sensitivity statistic).

    This removes the largest ``upper_fraction`` of observations (by value) and
    averages the rest. Its purpose is *diagnostic*: it quantifies how sensitive
    the ordinary mean is to the worst few subjects.

    .. warning::

        The trimmed mean is **not** a proper scoring rule. Trimming discards
        exactly the catastrophic cases that a proper rule is designed to
        penalize. It must never replace the primary, untrimmed mean log loss as
        the headline summary; report it only as a labelled sensitivity analysis.

    Only the **upper** tail is trimmed (large losses), because the motivating
    concern is a few catastrophic observations dominating the mean. The number
    trimmed is ``floor(upper_fraction * n)``; with ``upper_fraction = 0.01`` and
    ``n < 100`` this trims zero observations, so the result equals the ordinary
    mean — trimming never silently activates on tiny cohorts.

    Parameters
    ----------
    values : sequence of float
        The sample. May contain ``+inf``; trimming the upper tail removes the
        largest values first, so sufficiently many infinities are trimmed away.
    upper_fraction : float, optional
        Fraction of the upper tail to trim, in ``[0, 1)``. Defaults to ``0.01``
        (1%).

    Returns
    -------
    float
        The trimmed mean. May be ``+inf`` if infinities remain after trimming.

    Raises
    ------
    ValueError
        If ``values`` is empty, ``upper_fraction`` is outside ``[0, 1)``, the
        sample contains ``NaN``, or every observation is trimmed.

    Examples
    --------
    >>> from immunomath.statistics import trimmed_mean
    >>> trimmed_mean([1.0, 2.0, 3.0, 100.0], upper_fraction=0.25)  # drop the 100
    2.0
    >>> trimmed_mean([1.0, 2.0, 3.0])  # n<100 at 1% -> no trimming
    2.0
    """
    if not 0.0 <= upper_fraction < 1.0:
        raise ValueError(f"upper_fraction must lie in [0, 1), got {upper_fraction!r}.")
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute a trimmed mean of an empty sample.")
    if np.isnan(arr).any():
        raise ValueError("Input contains NaN; resolve missing values explicitly.")

    n = arr.size
    k = math.floor(upper_fraction * n)
    ordered = np.sort(arr)
    kept = ordered[: n - k] if k > 0 else ordered
    if kept.size == 0:
        raise ValueError("All observations were trimmed; nothing to average.")
    return math.fsum(float(x) for x in kept) / kept.size


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap confidence interval and its diagnostics.

    Parameters
    ----------
    point_estimate : float
        The statistic computed on the original sample (not a bootstrap mean).
    low, high : float
        The lower and upper confidence bounds.
    confidence : float
        The two-sided confidence level (e.g. ``0.95``).
    method : str
        ``"percentile"`` or ``"bca"``.
    n_resamples : int
        The number of bootstrap resamples drawn.
    n_infinite_resamples : int
        The number of resamples whose statistic was infinite. When this is
        nonzero the affected bound is reported as infinite rather than a
        misleading finite number.
    seed : int | None
        The seed used, for reproducibility.
    """

    point_estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n_resamples: int
    n_infinite_resamples: int
    seed: int | None


def _mean_with_inf(sample: np.ndarray) -> float:
    """Arithmetic mean that returns +/-inf if any element is +/-inf."""
    if np.isposinf(sample).any():
        return math.inf
    if np.isneginf(sample).any():
        return -math.inf
    return float(math.fsum(sample.tolist()) / sample.size)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Literal["mean"] = "mean",
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int | None = None,
) -> BootstrapResult:
    r"""Seeded subject-level bootstrap confidence interval for the mean.

    Resampling is at the **subject** level (ordinary nonparametric bootstrap):
    each resample draws ``n`` subjects with replacement and recomputes the
    statistic. Results are **reproducible** given a fixed ``seed`` (a NumPy
    :class:`~numpy.random.Generator` seeded deterministically).

    Infinite values are respected: if a resample includes a subject with an
    infinite score, that resample's mean is infinite, counted in
    ``n_infinite_resamples``. If infinite resamples reach into a confidence
    bound, that bound is reported as infinite rather than a misleadingly finite
    quantile.

    Parameters
    ----------
    values : sequence of float
        Per-subject statistic values (e.g. individual log losses).
    statistic : {"mean"}, optional
        The statistic to bootstrap. Currently only ``"mean"`` (the proper-score
        cohort summary) is supported.
    confidence : float, optional
        Two-sided confidence level in ``(0, 1)``. Defaults to ``0.95``.
    n_resamples : int, optional
        Number of bootstrap resamples. Defaults to ``2000``.
    method : {"percentile", "bca"}, optional
        Interval method. ``"percentile"`` (default) takes empirical quantiles of
        the resample statistics. ``"bca"`` applies bias-correction and
        acceleration (Efron) and is more accurate for skewed statistics.
    seed : int or None, optional
        Seed for reproducible resampling.

    Returns
    -------
    BootstrapResult
        The point estimate, bounds, and diagnostics.

    Raises
    ------
    ValueError
        If inputs are empty/invalid, or the confidence level is out of range.

    Notes
    -----
    The BCa interval adjusts the percentile endpoints using a bias-correction
    :math:`\hat{z}_0` (from the fraction of resample statistics below the point
    estimate) and an acceleration :math:`\hat{a}` (from the jackknife skewness);
    see Efron & Tibshirani (1993). When the statistic is infinite for some
    resamples, BCa falls back to reporting the affected bound as infinite.

    Examples
    --------
    >>> from immunomath.statistics import bootstrap_ci
    >>> res = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=0, n_resamples=500)
    >>> res.low < res.point_estimate < res.high
    True
    >>> res.point_estimate
    0.3
    """
    if statistic != "mean":
        raise ValueError(f"Unsupported statistic: {statistic!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}.")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}.")
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    if np.isnan(arr).any():
        raise ValueError("Input contains NaN; resolve missing values explicitly.")

    point = _mean_with_inf(arr)
    n = arr.size
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(n_resamples, n))
    resamples = arr[idx]
    has_pos_inf = np.isposinf(resamples).any(axis=1)
    has_neg_inf = np.isneginf(resamples).any(axis=1)
    stats = np.empty(n_resamples, dtype=float)
    finite_mask = ~(has_pos_inf | has_neg_inf)
    stats[finite_mask] = resamples[finite_mask].mean(axis=1)
    stats[has_pos_inf] = math.inf
    stats[has_neg_inf & ~has_pos_inf] = -math.inf
    n_infinite = int((~finite_mask).sum())

    alpha = 1.0 - confidence
    if method == "percentile":
        lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    elif method == "bca":
        lo_q, hi_q = _bca_quantiles(arr, stats, point, alpha)
    else:
        raise ValueError(f"Unsupported method: {method!r}.")

    low = _inf_aware_quantile(stats, lo_q)
    high = _inf_aware_quantile(stats, hi_q)

    return BootstrapResult(
        point_estimate=point,
        low=low,
        high=high,
        confidence=confidence,
        method=method,
        n_resamples=n_resamples,
        n_infinite_resamples=n_infinite,
        seed=seed,
    )


def _inf_aware_quantile(stats: np.ndarray, q: float) -> float:
    """Percentile of resample statistics that respects infinite entries."""
    ordered = np.sort(stats)
    n = ordered.size
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    a, b = ordered[lo], ordered[hi]
    if a == b:
        return float(a)
    if math.isinf(a) or math.isinf(b):
        return float(b if q >= 0.5 else a)
    frac = pos - lo
    return float(a + frac * (b - a))


def _bca_quantiles(
    data: np.ndarray, stats: np.ndarray, point: float, alpha: float
) -> tuple[float, float]:
    """Compute BCa-adjusted lower/upper quantile levels.

    Falls back to percentile levels when infinities make the bias-correction
    undefined.
    """
    from scipy.stats import norm

    finite_stats = stats[np.isfinite(stats)]
    if finite_stats.size == 0 or not math.isfinite(point):
        return alpha / 2.0, 1.0 - alpha / 2.0

    prop_less = float((finite_stats < point).mean())
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    finite_data = data[np.isfinite(data)]
    nj = finite_data.size
    if nj < 2:
        return alpha / 2.0, 1.0 - alpha / 2.0
    total = float(math.fsum(finite_data.tolist()))
    jack = (total - finite_data) / (nj - 1)
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = 6.0 * (float(np.sum(diff**2)) ** 1.5)
    a_hat = 0.0 if denom == 0.0 else float(np.sum(diff**3)) / denom

    z_alpha_lo = norm.ppf(alpha / 2.0)
    z_alpha_hi = norm.ppf(1.0 - alpha / 2.0)

    def adjust(z: float) -> float:
        num = z0 + z
        return float(norm.cdf(z0 + num / (1.0 - a_hat * num)))

    return adjust(z_alpha_lo), adjust(z_alpha_hi)


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    method: Literal["percentile", "bca"] = "percentile",
    seed: int | None = None,
) -> BootstrapResult:
    r"""Paired subject-level bootstrap CI for the mean difference ``a - b``.

    Both inputs are aligned per subject (same individuals, same order). Each
    bootstrap resample draws the **same** subject indices for both arms,
    preserving the pairing, and the statistic is the mean of the paired
    differences :math:`d_i = a_i - b_i`.

    This is the foundation for Milestone 3 comparisons; it is included now
    because the scientific decision requires paired subject-level resampling.

    Parameters
    ----------
    values_a, values_b : sequence of float
        Per-subject statistics for arm A and arm B, aligned and equal length.
    confidence, n_resamples, method, seed
        As in :func:`bootstrap_ci`.

    Returns
    -------
    BootstrapResult
        Point estimate and CI for the mean paired difference.

    Raises
    ------
    ValueError
        If lengths differ, inputs are empty, or a paired difference is an
        indeterminate ``inf - inf``.

    Examples
    --------
    >>> from immunomath.statistics import paired_bootstrap_ci
    >>> a = [0.5, 0.4, 0.6, 0.3]
    >>> b = [0.4, 0.5, 0.4, 0.2]
    >>> res = paired_bootstrap_ci(a, b, seed=1, n_resamples=500)
    >>> round(res.point_estimate, 4)
    0.075
    """
    arr_a = np.asarray(list(values_a), dtype=float)
    arr_b = np.asarray(list(values_b), dtype=float)
    if arr_a.size != arr_b.size:
        raise ValueError(
            f"Paired inputs must have equal length; got {arr_a.size} and {arr_b.size}."
        )
    if arr_a.size == 0:
        raise ValueError("Cannot bootstrap an empty paired sample.")
    if np.isnan(arr_a).any() or np.isnan(arr_b).any():
        raise ValueError("Input contains NaN; resolve missing values explicitly.")

    both_pos_inf = np.isposinf(arr_a) & np.isposinf(arr_b)
    both_neg_inf = np.isneginf(arr_a) & np.isneginf(arr_b)
    if both_pos_inf.any() or both_neg_inf.any():
        raise ValueError(
            "A paired difference is an indeterminate inf - inf form. Both arms "
            "assigned the truth zero probability for the same subject; the "
            "difference in log loss is undefined. Resolve these subjects explicitly."
        )
    diffs = arr_a - arr_b
    return bootstrap_ci(
        diffs,
        statistic="mean",
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
        seed=seed,
    )
