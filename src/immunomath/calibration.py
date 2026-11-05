r"""Probabilistic calibration diagnostics.

A forecast is *calibrated* if, among all cases where it reports probability
``p`` for something, that something happens a fraction ``p`` of the time.
Calibration is necessary for honest probabilistic inference but is not a
substitute for a proper scoring rule: a forecaster can be perfectly calibrated
and still uninformative (always reporting the base rate).

Two diagnostics are provided:

*   :func:`top_label_calibration` compares the probability assigned to the
    **highest-probability outcome** with how often that outcome was correct.
*   :func:`event_calibration` compares the aggregate probability of a
    **user-defined event** (for example "allele X is present") with how often
    the event occurred.

Both share one explicit binning and summary machinery. The binning strategy
(``fixed_width`` or ``equal_frequency``), the number of bins (default 10), and
the interval method are always recorded in the result so a calibration table is
never ambiguous about how it was produced.

Summary statistics
------------------
With ``N`` evaluated cases, bin counts ``n_b``, mean predicted probability
``conf_b`` and observed rate ``acc_b`` in bin ``b``:

.. math::

    \mathrm{ECE} = \sum_b \frac{n_b}{N}\,\lvert \mathrm{acc}_b - \mathrm{conf}_b \rvert,
    \qquad
    \mathrm{MCE} = \max_b \lvert \mathrm{acc}_b - \mathrm{conf}_b \rvert .

ECE is weighted by bin count; empty bins contribute nothing. Both depend on the
binning and are **not** proper scoring rules.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np

from .cohort import Cohort
from .diagnostics import top_k_accuracy

__all__ = [
    "DEFAULT_N_BINS",
    "CalibrationBin",
    "CalibrationResult",
    "event_calibration",
    "top_label_calibration",
]

#: Default number of calibration bins.
DEFAULT_N_BINS: int = 10

Binning = Literal["fixed_width", "equal_frequency"]


@dataclass(frozen=True)
class CalibrationBin:
    """One calibration bin.

    Attributes
    ----------
    index : int
        Bin position, 0-based, in increasing order of predicted probability.
    lower, upper : float
        Bin extent. For ``fixed_width`` binning these are the nominal edges; for
        ``equal_frequency`` they are the smallest and largest predicted
        probability actually in the bin.
    count : int
        Number of evaluated cases in the bin.
    mean_predicted : float
        Mean predicted probability (``nan`` for an empty bin).
    observed_rate : float
        Empirical frequency of the outcome/event (``nan`` for an empty bin).
    ci_low, ci_high : float
        Wilson score interval for ``observed_rate`` (``nan`` for an empty bin).
    """

    index: int
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float
    ci_low: float
    ci_high: float

    @property
    def gap(self) -> float:
        """``observed_rate - mean_predicted`` (positive means under-confident)."""
        return self.observed_rate - self.mean_predicted


@dataclass(frozen=True)
class CalibrationResult:
    """A calibration table with summary statistics and explicit denominators.

    Attributes
    ----------
    kind : str
        ``"top_label"`` or ``"event"``.
    label : str
        Description of what was calibrated (event name, or ``"top prediction"``).
    binning : str
        ``"fixed_width"`` or ``"equal_frequency"``.
    n_bins_requested : int
        The requested number of bins.
    n_total : int
        Subjects in the cohort.
    n_evaluated : int
        Subjects contributing to the table (the denominator of ECE).
    n_undefined : int
        Subjects excluded because the quantity being calibrated is not
        identifiable (see ``undefined_reasons``).
    undefined_reasons : tuple[tuple[str, str], ...]
        ``(subject_id, reason)`` for each excluded subject (ids as strings).
    ece, mce : float
        Expected / maximum calibration error (``nan`` if nothing was evaluated).
    brier : float or None
        Mean squared error of the predicted probability against the 0/1
        outcome (event calibration only; a proper score for the event).
    ci_confidence : float
        Confidence level of the per-bin Wilson intervals.
    bins : tuple[CalibrationBin, ...]
        The bins, in increasing order of predicted probability.
    """

    kind: str
    label: str
    binning: str
    n_bins_requested: int
    n_total: int
    n_evaluated: int
    n_undefined: int
    undefined_reasons: tuple[tuple[str, str], ...]
    ece: float
    mce: float
    brier: float | None
    ci_confidence: float
    bins: tuple[CalibrationBin, ...]


def wilson_interval(successes: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    r"""Wilson score interval for a binomial proportion.

    With observed proportion :math:`\hat p = s/n` and normal quantile :math:`z`,

    .. math::

        \frac{\hat p + z^2/2n \pm z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1 + z^2/n}.

    ``successes`` may be fractional (tie credit); the interval is then an
    approximation that treats the fractional count as a success count.

    Parameters
    ----------
    successes : float
        Number (or fractional credit) of successes, in ``[0, n]``.
    n : int
        Number of trials, positive.
    confidence : float, optional
        Two-sided confidence level in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(low, high)``, clipped to ``[0, 1]``.

    Examples
    --------
    >>> low, high = wilson_interval(8, 10)
    >>> round(low, 3), round(high, 3)
    (0.49, 0.943)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    # At the boundaries the exact bounds are 0 and 1; cancellation would otherwise
    # leave a ~1e-17 gap so the interval fails to contain the observed proportion.
    low = 0.0 if successes <= 0 else max(0.0, centre - half)
    high = 1.0 if successes >= n else min(1.0, centre + half)
    return low, high


def _group_indices(
    predicted: np.ndarray, n_bins: int, binning: Binning
) -> list[tuple[np.ndarray, float, float]]:
    """Partition case indices into bins; return ``(indices, lower, upper)``."""
    if binning == "fixed_width":
        idx = np.minimum((predicted * n_bins).astype(int), n_bins - 1)
        return [(np.flatnonzero(idx == b), b / n_bins, (b + 1) / n_bins) for b in range(n_bins)]
    if binning == "equal_frequency":
        if predicted.size == 0:
            return []
        order = np.argsort(predicted, kind="stable")
        groups = np.array_split(order, min(n_bins, predicted.size))
        return [(g, float(predicted[g].min()), float(predicted[g].max())) for g in groups]
    raise ValueError(f"binning must be 'fixed_width' or 'equal_frequency', got {binning!r}.")


def _calibrate(
    *,
    kind: str,
    label: str,
    predicted: list[float],
    observed: list[float],
    n_total: int,
    undefined: list[tuple[str, str]],
    n_bins: int,
    binning: Binning,
    confidence: float,
    with_brier: bool,
) -> CalibrationResult:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}.")
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.size and (pred.min() < -1e-9 or pred.max() > 1.0 + 1e-9):
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    pred = np.clip(pred, 0.0, 1.0)

    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    n = pred.size
    for b, (members, lower, upper) in enumerate(_group_indices(pred, n_bins, binning)):
        count = int(members.size)
        if count == 0:
            bins.append(CalibrationBin(b, lower, upper, 0, math.nan, math.nan, math.nan, math.nan))
            continue
        mean_pred = math.fsum(pred[members].tolist()) / count
        successes = math.fsum(obs[members].tolist())
        rate = successes / count
        lo, hi = wilson_interval(successes, count, confidence)
        bins.append(CalibrationBin(b, lower, upper, count, mean_pred, rate, lo, hi))
        gap = abs(rate - mean_pred)
        ece += count / n * gap
        mce = max(mce, gap)

    brier = None
    if with_brier:
        brier = math.fsum(((pred - obs) ** 2).tolist()) / n if n else math.nan

    return CalibrationResult(
        kind=kind,
        label=label,
        binning=binning,
        n_bins_requested=n_bins,
        n_total=n_total,
        n_evaluated=int(n),
        n_undefined=len(undefined),
        undefined_reasons=tuple(undefined),
        ece=ece if n else math.nan,
        mce=mce if n else math.nan,
        brier=brier,
        ci_confidence=confidence,
        bins=tuple(bins),
    )


def top_label_calibration(
    cohort: Cohort,
    *,
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = "fixed_width",
    confidence: float = 0.95,
) -> CalibrationResult:
    r"""Calibration of the top-probability prediction.

    For each subject the *confidence* is the largest probability the
    distribution assigns to a listed outcome, and the *correctness* is the
    fractional top-1 credit of the truth
    (:func:`~immunomath.diagnostics.top_k_accuracy` with ``k=1``): ``1`` when
    the truth is the unique mode, ``0`` when it is not, and ``1/t`` when it is
    one of ``t`` tied modes. Credit is therefore invariant to candidate order.

    Parameters
    ----------
    cohort : Cohort
        The cohort to evaluate.
    n_bins : int, optional
        Number of bins; default ``10``.
    binning : {"fixed_width", "equal_frequency"}, optional
        ``fixed_width`` uses ``n_bins`` equal intervals of ``[0, 1]`` (empty bins
        are kept and reported with count 0). ``equal_frequency`` ranks subjects
        by confidence and splits them into ``n_bins`` groups of near-equal size
        (ties are split by input order, so a bin boundary can fall inside a run
        of equal confidences).
    confidence : float, optional
        Level of the per-bin Wilson interval for the empirical accuracy.

    Returns
    -------
    CalibrationResult

    Notes
    -----
    Subjects whose truth probability is not identifiable (truth unlisted in an
    open distribution) are excluded and counted in ``n_undefined`` with reasons;
    they are never treated as incorrect.

    Examples
    --------
    >>> from immunomath import Cohort, Distribution
    >>> cohort = Cohort(
    ...     ["a", "b", "c", "d"], ["G1", "G1", "G2", "G2"],
    ...     [Distribution({"G1": 0.9, "G2": 0.1}), Distribution({"G1": 0.9, "G2": 0.1}),
    ...      Distribution({"G1": 0.9, "G2": 0.1}), Distribution({"G1": 0.2, "G2": 0.8})],
    ... )
    >>> result = top_label_calibration(cohort, n_bins=10)
    >>> result.n_evaluated
    4
    >>> round(result.ece, 3)
    0.225
    """
    predicted: list[float] = []
    observed: list[float] = []
    undefined: list[tuple[str, str]] = []
    for sid, truth, dist in zip(
        cohort.subject_ids, cohort.truths, cohort.distributions, strict=True
    ):
        if not dist.probabilities:
            undefined.append((str(sid), "distribution lists no outcomes"))
            continue
        try:
            correct = top_k_accuracy(truth, dist, k=1)
        except KeyError:
            undefined.append((str(sid), f"truth {truth!r} not identifiable (open distribution)"))
            continue
        predicted.append(max(dist.probabilities))
        observed.append(correct)
    return _calibrate(
        kind="top_label",
        label="top prediction",
        predicted=predicted,
        observed=observed,
        n_total=len(cohort),
        undefined=undefined,
        n_bins=n_bins,
        binning=binning,
        confidence=confidence,
        with_brier=False,
    )


def event_calibration(
    cohort: Cohort,
    event: Collection[Hashable],
    *,
    label: str | None = None,
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = "fixed_width",
    confidence: float = 0.95,
) -> CalibrationResult:
    """Calibration of a user-defined event (a set of outcomes).

    The predicted event probability is the summed probability of the outcomes in
    ``event`` (:meth:`~immunomath.distributions.Distribution.event_probability`);
    the observed indicator is ``1`` if the subject's truth is in ``event``.
    Typical events are "genotype contains allele X" for pharmacogenomic
    questions where allele presence matters more than exact genotype.

    Parameters
    ----------
    cohort : Cohort
        The cohort to evaluate.
    event : collection of Hashable
        The outcomes that constitute the event; must be non-empty.
    label : str, optional
        Name used in the result.
    n_bins, binning, confidence
        As in :func:`top_label_calibration`.

    Returns
    -------
    CalibrationResult
        Includes ``brier``, the mean squared error of the event probability
        (a strictly proper score for the binary event).

    Notes
    -----
    The event probability is **identifiable only for complete distributions**.
    For an open distribution (one carrying residual mass) the reported listed
    mass is only a lower bound on the event probability, so such subjects are
    excluded and counted in ``n_undefined`` rather than scored as if exact.

    Examples
    --------
    >>> from immunomath import Cohort, Distribution
    >>> cohort = Cohort(
    ...     ["a", "b"], ["G1", "G3"],
    ...     [Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1}),
    ...      Distribution({"G1": 0.2, "G2": 0.1, "G3": 0.7})],
    ... )
    >>> result = event_calibration(cohort, {"G1", "G2"}, label="G1 or G2")
    >>> result.n_evaluated, round(result.brier, 4)
    (2, 0.05)
    """
    outcomes = set(event)
    if not outcomes:
        raise ValueError("event must contain at least one outcome.")
    predicted: list[float] = []
    observed: list[float] = []
    undefined: list[tuple[str, str]] = []
    for sid, truth, dist in zip(
        cohort.subject_ids, cohort.truths, cohort.distributions, strict=True
    ):
        if not dist.is_complete:
            undefined.append(
                (
                    str(sid),
                    f"event probability not identifiable: distribution carries residual "
                    f"mass {dist.residual_mass!r}",
                )
            )
            continue
        predicted.append(dist.event_probability(outcomes))
        observed.append(1.0 if truth in outcomes else 0.0)
    return _calibrate(
        kind="event",
        label=label if label is not None else f"event of {len(outcomes)} outcome(s)",
        predicted=predicted,
        observed=observed,
        n_total=len(cohort),
        undefined=undefined,
        n_bins=n_bins,
        binning=binning,
        confidence=confidence,
        with_brier=True,
    )
