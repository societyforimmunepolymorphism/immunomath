r"""Observation-weighted proper scoring and paired model comparison.

This module adds **covariate-weighted** aggregation of the existing proper
scoring rules (:func:`~immunomath.scoring.negative_log_score` and
:func:`~immunomath.scoring.brier_score`), a **paired** comparison of two
probabilistic methods on identical observations, and **stratified** validation
with a group-balanced macro-average.

It is deliberately *domain-independent*: it knows nothing about HLA, typing
resolution, or haplotype frequencies. It consumes externally supplied positive
weights and (optionally) externally supplied stratum labels. The mapping from a
scientific covariate such as a typing-resolution score (TRS) to a weight is a
separate, optional adapter (see :mod:`immunomath.trs`); it is **not** part of
this kernel.

Motivation
----------
Conventional cohort means average predictive performance over all observations
equally. When most observations are near-deterministic (e.g. high-resolution
typings) and a minority are highly uncertain, the uncertain minority can be
diluted to statistical invisibility even though they are where modelling choices
matter most. Covariate weighting lets a caller upweight those observations
*without* looking at the outcome.

.. math::

    L_w = \frac{\sum_i w_i L_i}{\sum_i w_i},
    \qquad
    N_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.

Conditional propriety
---------------------
Each underlying scoring rule is strictly proper. A fixed, strictly positive
weight :math:`w_i` that depends only on **pre-registered covariates** — never on
the observed truth, on whether an imputation "succeeded", nor on either model's
posterior probabilities — is a constant with respect to the forecast. Scaling a
proper loss by a positive constant preserves the location of its unique minimum,
so the per-observation expected weighted loss is still minimized by honest
reporting. The weighted average therefore remains a proper criterion
**conditional on the covariates**. Weights that peek at the outcome or at a
model's predictions would break this guarantee and are a statistical error, not
merely a modelling preference.

Infinite and undefined losses
------------------------------
The log score is ``+inf`` when a model assigns the truth zero probability, and a
loss is *undefined* when the truth is unidentifiable in an open distribution
(residual mass). These are never silently dropped: weighted means that include a
finite positive weight on a ``+inf`` loss are ``+inf``; undefined observations
are excluded from aggregates and reported with explicit counts. A paired
difference ``inf - inf`` is indeterminate and is excluded (never silently
treated as a zero improvement).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .distributions import Distribution
from .scoring import brier_score, negative_log_score

__all__ = [
    "LossMetric",
    "ObservationLoss",
    "PairedWeightedComparison",
    "StratifiedComparison",
    "StratumComparison",
    "WeightedScore",
    "compare_methods_weighted",
    "per_observation_losses",
    "stratified_comparison",
    "validate_weights",
    "weighted_mean_loss",
]


class LossMetric(StrEnum):
    """A selectable per-observation proper scoring rule.

    * ``LOG`` -- the negative logarithmic score (:func:`negative_log_score`),
      in nats; ``+inf`` when the truth has zero probability.
    * ``BRIER`` -- the multiclass Brier score (:func:`brier_score`), in
      ``[0, 2]``; always finite for an identifiable truth.
    """

    LOG = "log"
    BRIER = "brier"


@dataclass(frozen=True)
class ObservationLoss:
    """One observation's loss under a chosen metric, with identifiability.

    Parameters
    ----------
    index : int
        Position of the observation in the input sequence.
    loss : float or None
        The per-observation loss: a finite float, ``math.inf`` (log score on a
        zero-probability truth), or ``None`` when the loss is *undefined*
        (truth unidentifiable in an open distribution).
    defined : bool
        ``True`` iff the truth probability is identifiable (``loss`` is a float,
        possibly ``+inf``); ``False`` iff ``loss`` is ``None``.
    reason : str or None
        For undefined observations, why the probability is not identifiable.
    """

    index: int
    loss: float | None
    defined: bool
    reason: str | None = None


def per_observation_losses(
    truths: Sequence[Hashable],
    distributions: Sequence[Distribution],
    *,
    metric: LossMetric | str = LossMetric.LOG,
    floor: float | None = None,
) -> tuple[ObservationLoss, ...]:
    r"""Compute per-observation losses, reusing the existing proper scorers.

    This performs **no** aggregation; it classifies each observation as a finite
    loss, an infinite loss (log score only), or an undefined loss (truth
    unidentifiable in an open distribution). It is the shared front end for
    :func:`weighted_mean_loss`, :func:`compare_methods_weighted`, and
    :func:`stratified_comparison`, so residual/zero-probability semantics are
    defined in exactly one place (the underlying scorers).

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes, one per observation.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``truths``.
    metric : LossMetric or str, optional
        ``"log"`` (default) or ``"brier"``.
    floor : float, optional
        Optional probability floor forwarded to the log scorer; ignored by the
        Brier scorer. See :func:`negative_log_score`.

    Returns
    -------
    tuple of ObservationLoss
        Per-observation losses in input order.

    Raises
    ------
    ValueError
        If ``truths`` and ``distributions`` differ in length, are empty, or
        ``metric`` is not recognized.

    Examples
    --------
    >>> from immunomath import Distribution
    >>> from immunomath.weighting import per_observation_losses
    >>> losses = per_observation_losses(["G1", "G2"],
    ...     [Distribution({"G1": 1.0}), Distribution({"G1": 0.5, "G2": 0.5})])
    >>> losses[0].loss
    0.0
    >>> round(losses[1].loss, 6)
    0.693147
    """
    metric = LossMetric(metric)
    if len(truths) != len(distributions):
        raise ValueError(
            f"truths and distributions must have equal length; got "
            f"{len(truths)} and {len(distributions)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot score an empty set of observations.")

    out: list[ObservationLoss] = []
    for i, (truth, dist) in enumerate(zip(truths, distributions, strict=True)):
        try:
            if metric is LossMetric.LOG:
                loss = negative_log_score(truth, dist, floor=floor)
            else:
                loss = brier_score(truth, dist)
        except KeyError:
            out.append(
                ObservationLoss(
                    index=i,
                    loss=None,
                    defined=False,
                    reason=(
                        f"truth {truth!r} is not identifiable in an open distribution "
                        f"carrying residual mass {dist.residual_mass!r}"
                    ),
                )
            )
            continue
        out.append(ObservationLoss(index=i, loss=loss, defined=True))
    return tuple(out)


def validate_weights(weights: Sequence[float], *, n: int) -> tuple[float, ...]:
    r"""Validate and return externally supplied positive weights.

    Weights must be finite and **strictly positive**. Zero, negative, ``NaN``,
    and infinite weights are rejected, as is a length mismatch. The weights are
    **not** normalized or otherwise modified — only validated and returned as a
    tuple of floats.

    Parameters
    ----------
    weights : sequence of float
        One weight per observation.
    n : int
        The required number of weights (the number of observations).

    Returns
    -------
    tuple of float
        The validated weights, unchanged.

    Raises
    ------
    ValueError
        If the length is wrong, or any weight is non-finite or non-positive.

    Examples
    --------
    >>> from immunomath.weighting import validate_weights
    >>> validate_weights([1.0, 2.0, 0.5], n=3)
    (1.0, 2.0, 0.5)
    """
    if len(weights) != n:
        raise ValueError(
            f"weights must have one entry per observation; got {len(weights)} for {n}."
        )
    validated: list[float] = []
    for i, w in enumerate(weights):
        wf = float(w)
        if math.isnan(wf):
            raise ValueError(f"weights[{i}] is NaN; weights must be finite and positive.")
        if math.isinf(wf):
            raise ValueError(f"weights[{i}] is infinite; weights must be finite and positive.")
        if wf <= 0.0:
            raise ValueError(
                f"weights[{i}] is {wf!r}; weights must be strictly positive. "
                "A zero or negative weight cannot express relative importance and "
                "would break the effective-sample-size accounting."
            )
        validated.append(wf)
    return tuple(validated)


def _effective_sample_size(weights: Sequence[float]) -> float:
    r"""Kish effective sample size :math:`(\sum w)^2 / \sum w^2`."""
    s1 = math.fsum(weights)
    s2 = math.fsum(w * w for w in weights)
    return (s1 * s1) / s2 if s2 > 0.0 else 0.0


def _weighted_mean(losses: Sequence[float], weights: Sequence[float]) -> float:
    r"""Weighted mean that returns ``+inf`` if any positively weighted loss is ``+inf``.

    Every weight here is already validated as strictly positive, so a single
    ``+inf`` loss makes the weighted mean ``+inf`` (never silently dropped).
    """
    if any(math.isinf(loss) for loss in losses):
        return math.inf
    numerator = math.fsum(w * loss for w, loss in zip(weights, losses, strict=True))
    denominator = math.fsum(weights)
    return numerator / denominator


@dataclass(frozen=True)
class WeightedScore:
    r"""Result of a weighted (and unweighted) aggregation of per-observation losses.

    Unweighted results are always reported alongside the weighted ones, so a
    weighted summary can never be read in isolation.

    Parameters
    ----------
    metric : LossMetric
        The per-observation scoring rule used.
    n_total : int
        Number of input observations.
    n_scored : int
        Identifiable observations (finite or infinite loss); the aggregation
        denominator (by count).
    n_undefined : int
        Observations excluded because the truth was unidentifiable.
    n_infinite : int
        Identifiable observations whose loss is ``+inf``.
    total_weight : float
        Sum of the weights over the ``n_scored`` observations.
    effective_sample_size : float
        Kish effective sample size over the ``n_scored`` weights,
        :math:`(\sum w)^2 / \sum w^2`. Equals ``n_scored`` for uniform weights.
    weighted_mean : float
        Weighted mean loss over the identifiable observations; ``+inf`` if any
        has an infinite loss; ``nan`` only if ``n_scored == 0``.
    unweighted_mean : float
        Ordinary (uniform-weight) mean loss over the identifiable observations,
        reported for comparison; same infinity/empty conventions.
    undefined_reasons : tuple[tuple[int, str], ...]
        ``(index, reason)`` for each excluded observation.
    """

    metric: LossMetric
    n_total: int
    n_scored: int
    n_undefined: int
    n_infinite: int
    total_weight: float
    effective_sample_size: float
    weighted_mean: float
    unweighted_mean: float
    undefined_reasons: tuple[tuple[int, str], ...]


def weighted_mean_loss(
    truths: Sequence[Hashable],
    distributions: Sequence[Distribution],
    weights: Sequence[float] | None = None,
    *,
    metric: LossMetric | str = LossMetric.LOG,
    floor: float | None = None,
) -> WeightedScore:
    r"""Aggregate per-observation proper losses with external positive weights.

    .. math::

        L_w = \frac{\sum_i w_i L_i}{\sum_i w_i}.

    With ``weights=None`` (the default) all weights are one, so the weighted and
    unweighted means coincide and this reproduces ordinary scoring exactly. The
    unweighted mean is **always** reported alongside the weighted mean.

    Weights must be externally supplied, strictly positive, and determined
    **independently of the outcome and of any model's predictions** (see the
    module docstring on conditional propriety). They are validated but never
    normalized or clipped.

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``truths``.
    weights : sequence of float, optional
        One strictly-positive weight per observation. ``None`` means uniform
        weights (ordinary scoring).
    metric : LossMetric or str, optional
        ``"log"`` (default) or ``"brier"``.
    floor : float, optional
        Optional probability floor for the log score; ignored for Brier.

    Returns
    -------
    WeightedScore
        Weighted and unweighted means, denominators, total weight, and the
        effective sample size.

    Raises
    ------
    ValueError
        If lengths differ, inputs are empty, the metric is unknown, or any
        weight is non-finite or non-positive.

    Notes
    -----
    Undefined observations (unidentifiable truth in an open distribution) are
    excluded from both means and from the weight totals, and reported via
    :attr:`WeightedScore.undefined_reasons`. An identifiable ``+inf`` loss with a
    finite positive weight makes the weighted mean ``+inf``.

    Examples
    --------
    >>> from immunomath import Distribution
    >>> from immunomath.weighting import weighted_mean_loss
    >>> truths = ["G1", "G2"]
    >>> dists = [Distribution({"G1": 0.9, "G2": 0.1}),
    ...          Distribution({"G1": 0.4, "G2": 0.6})]
    >>> uniform = weighted_mean_loss(truths, dists)
    >>> round(uniform.weighted_mean, 6) == round(uniform.unweighted_mean, 6)
    True
    >>> # The second observation (p=0.6 on the truth) has the larger loss, so
    >>> # upweighting it raises the weighted mean above the uniform mean.
    >>> upweight_second = weighted_mean_loss(truths, dists, [1.0, 3.0])
    >>> upweight_second.weighted_mean > uniform.weighted_mean
    True
    >>> round(uniform.effective_sample_size, 6)
    2.0
    """
    metric = LossMetric(metric)
    losses = per_observation_losses(truths, distributions, metric=metric, floor=floor)
    n_total = len(losses)

    if weights is None:
        raw_weights: tuple[float, ...] = tuple(1.0 for _ in range(n_total))
    else:
        raw_weights = validate_weights(weights, n=n_total)

    scored_losses: list[float] = []
    scored_weights: list[float] = []
    undefined: list[tuple[int, str]] = []
    for ol, w in zip(losses, raw_weights, strict=True):
        if ol.defined and ol.loss is not None:
            scored_losses.append(ol.loss)
            scored_weights.append(w)
        else:
            undefined.append((ol.index, ol.reason or "unidentifiable"))

    n_scored = len(scored_losses)
    n_undefined = len(undefined)
    n_infinite = sum(1 for x in scored_losses if math.isinf(x))

    if n_scored == 0:
        weighted = math.nan
        unweighted = math.nan
        total_weight = 0.0
        n_eff = 0.0
    else:
        weighted = _weighted_mean(scored_losses, scored_weights)
        unweighted = math.inf if n_infinite > 0 else math.fsum(scored_losses) / n_scored
        total_weight = math.fsum(scored_weights)
        n_eff = _effective_sample_size(scored_weights)

    return WeightedScore(
        metric=metric,
        n_total=n_total,
        n_scored=n_scored,
        n_undefined=n_undefined,
        n_infinite=n_infinite,
        total_weight=total_weight,
        effective_sample_size=n_eff,
        weighted_mean=weighted,
        unweighted_mean=unweighted,
        undefined_reasons=tuple(undefined),
    )


# ---------------------------------------------------------------------- #
# Paired comparison of two methods on identical observations.
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedWeightedComparison:
    r"""Paired comparison of an incumbent and a candidate method.

    The per-observation improvement is

    .. math::

        \Delta_i = L_{i,\mathrm{incumbent}} - L_{i,\mathrm{candidate}},

    so a **positive** :math:`\Delta_i` means the candidate assigns the truth a
    *lower* loss (an improvement). Conventional (unweighted) and weighted summaries
    are reported side by side; neither is omitted.

    Attributes
    ----------
    metric : LossMetric
        The per-observation scoring rule used for both methods.
    label_incumbent, label_candidate : str
        Human-readable names for the two methods.
    n_total : int
        Number of input observations.
    n_paired : int
        Observations with a **defined** difference (both methods identifiable
        and not an ``inf - inf``); the denominator of the paired statistics.
    n_excluded : int
        Observations excluded (undefined in either method, or ``inf - inf``).
    exclusion_reasons : tuple[tuple[int, str], ...]
        ``(index, reason)`` for each excluded observation.
    mean_loss_incumbent, mean_loss_candidate : float
        Conventional (unweighted) mean loss of each method over paired obs.
    weighted_mean_loss_incumbent, weighted_mean_loss_candidate : float
        Weighted mean loss of each method over paired obs.
    mean_improvement : float
        Conventional mean of :math:`\Delta_i` (positive favors the candidate).
    weighted_mean_improvement : float
        Weighted mean of :math:`\Delta_i`.
    n_improved, n_worsened, n_unchanged : int
        Counts of observations the candidate improves, worsens, or leaves
        unchanged (within ``tolerance``).
    total_weight : float
        Sum of weights over paired observations.
    effective_sample_size : float
        Kish effective sample size over paired weights.
    improvements : tuple[tuple[int, float], ...]
        ``(index, delta_i)`` for every paired observation, in order (the
        distribution of individual score changes).
    tolerance : float
        Absolute tolerance below which a difference counts as unchanged.
    """

    metric: LossMetric
    label_incumbent: str
    label_candidate: str
    n_total: int
    n_paired: int
    n_excluded: int
    exclusion_reasons: tuple[tuple[int, str], ...]
    mean_loss_incumbent: float
    mean_loss_candidate: float
    weighted_mean_loss_incumbent: float
    weighted_mean_loss_candidate: float
    mean_improvement: float
    weighted_mean_improvement: float
    n_improved: int
    n_worsened: int
    n_unchanged: int
    total_weight: float
    effective_sample_size: float
    improvements: tuple[tuple[int, float], ...]
    tolerance: float


def _paired_defined(a: ObservationLoss, b: ObservationLoss) -> tuple[bool, str | None]:
    """Return ``(defined, reason)`` for a paired difference of two losses."""
    if not a.defined or not b.defined:
        which = []
        if not a.defined:
            which.append("incumbent")
        if not b.defined:
            which.append("candidate")
        return False, f"truth not identifiable in {' and '.join(which)}"
    assert a.loss is not None and b.loss is not None
    if math.isinf(a.loss) and math.isinf(b.loss):
        return False, "indeterminate inf - inf (both methods assign truth zero probability)"
    return True, None


def compare_methods_weighted(
    truths: Sequence[Hashable],
    incumbent: Sequence[Distribution],
    candidate: Sequence[Distribution],
    weights: Sequence[float] | None = None,
    *,
    metric: LossMetric | str = LossMetric.LOG,
    label_incumbent: str = "incumbent",
    label_candidate: str = "candidate",
    floor: float | None = None,
    tolerance: float = 1e-12,
) -> PairedWeightedComparison:
    r"""Compare two methods on identical observations, weighted and unweighted.

    Both methods are scored on **exactly the same** truths and (if supplied) the
    same weights. The per-observation improvement is
    :math:`\Delta_i = L_{i,\mathrm{incumbent}} - L_{i,\mathrm{candidate}}`
    (positive favors the candidate). Conventional and weighted means of each
    method's loss and of the improvement are reported together.

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes shared by both methods.
    incumbent, candidate : sequence of Distribution
        The two methods' predicted distributions, aligned with ``truths``.
    weights : sequence of float, optional
        Shared strictly-positive weights; ``None`` means uniform. The same
        weights apply to both methods — a weight may **not** depend on either
        method's predictions.
    metric : LossMetric or str, optional
        ``"log"`` (default) or ``"brier"``.
    label_incumbent, label_candidate : str, optional
        Names used in the result.
    floor : float, optional
        Optional probability floor for the log score (both methods); ignored
        for Brier.
    tolerance : float, optional
        Absolute tolerance below which an improvement counts as unchanged.

    Returns
    -------
    PairedWeightedComparison

    Raises
    ------
    ValueError
        If the three sequences differ in length, are empty, the metric is
        unknown, or any weight is invalid.

    Notes
    -----
    A difference is **defined** only when both methods yield an identifiable
    score for the observation. Observations undefined in either method, and the
    indeterminate ``inf - inf`` case, are excluded and reported separately —
    never silently counted as a zero improvement. A one-sided ``+inf``
    (``inf - finite`` or ``finite - inf``) is a genuine, defined improvement or
    deterioration and is retained.

    Examples
    --------
    >>> from immunomath import Distribution
    >>> from immunomath.weighting import compare_methods_weighted
    >>> truths = ["G1", "G2"]
    >>> incumbent = [Distribution({"G1": 0.5, "G2": 0.5}),
    ...              Distribution({"G1": 0.5, "G2": 0.5})]
    >>> candidate = [Distribution({"G1": 0.9, "G2": 0.1}),
    ...              Distribution({"G1": 0.1, "G2": 0.9})]
    >>> cmp = compare_methods_weighted(truths, incumbent, candidate)
    >>> cmp.n_improved
    2
    >>> cmp.mean_improvement > 0  # candidate is better
    True
    """
    metric = LossMetric(metric)
    if not (len(truths) == len(incumbent) == len(candidate)):
        raise ValueError(
            "truths, incumbent, and candidate must have equal length; got "
            f"{len(truths)}, {len(incumbent)}, {len(candidate)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot compare an empty set of observations.")

    n_total = len(truths)
    raw_weights = (
        tuple(1.0 for _ in range(n_total))
        if weights is None
        else validate_weights(weights, n=n_total)
    )

    losses_a = per_observation_losses(truths, incumbent, metric=metric, floor=floor)
    losses_b = per_observation_losses(truths, candidate, metric=metric, floor=floor)

    paired_a: list[float] = []
    paired_b: list[float] = []
    paired_w: list[float] = []
    deltas: list[tuple[int, float]] = []
    excluded: list[tuple[int, str]] = []

    for i, (a, b, w) in enumerate(zip(losses_a, losses_b, raw_weights, strict=True)):
        defined, reason = _paired_defined(a, b)
        if not defined:
            excluded.append((i, reason or "undefined"))
            continue
        assert a.loss is not None and b.loss is not None
        la, lb = a.loss, b.loss
        delta = la - lb  # +inf or -inf possible (one-sided), never inf-inf here
        paired_a.append(la)
        paired_b.append(lb)
        paired_w.append(w)
        deltas.append((i, delta))

    n_paired = len(deltas)
    n_excluded = len(excluded)

    if n_paired == 0:
        return PairedWeightedComparison(
            metric=metric,
            label_incumbent=label_incumbent,
            label_candidate=label_candidate,
            n_total=n_total,
            n_paired=0,
            n_excluded=n_excluded,
            exclusion_reasons=tuple(excluded),
            mean_loss_incumbent=math.nan,
            mean_loss_candidate=math.nan,
            weighted_mean_loss_incumbent=math.nan,
            weighted_mean_loss_candidate=math.nan,
            mean_improvement=math.nan,
            weighted_mean_improvement=math.nan,
            n_improved=0,
            n_worsened=0,
            n_unchanged=0,
            total_weight=0.0,
            effective_sample_size=0.0,
            improvements=(),
            tolerance=tolerance,
        )

    delta_values = [d for _, d in deltas]
    n_improved = sum(1 for d in delta_values if d > tolerance)
    n_worsened = sum(1 for d in delta_values if d < -tolerance)
    n_unchanged = n_paired - n_improved - n_worsened

    def umean(xs: Sequence[float]) -> float:
        return math.inf if any(math.isinf(x) for x in xs) else math.fsum(xs) / len(xs)

    return PairedWeightedComparison(
        metric=metric,
        label_incumbent=label_incumbent,
        label_candidate=label_candidate,
        n_total=n_total,
        n_paired=n_paired,
        n_excluded=n_excluded,
        exclusion_reasons=tuple(excluded),
        mean_loss_incumbent=umean(paired_a),
        mean_loss_candidate=umean(paired_b),
        weighted_mean_loss_incumbent=_weighted_mean(paired_a, paired_w),
        weighted_mean_loss_candidate=_weighted_mean(paired_b, paired_w),
        mean_improvement=umean(delta_values),
        weighted_mean_improvement=_weighted_mean(delta_values, paired_w),
        n_improved=n_improved,
        n_worsened=n_worsened,
        n_unchanged=n_unchanged,
        total_weight=math.fsum(paired_w),
        effective_sample_size=_effective_sample_size(paired_w),
        improvements=tuple(deltas),
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------- #
# Stratified validation with a group-balanced macro-average.
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class StratumComparison:
    """Per-stratum paired comparison summary.

    Attributes
    ----------
    label : Hashable
        The stratum label.
    n : int
        Number of observations in the stratum (``n_paired`` for improvements).
    mean_log_incumbent, mean_log_candidate : float
        Mean log score of each method within the stratum (over paired obs).
    mean_brier_incumbent, mean_brier_candidate : float
        Mean Brier score of each method within the stratum (over paired obs).
    mean_improvement_log, mean_improvement_brier : float
        Mean per-observation improvement (incumbent - candidate) within the
        stratum, for the log and Brier metrics. Positive favors the candidate.
    n_improved, n_worsened, n_unchanged : int
        Improvement counts for the primary metric (log).
    """

    label: Hashable
    n: int
    mean_log_incumbent: float
    mean_log_candidate: float
    mean_brier_incumbent: float
    mean_brier_candidate: float
    mean_improvement_log: float
    mean_improvement_brier: float
    n_improved: int
    n_worsened: int
    n_unchanged: int


@dataclass(frozen=True)
class StratifiedComparison:
    r"""Stratified comparison with a group-balanced macro-average.

    The macro-average gives each **nonempty** stratum equal aggregate weight,

    .. math::

        L_{\mathrm{balanced}} = \frac{1}{K} \sum_{k=1}^{K} \bar L_k,

    where :math:`K` is the number of nonempty strata and :math:`\bar L_k` is the
    (conventional, unweighted) mean loss within stratum :math:`k`. The
    conventional observation-weighted mean over the whole cohort is reported
    alongside it. Comparing the two reveals a candidate that improves the
    cohort mean while degrading a small, poorly resolved stratum.

    Attributes
    ----------
    strata : tuple[StratumComparison, ...]
        Per-stratum summaries, ordered by first appearance of each label.
    empty_strata : tuple[Hashable, ...]
        Declared strata (from ``stratum_order``) that contained no observations;
        excluded from the macro-average.
    macro_mean_log_incumbent, macro_mean_log_candidate : float
        Balanced macro-average log score of each method over nonempty strata.
    macro_mean_improvement_log : float
        Balanced macro-average of the per-stratum mean log improvement.
    overall_mean_log_incumbent, overall_mean_log_candidate : float
        Conventional cohort-wide (observation-weighted) mean log score.
    overall_mean_improvement_log : float
        Conventional cohort-wide mean log improvement.
    """

    strata: tuple[StratumComparison, ...]
    empty_strata: tuple[Hashable, ...]
    macro_mean_log_incumbent: float
    macro_mean_log_candidate: float
    macro_mean_improvement_log: float
    overall_mean_log_incumbent: float
    overall_mean_log_candidate: float
    overall_mean_improvement_log: float


def _macro_average(values: Sequence[float]) -> float:
    """Mean over nonempty strata, returning ``+inf`` if any stratum is infinite."""
    if not values:
        return math.nan
    if any(math.isinf(v) for v in values):
        return math.inf
    return math.fsum(values) / len(values)


def stratified_comparison(
    truths: Sequence[Hashable],
    incumbent: Sequence[Distribution],
    candidate: Sequence[Distribution],
    strata: Sequence[Hashable],
    *,
    stratum_order: Sequence[Hashable] | None = None,
    label_incumbent: str = "incumbent",
    label_candidate: str = "candidate",
    floor: float | None = None,
    tolerance: float = 1e-12,
) -> StratifiedComparison:
    r"""Compare two methods within caller-defined strata, with a macro-average.

    Each observation carries a caller-supplied stratum label (for example a
    coarse TRS band: ``"low"``, ``"intermediate"``, ``"high"``). Within each
    nonempty stratum this reports mean log and Brier scores for both methods and
    the paired improvement; across strata it reports a group-balanced
    macro-average alongside the conventional cohort-wide mean.

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes.
    incumbent, candidate : sequence of Distribution
        The two methods' predicted distributions, aligned with ``truths``.
    strata : sequence of Hashable
        Stratum label per observation, aligned with ``truths``.
    stratum_order : sequence of Hashable, optional
        The declared set/order of strata. Any declared stratum with no
        observations is reported in :attr:`StratifiedComparison.empty_strata`
        and **excluded** from the macro-average (an empty group contributes no
        mean). If ``None``, strata are taken in order of first appearance and no
        stratum can be empty by construction.
    label_incumbent, label_candidate : str, optional
        Names used in the result.
    floor : float, optional
        Optional probability floor for the log score; ignored for Brier.
    tolerance : float, optional
        Absolute tolerance below which an improvement counts as unchanged.

    Returns
    -------
    StratifiedComparison

    Raises
    ------
    ValueError
        If the aligned sequences differ in length, are empty, or ``strata``
        contains a label not present in a non-``None`` ``stratum_order``.

    Notes
    -----
    **Empty strata.** A declared stratum with zero observations has no mean to
    contribute. It is listed explicitly and dropped from the macro-average
    denominator :math:`K`, rather than being charged a zero or a ``NaN`` (which
    would silently bias the balanced score). Per-stratum statistics are computed
    over that stratum's **paired** observations, so a stratum in which every
    observation is undefined/``inf - inf`` also contributes no comparison.

    Examples
    --------
    >>> from immunomath import Distribution
    >>> from immunomath.weighting import stratified_comparison
    >>> truths = ["G1", "G1", "G2"]
    >>> inc = [Distribution({"G1": 0.9, "G2": 0.1})] * 2 + [
    ...     Distribution({"G1": 0.5, "G2": 0.5})]
    >>> cand = [Distribution({"G1": 0.9, "G2": 0.1})] * 2 + [
    ...     Distribution({"G1": 0.1, "G2": 0.9})]
    >>> strata = ["high", "high", "low"]
    >>> result = stratified_comparison(truths, inc, cand, strata,
    ...                                stratum_order=["low", "high"])
    >>> [s.label for s in result.strata]
    ['low', 'high']
    >>> result.strata[0].mean_improvement_log > 0  # candidate better in 'low'
    True
    """
    if not (len(truths) == len(incumbent) == len(candidate) == len(strata)):
        raise ValueError(
            "truths, incumbent, candidate, and strata must have equal length; got "
            f"{len(truths)}, {len(incumbent)}, {len(candidate)}, {len(strata)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot compare an empty set of observations.")

    if stratum_order is None:
        order: list[Hashable] = []
        seen: set[Hashable] = set()
        for s in strata:
            if s not in seen:
                seen.add(s)
                order.append(s)
    else:
        order = list(stratum_order)
        declared = set(order)
        unknown = {s for s in strata if s not in declared}
        if unknown:
            bad = sorted(map(repr, unknown))
            raise ValueError(f"strata contains labels not present in stratum_order: {bad}.")

    # Group observation indices by stratum label.
    groups: dict[Hashable, list[int]] = {label: [] for label in order}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    stratum_results: list[StratumComparison] = []
    empty: list[Hashable] = []
    macro_log_inc: list[float] = []
    macro_log_cand: list[float] = []
    macro_improve_log: list[float] = []

    for label in order:
        idx = groups.get(label, [])
        if not idx:
            empty.append(label)
            continue
        g_truths = [truths[i] for i in idx]
        g_inc = [incumbent[i] for i in idx]
        g_cand = [candidate[i] for i in idx]

        log_cmp = compare_methods_weighted(
            g_truths,
            g_inc,
            g_cand,
            metric=LossMetric.LOG,
            label_incumbent=label_incumbent,
            label_candidate=label_candidate,
            floor=floor,
            tolerance=tolerance,
        )
        brier_cmp = compare_methods_weighted(
            g_truths,
            g_inc,
            g_cand,
            metric=LossMetric.BRIER,
            label_incumbent=label_incumbent,
            label_candidate=label_candidate,
            tolerance=tolerance,
        )

        stratum_results.append(
            StratumComparison(
                label=label,
                n=log_cmp.n_paired,
                mean_log_incumbent=log_cmp.mean_loss_incumbent,
                mean_log_candidate=log_cmp.mean_loss_candidate,
                mean_brier_incumbent=brier_cmp.mean_loss_incumbent,
                mean_brier_candidate=brier_cmp.mean_loss_candidate,
                mean_improvement_log=log_cmp.mean_improvement,
                mean_improvement_brier=brier_cmp.mean_improvement,
                n_improved=log_cmp.n_improved,
                n_worsened=log_cmp.n_worsened,
                n_unchanged=log_cmp.n_unchanged,
            )
        )
        if log_cmp.n_paired > 0:
            macro_log_inc.append(log_cmp.mean_loss_incumbent)
            macro_log_cand.append(log_cmp.mean_loss_candidate)
            macro_improve_log.append(log_cmp.mean_improvement)

    # Conventional cohort-wide (observation-weighted) log comparison.
    overall = compare_methods_weighted(
        list(truths),
        list(incumbent),
        list(candidate),
        metric=LossMetric.LOG,
        label_incumbent=label_incumbent,
        label_candidate=label_candidate,
        floor=floor,
        tolerance=tolerance,
    )

    return StratifiedComparison(
        strata=tuple(stratum_results),
        empty_strata=tuple(empty),
        macro_mean_log_incumbent=_macro_average(macro_log_inc),
        macro_mean_log_candidate=_macro_average(macro_log_cand),
        macro_mean_improvement_log=_macro_average(macro_improve_log),
        overall_mean_log_incumbent=overall.mean_loss_incumbent,
        overall_mean_log_candidate=overall.mean_loss_candidate,
        overall_mean_improvement_log=overall.mean_improvement,
    )
