"""Risk-coverage analysis with abstention.

A selective predictor may *decline* to answer. Sweeping a confidence threshold
traces a **risk-coverage curve**: as the threshold rises, fewer subjects are
retained (lower *coverage*) and, if the confidence score is informative, the
retained predictions are more accurate (lower *risk*).

Definitions
-----------
For a threshold ``t`` a subject is **retained** if its confidence score is at
least ``t`` and its predictor did not abstain.

*   ``coverage`` = retained / evaluable subjects. Subjects the predictor
    abstained on stay in the denominator, so abstention lowers coverage.
*   ``accuracy`` = mean fractional top-1 credit over retained subjects.
*   ``error_rate`` = ``1 - accuracy``.
*   ``mean_log_loss`` = mean negative log score over retained subjects (``+inf``
    if any retained subject has an identifiable-zero truth).

Accuracy and log loss are reported **together** because selecting on
confidence can improve one while hiding the other: a confidence score that
retains only cases with large top probability can raise accuracy yet still
retain catastrophic overconfident errors, which only the log loss reveals.

The confidence score is user-defined (see :func:`risk_coverage`). The default is
the top predicted probability.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from .cohort import Cohort
from .diagnostics import top_k_accuracy
from .distributions import Distribution
from .scoring import negative_log_score

__all__ = [
    "ConfidenceScore",
    "RiskCoveragePoint",
    "RiskCoverageResult",
    "risk_coverage",
]

#: A confidence score: a callable on a distribution, or a per-subject mapping.
#: ``None`` means the predictor abstained for that subject.
ConfidenceScore = Callable[[Distribution], float | None] | Mapping[Hashable, float | None]


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One operating point of the risk-coverage curve.

    Attributes
    ----------
    threshold : float
        Retain subjects with confidence score ``>= threshold``.
    n_retained : int
        Retained subjects.
    coverage : float
        ``n_retained / n_evaluable``.
    accuracy, error_rate : float
        Over retained subjects (``nan`` if none retained).
    mean_log_loss : float
        Mean negative log score over retained subjects; ``+inf`` if any retained
        subject has infinite loss (``nan`` if none retained).
    n_infinite_retained : int
        Retained subjects with infinite loss.
    """

    threshold: float
    n_retained: int
    coverage: float
    accuracy: float
    error_rate: float
    mean_log_loss: float
    n_infinite_retained: int


@dataclass(frozen=True)
class RiskCoverageResult:
    """A risk-coverage curve with explicit denominators.

    Attributes
    ----------
    score_source : str
        ``"top_probability"``, ``"callable"`` or ``"mapping"``.
    n_total : int
        Subjects in the cohort.
    n_evaluable : int
        Subjects with an identifiable truth (the coverage denominator).
    n_undefined : int
        Subjects excluded because the truth probability is not identifiable.
    n_abstained : int
        Evaluable subjects on which the predictor abstained (never retained).
    full_coverage : RiskCoveragePoint
        Metrics for *all* evaluable subjects that did not abstain, at the lowest
        threshold; the reference operating point.
    points : tuple[RiskCoveragePoint, ...]
        Operating points in order of increasing coverage (decreasing threshold).
    """

    score_source: str
    n_total: int
    n_evaluable: int
    n_undefined: int
    n_abstained: int
    full_coverage: RiskCoveragePoint | None
    points: tuple[RiskCoveragePoint, ...]


def _point(
    threshold: float, members: list[int], correct: list[float], loss: list[float], n_eval: int
) -> RiskCoveragePoint:
    n = len(members)
    if n == 0:
        return RiskCoveragePoint(threshold, 0, 0.0, math.nan, math.nan, math.nan, 0)
    acc = math.fsum(correct[i] for i in members) / n
    n_inf = sum(1 for i in members if math.isinf(loss[i]))
    mean_loss = math.inf if n_inf else math.fsum(loss[i] for i in members) / n
    return RiskCoveragePoint(threshold, n, n / n_eval, acc, 1.0 - acc, mean_loss, n_inf)


def risk_coverage(
    cohort: Cohort,
    *,
    confidence: ConfidenceScore | None = None,
    thresholds: Sequence[float] | None = None,
    floor: float | None = None,
) -> RiskCoverageResult:
    """Compute a risk-coverage curve, optionally with abstention.

    Parameters
    ----------
    cohort : Cohort
        The cohort to evaluate.
    confidence : callable or mapping, optional
        The confidence score used to rank and select subjects.

        * ``None`` (default): the top predicted probability.
        * a callable ``f(distribution) -> float | None``.
        * a mapping ``subject_id -> float | None`` (every subject must be
          present).

        A value of ``None`` means the predictor **abstained** for that subject.
        Scores must not be NaN.
    thresholds : sequence of float, optional
        Evaluate exactly these thresholds. By default every distinct score is
        used, which traces the full curve. Subjects tied on score are always
        retained or dropped together, so the curve does not depend on input
        order.
    floor : float, optional
        Optional probability floor passed to the log score.

    Returns
    -------
    RiskCoverageResult

    Raises
    ------
    ValueError
        If a score is NaN or a mapping omits a subject.

    Notes
    -----
    Subjects with a non-identifiable truth (open-world residual) are excluded and
    counted in ``n_undefined``; they do not enter the coverage denominator.

    Examples
    --------
    >>> from immunomath import Cohort, Distribution
    >>> cohort = Cohort(
    ...     ["a", "b", "c", "d"], ["G1", "G1", "G2", "G1"],
    ...     [Distribution({"G1": 0.95, "G2": 0.05}), Distribution({"G1": 0.8, "G2": 0.2}),
    ...      Distribution({"G1": 0.7, "G2": 0.3}), Distribution({"G1": 0.6, "G2": 0.4})],
    ... )
    >>> result = risk_coverage(cohort)
    >>> [(p.n_retained, round(p.accuracy, 3)) for p in result.points]
    [(1, 1.0), (2, 1.0), (3, 0.667), (4, 0.75)]
    """
    if floor is not None and not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must lie in (0, 1], got {floor!r}.")

    if confidence is None:
        source = "top_probability"
    elif callable(confidence):
        source = "callable"
    else:
        source = "mapping"

    scores: list[float | None] = []
    correct: list[float] = []
    loss: list[float] = []
    n_undefined = 0
    for sid, truth, dist in zip(
        cohort.subject_ids, cohort.truths, cohort.distributions, strict=True
    ):
        try:
            c = top_k_accuracy(truth, dist, k=1)
            ls = negative_log_score(truth, dist, floor=floor)
        except KeyError:
            n_undefined += 1
            continue
        if confidence is None:
            s: float | None = max(dist.probabilities) if dist.probabilities else None
        elif callable(confidence):
            s = confidence(dist)
        else:
            if sid not in confidence:
                raise ValueError(f"confidence mapping has no entry for subject {sid!r}.")
            s = confidence[sid]
        if s is not None and math.isnan(s):
            raise ValueError(f"confidence score for subject {sid!r} is NaN.")
        scores.append(s)
        correct.append(c)
        loss.append(ls)

    n_eval = len(scores)
    n_abstained = sum(1 for s in scores if s is None)

    if thresholds is None:
        cut_points = sorted({s for s in scores if s is not None}, reverse=True)
    else:
        cut_points = sorted(set(thresholds), reverse=True)

    points: list[RiskCoveragePoint] = []
    for t in cut_points:
        members = [i for i, s in enumerate(scores) if s is not None and s >= t]
        points.append(_point(t, members, correct, loss, max(n_eval, 1)))

    full = None
    if n_eval - n_abstained > 0:
        everyone = [i for i, s in enumerate(scores) if s is not None]
        lowest = min(s for s in scores if s is not None)
        full = _point(lowest, everyone, correct, loss, n_eval)

    return RiskCoverageResult(
        score_source=source,
        n_total=len(cohort),
        n_evaluable=n_eval,
        n_undefined=n_undefined,
        n_abstained=n_abstained,
        full_coverage=full,
        points=tuple(points),
    )
