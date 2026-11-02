"""Proper scoring rules for probabilistic predictions.

This module implements *strictly proper scoring rules*: the negative
logarithmic score and the multiclass Brier score. A scoring rule is **proper**
if a forecaster minimizes their expected penalty by reporting their true belief,
and **strictly proper** if that true belief is the *unique* minimizer. Proper
scoring rules therefore reward both correctness and honest calibration, and
cannot be gamed by hedging or overconfidence.

These functions are pure: they take a truth outcome and a
:class:`~immunomath.distributions.Distribution` and return a float. They do not
mutate state, perform I/O, or silently repair inputs.

See Also
--------
immunomath.diagnostics
    Descriptive, non-proper diagnostics (entropy, rank, accuracy). Those
    quantities describe a distribution but do **not** constitute proper scoring
    rules and must not be interpreted as measures of forecast quality.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .distributions import Distribution

__all__ = [
    "CohortAccounting",
    "CohortScore",
    "ScoreClass",
    "SubjectScore",
    "brier_score",
    "mean_brier_score",
    "mean_negative_log_score",
    "negative_log_score",
    "score_cohort_accounting",
]


@dataclass(frozen=True)
class CohortScore:
    """Result of scoring a cohort: individual values plus the cohort mean.

    Parameters
    ----------
    individual : tuple of float
        The per-subject scores, in input order. May contain ``math.inf``.
    mean : float
        The arithmetic mean of ``individual``. If any individual score is
        ``+inf``, the mean is ``+inf`` (infinite losses are never silently
        discarded).
    n_infinite : int
        The number of individual scores equal to ``+inf``.
    """

    individual: tuple[float, ...]
    mean: float
    n_infinite: int


def negative_log_score(
    truth: Hashable,
    distribution: Distribution,
    *,
    floor: float | None = None,
) -> float:
    r"""Compute the negative logarithmic score (log loss) for one prediction.

    For an observed true outcome :math:`g` and predicted distribution
    :math:`P`, the negative logarithmic score is

    .. math::

        L = -\ln P(g).

    It uses the **natural logarithm**, so the score is measured in *nats*.

    Parameters
    ----------
    truth : Hashable
        The observed true outcome.
    distribution : Distribution
        The predicted probability distribution.
    floor : float, optional
        An optional, explicitly configured probability floor in ``(0, 1]``. When
        provided, the probability of truth is replaced by ``max(p, floor)``
        before taking the logarithm. This is an **opt-in** convenience for
        numerical experiments only; it is off by default. ImmunoMath performs no
        implicit epsilon clipping.

    Returns
    -------
    float
        The negative log score in nats. ``0.0`` for a perfect prediction;
        ``math.inf`` when the truth has zero probability (and no floor is set).

    Raises
    ------
    KeyError
        If ``truth`` is not listed in an *open* distribution (one carrying
        residual mass), its probability is not identifiable and the exact-outcome
        log score is undefined. Use event-level scoring instead.
    ValueError
        If ``floor`` is supplied and does not lie in ``(0, 1]``.

    Notes
    -----
    **Why it is strictly proper.** The expected score under a forecaster's true
    belief :math:`Q` is minimized uniquely when the reported distribution
    :math:`P` equals :math:`Q`; this follows from Gibbs' inequality. Honest
    reporting is therefore optimal.

    **Sensitivity to overconfidence.** Because :math:`-\ln p \to \infty` as
    :math:`p \to 0^+`, assigning a vanishing probability to an outcome that then
    occurs incurs an unbounded penalty. A single confidently wrong prediction can
    dominate a cohort mean. This sensitivity is a feature for safety-critical
    evaluation, but it motivates the catastrophic-error analysis of Milestone 2.

    Examples
    --------
    >>> from immunomath import Distribution, negative_log_score
    >>> d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
    >>> round(negative_log_score("G1", d), 6)
    0.223144
    >>> negative_log_score("G1", Distribution({"G1": 1.0}))
    0.0
    >>> negative_log_score("G2", Distribution({"G1": 1.0, "G2": 0.0}))
    inf
    """
    if floor is not None and not (0.0 < floor <= 1.0):
        raise ValueError(f"floor must lie in (0, 1], got {floor!r}.")

    p = distribution.probability(truth)  # may raise KeyError for open dists
    if floor is not None:
        p = max(p, floor)
    if p <= 0.0:
        return math.inf
    # Add 0.0 to normalize -0.0 (from -log(1.0)) to a positive zero.
    return -math.log(p) + 0.0


def mean_negative_log_score(
    truths: Sequence[Hashable],
    distributions: Sequence[Distribution],
    *,
    floor: float | None = None,
) -> CohortScore:
    r"""Compute per-subject and mean negative log score for a cohort.

    The cohort mean is

    .. math::

        \bar{L} = \frac{1}{N} \sum_{i=1}^{N} L_i .

    Infinite individual losses are **not** discarded: if any subject has an
    infinite loss, the reported mean is ``+inf``. The number of infinite losses
    is reported separately so callers can diagnose them.

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes, one per subject.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``truths``.
    floor : float, optional
        See :func:`negative_log_score`.

    Returns
    -------
    CohortScore
        Individual scores, the mean, and the count of infinite losses.

    Raises
    ------
    ValueError
        If ``truths`` and ``distributions`` have different lengths, or are empty.

    Examples
    --------
    >>> from immunomath import Distribution, mean_negative_log_score
    >>> d1 = Distribution({"G1": 1.0})
    >>> d2 = Distribution({"G1": 0.5, "G2": 0.5})
    >>> result = mean_negative_log_score(["G1", "G2"], [d1, d2])
    >>> round(result.mean, 6)
    0.346574
    >>> result.n_infinite
    0
    """
    if len(truths) != len(distributions):
        raise ValueError(
            f"truths and distributions must have equal length; got "
            f"{len(truths)} and {len(distributions)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot score an empty cohort.")

    individual = tuple(
        negative_log_score(t, d, floor=floor) for t, d in zip(truths, distributions, strict=True)
    )
    n_infinite = sum(1 for x in individual if math.isinf(x))
    mean = math.inf if n_infinite > 0 else math.fsum(individual) / len(individual)
    return CohortScore(individual=individual, mean=mean, n_infinite=n_infinite)


def brier_score(truth: Hashable, distribution: Distribution) -> float:
    r"""Compute the multiclass Brier score (sum form) for one prediction.

    Using the complete probability distribution, the Brier score is

    .. math::

        B = \sum_{k} \left( p_k - y_k \right)^2 ,

    where :math:`y_k = 1` for the true outcome and :math:`0` otherwise. This is
    the **sum form**, with range :math:`[0, 2]`: a perfect prediction scores
    ``0``, and the worst case (all mass on a single wrong outcome) scores ``2``.

    Relationship to binary Brier scoring
    -------------------------------------
    The conventional *binary* Brier score is :math:`(p - y)^2`, with range
    :math:`[0, 1]`. Restricting the multiclass sum form to two complementary
    classes yields :math:`(p-y)^2 + ((1-p)-(1-y))^2 = 2 (p-y)^2`, exactly
    **twice** the binary score. ImmunoMath reports the sum form by default for
    consistency across arbitrary numbers of classes.

    Residual mass
    -------------
    Only explicitly listed outcomes contribute. For an *open* distribution the
    residual mass is **not** expanded into hypothetical outcomes; the squared
    error is computed over the listed support plus the truth. If the truth is
    not listed in an open distribution its exact probability is unidentifiable
    and a :class:`KeyError` is raised.

    Parameters
    ----------
    truth : Hashable
        The observed true outcome.
    distribution : Distribution
        The predicted probability distribution.

    Returns
    -------
    float
        The Brier score in ``[0, 2]``.

    Raises
    ------
    KeyError
        If ``truth`` is unidentifiable in an open distribution.

    Examples
    --------
    >>> from immunomath import Distribution, brier_score
    >>> d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
    >>> round(brier_score("G1", d), 10)
    0.065
    >>> brier_score("G1", Distribution({"G1": 1.0}))
    0.0
    """
    p_truth = distribution.probability(truth)  # may raise KeyError for open dists

    terms: list[float] = []
    truth_listed = truth in distribution
    for outcome in distribution:
        p = distribution[outcome]
        y = 1.0 if outcome == truth else 0.0
        terms.append((p - y) ** 2)
    if not truth_listed:
        # Complete distribution, truth omitted: y=1 but p=0 for the truth class.
        terms.append((p_truth - 1.0) ** 2)
    return math.fsum(terms)


def mean_brier_score(
    truths: Sequence[Hashable],
    distributions: Sequence[Distribution],
) -> float:
    r"""Compute the mean multiclass Brier score over a cohort.

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``truths``.

    Returns
    -------
    float
        The mean Brier score in ``[0, 2]``.

    Raises
    ------
    ValueError
        If lengths differ or the cohort is empty.

    Examples
    --------
    >>> from immunomath import Distribution, mean_brier_score
    >>> d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
    >>> round(mean_brier_score(["G1"], [d]), 10)
    0.065
    """
    if len(truths) != len(distributions):
        raise ValueError(
            f"truths and distributions must have equal length; got "
            f"{len(truths)} and {len(distributions)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot score an empty cohort.")
    values = [brier_score(t, d) for t, d in zip(truths, distributions, strict=True)]
    return math.fsum(values) / len(values)


def _as_sequence(x: Iterable[Hashable]) -> Sequence[Hashable]:
    """Return ``x`` as a concrete sequence (helper for defensive copying)."""
    return list(x)


# ---------------------------------------------------------------------- #
# Undefined-subject accounting (Milestone 2, Decision 3)
# ---------------------------------------------------------------------- #
class ScoreClass(StrEnum):
    """Classification of a single subject's identifiability under scoring.

    The three cases are mutually exclusive and jointly exhaustive:

    * ``FINITE`` -- the truth has an *identifiable, strictly positive*
      probability, so the logarithmic score is a finite real number.
    * ``INFINITE`` -- the truth has an *identifiable zero* probability
      (explicitly listed with probability 0, or unlisted in a *complete*
      distribution), so the logarithmic score is ``+inf``. This is a genuine,
      identifiable outcome, not a defect: the forecaster ruled out the truth.
    * ``UNDEFINED`` -- the truth's probability is *not identifiable*, e.g. it is
      unlisted in an *open* distribution carrying residual mass. No score exists;
      the subject cannot enter the mean and is accounted for separately.
    """

    FINITE = "finite"
    INFINITE = "infinite"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class SubjectScore:
    """Per-subject scoring outcome with its identifiability classification.

    Parameters
    ----------
    index : int
        The subject's position in the input cohort.
    score_class : ScoreClass
        Whether the subject's score is finite, infinite, or undefined.
    score : float or None
        The negative log score in nats: a finite float for ``FINITE``,
        ``math.inf`` for ``INFINITE``, and ``None`` for ``UNDEFINED``.
    reason : str or None
        For ``UNDEFINED`` subjects, why the probability is not identifiable.
    """

    index: int
    score_class: ScoreClass
    score: float | None
    reason: str | None = None


@dataclass(frozen=True)
class CohortAccounting:
    """Cohort accounting distinguishing finite, infinite, and undefined subjects.

    Never silently drops infinite losses or undefined subjects: both are counted
    with explicit denominators, and undefined subjects carry reasons.

    Parameters
    ----------
    subjects : tuple of SubjectScore
        Per-subject outcomes, in input order.
    n_total, n_scored, n_undefined, n_infinite : int
        Counts. ``n_scored`` (finite + infinite, the *identifiable* subjects) is
        the denominator of :attr:`primary_mean`.
    primary_mean : float
        Mean negative log score over the identifiable subjects; ``+inf`` if any
        identifiable subject has an infinite loss; ``nan`` only when
        ``n_scored == 0`` (reported explicitly, never hidden).
    """

    subjects: tuple[SubjectScore, ...]
    n_total: int
    n_scored: int
    n_undefined: int
    n_infinite: int
    primary_mean: float

    @property
    def n_finite(self) -> int:
        """Number of identifiable subjects with a finite loss."""
        return self.n_scored - self.n_infinite

    @property
    def undefined_reasons(self) -> tuple[tuple[int, str], ...]:
        """``(index, reason)`` pairs for every undefined subject, in order."""
        return tuple(
            (s.index, s.reason or "unidentifiable")
            for s in self.subjects
            if s.score_class is ScoreClass.UNDEFINED
        )

    @property
    def mean_label(self) -> str:
        """Human-readable label describing what ``primary_mean`` applies to."""
        if self.n_undefined == 0:
            return f"mean over all {self.n_scored} identifiable subjects"
        return (
            f"mean over {self.n_scored} identifiable subjects "
            f"({self.n_undefined} of {self.n_total} undefined and excluded)"
        )


def score_cohort_accounting(
    truths: Sequence[Hashable],
    distributions: Sequence[Distribution],
    *,
    floor: float | None = None,
    strict: bool = False,
) -> CohortAccounting:
    r"""Score a cohort with explicit finite / infinite / undefined accounting.

    Each subject is classified (see :class:`ScoreClass`): identifiable with
    positive probability -> ``FINITE``; identifiable with zero probability ->
    ``INFINITE`` (log score ``+inf``); not identifiable -> ``UNDEFINED``
    (recorded with a reason, excluded from the mean).

    The **primary mean is a proper score** over the identifiable subjects
    (finite + infinite); a single identifiable zero makes it ``+inf`` and is
    never silently discarded (contrast the *descriptive* trimmed mean, which must
    never replace it).

    Parameters
    ----------
    truths : sequence of Hashable
        Observed true outcomes, one per subject.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``truths``.
    floor : float, optional
        Optional probability floor in ``(0, 1]`` applied to finite lookups; see
        :func:`negative_log_score`.
    strict : bool, optional
        If ``True``, raise :class:`ValueError` when any subject is undefined.

    Returns
    -------
    CohortAccounting
        Per-subject classifications and cohort-level counts and mean.

    Raises
    ------
    ValueError
        If lengths differ, the cohort is empty, ``floor`` is out of range, or
        ``strict`` is set and any subject is undefined.

    Examples
    --------
    >>> from immunomath import Distribution, Residual
    >>> from immunomath.scoring import score_cohort_accounting
    >>> complete = Distribution({"G1": 0.5, "G2": 0.5})
    >>> open_d = Distribution({"G1": 0.5, Residual(0.5): None})
    >>> acc = score_cohort_accounting(["G1", "G99"], [complete, open_d])
    >>> acc.n_total, acc.n_scored, acc.n_undefined, acc.n_infinite
    (2, 1, 1, 0)
    >>> round(acc.primary_mean, 6)
    0.693147
    >>> acc.undefined_reasons[0][0]
    1
    """
    if floor is not None and not (0.0 < floor <= 1.0):
        raise ValueError(f"floor must lie in (0, 1], got {floor!r}.")
    if len(truths) != len(distributions):
        raise ValueError(
            f"truths and distributions must have equal length; got "
            f"{len(truths)} and {len(distributions)}."
        )
    if len(truths) == 0:
        raise ValueError("Cannot score an empty cohort.")

    subjects: list[SubjectScore] = []
    for i, (truth, dist) in enumerate(zip(truths, distributions, strict=True)):
        try:
            p = dist.probability(truth)
        except KeyError:
            subjects.append(
                SubjectScore(
                    index=i,
                    score_class=ScoreClass.UNDEFINED,
                    score=None,
                    reason=_undefined_reason(truth, dist),
                )
            )
            continue
        if floor is not None:
            p = max(p, floor)
        if p <= 0.0:
            subjects.append(SubjectScore(index=i, score_class=ScoreClass.INFINITE, score=math.inf))
        else:
            subjects.append(
                SubjectScore(index=i, score_class=ScoreClass.FINITE, score=-math.log(p) + 0.0)
            )

    n_total = len(subjects)
    n_undefined = sum(1 for s in subjects if s.score_class is ScoreClass.UNDEFINED)
    n_infinite = sum(1 for s in subjects if s.score_class is ScoreClass.INFINITE)
    n_scored = n_total - n_undefined

    if strict and n_undefined > 0:
        bad = [s.index for s in subjects if s.score_class is ScoreClass.UNDEFINED]
        raise ValueError(
            f"strict mode: cohort contains {n_undefined} undefined subject(s) "
            f"(indices {bad}). Their truth probabilities are not identifiable; "
            "refusing to produce a cohort score."
        )

    if n_scored == 0:
        primary_mean = math.nan
    elif n_infinite > 0:
        primary_mean = math.inf
    else:
        finite_scores = [
            s.score for s in subjects if s.score_class is ScoreClass.FINITE and s.score is not None
        ]
        primary_mean = math.fsum(finite_scores) / n_scored

    return CohortAccounting(
        subjects=tuple(subjects),
        n_total=n_total,
        n_scored=n_scored,
        n_undefined=n_undefined,
        n_infinite=n_infinite,
        primary_mean=primary_mean,
    )


def _undefined_reason(truth: Hashable, distribution: Distribution) -> str:
    """Build a human-readable reason for an undefined (unidentifiable) subject."""
    return (
        f"truth {truth!r} is not listed in an open distribution carrying residual "
        f"mass {distribution.residual_mass!r}; its probability is not identifiable"
    )
