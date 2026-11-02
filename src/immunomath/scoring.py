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

from .distributions import Distribution

__all__ = [
    "CohortScore",
    "brier_score",
    "mean_brier_score",
    "mean_negative_log_score",
    "negative_log_score",
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
