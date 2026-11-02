"""Descriptive classification diagnostics (not proper scoring rules).

The quantities in this module *describe* a predicted distribution or its
relationship to the truth, but they are **not** proper scoring rules. In
particular:

*   **Accuracy** and **rank** depend only on the ordering of probabilities, not
    their magnitudes, so they cannot detect miscalibration.
*   **Entropy** measures the *spread* (uncertainty) of a distribution. It is a
    property of the prediction alone and does **not** involve the truth. A
    confident wrong prediction has low entropy; entropy must never be
    interpreted as a measure of correctness.

For quality assessment use the proper scoring rules in
:mod:`immunomath.scoring`.

Tie handling
------------
Evaluation statistics are invariant to candidate ordering:

*   :func:`top_k_accuracy` awards **fractional credit** when ties straddle the
    top-k boundary, so the result does not depend on how ties are broken.
*   :func:`rank_of_truth` reports both the **minimum** and **maximum** rank
    consistent with the ties, rather than committing to an arbitrary order.

A canonical sort key may be supplied for deterministic *presentation*, but it
does not change these statistics.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

from .distributions import Distribution

__all__ = [
    "RankOfTruth",
    "effective_number_of_outcomes",
    "entropy",
    "num_candidates",
    "probability_of_truth",
    "rank_of_truth",
    "top_k_accuracy",
]


def probability_of_truth(truth: Hashable, distribution: Distribution) -> float:
    """Return the probability assigned to the true outcome.

    This is closed-world lookup: a thin, well-named wrapper around
    :meth:`Distribution.probability`. For a *complete* distribution an unlisted
    truth returns ``0.0``; for an *open* distribution an unlisted truth raises
    :class:`KeyError` because its probability is unidentifiable.

    Parameters
    ----------
    truth : Hashable
        The observed true outcome.
    distribution : Distribution
        The predicted distribution.

    Returns
    -------
    float
        The probability assigned to ``truth``.

    Examples
    --------
    >>> from immunomath import Distribution, probability_of_truth
    >>> probability_of_truth("G1", Distribution({"G1": 0.8, "G2": 0.2}))
    0.8
    """
    return distribution.probability(truth)


def top_k_accuracy(
    truth: Hashable,
    distribution: Distribution,
    k: int = 1,
) -> float:
    r"""Return top-:math:`k` accuracy with order-invariant fractional ties.

    The truth is "in the top :math:`k`" if it is among the :math:`k`
    highest-probability outcomes. When ties at the probability threshold make
    the membership ambiguous, fractional credit is awarded so the result does
    **not** depend on an arbitrary tie-break order.

    Concretely, let :math:`p^\*` be the probability of the truth. Let
    :math:`s` be the number of outcomes with probability strictly greater than
    :math:`p^\*` ("sure winners") and :math:`t` be the number of outcomes
    (including the truth) tied at exactly :math:`p^\*`. Then:

    *   if :math:`s \ge k`, the truth cannot be in the top :math:`k`: credit
        ``0.0``;
    *   if :math:`s + t \le k`, the whole tie group fits: credit ``1.0``;
    *   otherwise :math:`k - s` of the :math:`t` tied slots are filled at
        random, and the truth's expected inclusion is
        :math:`(k - s) / t`.

    Parameters
    ----------
    truth : Hashable
        The observed true outcome.
    distribution : Distribution
        The predicted distribution.
    k : int, optional
        The cutoff rank, a positive integer. Defaults to ``1`` (top-1 accuracy).

    Returns
    -------
    float
        Fractional accuracy in ``[0, 1]``. Exactly ``0.0`` or ``1.0`` when
        there is no boundary tie.

    Raises
    ------
    ValueError
        If ``k`` is not a positive integer.
    KeyError
        If the truth is unidentifiable in an open distribution.

    Examples
    --------
    Unambiguous top-1 hit:

    >>> from immunomath import Distribution, top_k_accuracy
    >>> top_k_accuracy("G1", Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1}), k=1)
    1.0

    A two-way tie for the top: each tied outcome gets 1/2 credit at k=1,
    regardless of input order:

    >>> top_k_accuracy("G1", Distribution({"G1": 0.5, "G2": 0.5}), k=1)
    0.5
    >>> top_k_accuracy("G2", Distribution({"G1": 0.5, "G2": 0.5}), k=1)
    0.5
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    p_star = distribution.probability(truth)
    # A zero-probability truth is never "predicted"; it cannot be in the top-k.
    if p_star <= 0.0:
        return 0.0

    strictly_greater = sum(1 for p in distribution.probabilities if p > p_star)
    tied = sum(1 for p in distribution.probabilities if p == p_star)
    # The truth itself must be among the explicitly listed tied outcomes.
    # (p_star > 0 guarantees truth is listed.)

    if strictly_greater >= k:
        return 0.0
    if strictly_greater + tied <= k:
        return 1.0
    return (k - strictly_greater) / tied


@dataclass(frozen=True)
class RankOfTruth:
    """The rank of the true outcome, reported as a tie-robust interval.

    Ranks are 1-based: rank 1 is the highest-probability outcome. When several
    outcomes tie at the truth's probability, the truth's rank is ambiguous;
    ``min_rank`` and ``max_rank`` bracket the possible values. They are equal
    when there is no tie.

    Parameters
    ----------
    min_rank : int
        The best (smallest) rank consistent with the ties.
    max_rank : int
        The worst (largest) rank consistent with the ties.
    """

    min_rank: int
    max_rank: int

    @property
    def is_tied(self) -> bool:
        """Whether the truth is tied with at least one other outcome."""
        return self.min_rank != self.max_rank


def rank_of_truth(truth: Hashable, distribution: Distribution) -> RankOfTruth:
    r"""Return the 1-based rank of the truth as a ``(min, max)`` interval.

    With :math:`s` outcomes strictly more probable than the truth and :math:`t`
    outcomes (including the truth) tied at its probability, the truth occupies
    some rank in :math:`[\,s + 1,\; s + t\,]`. The interval endpoints are
    returned so the statistic is invariant to tie-break order.

    Parameters
    ----------
    truth : Hashable
        The observed true outcome.
    distribution : Distribution
        The predicted distribution.

    Returns
    -------
    RankOfTruth
        The minimum and maximum rank consistent with the ties.

    Raises
    ------
    KeyError
        If the truth is unidentifiable in an open distribution.
    ValueError
        If the truth has zero probability (its rank among the support is
        undefined); callers should treat this as "not ranked".

    Examples
    --------
    >>> from immunomath import Distribution, rank_of_truth
    >>> r = rank_of_truth("G2", Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1}))
    >>> (r.min_rank, r.max_rank)
    (2, 2)
    >>> r = rank_of_truth("G1", Distribution({"G1": 0.5, "G2": 0.5}))
    >>> (r.min_rank, r.max_rank, r.is_tied)
    (1, 2, True)
    """
    p_star = distribution.probability(truth)
    if p_star <= 0.0:
        raise ValueError(
            f"Truth {truth!r} has zero probability; its rank is undefined. "
            "Treat this as 'not ranked' in the support."
        )
    strictly_greater = sum(1 for p in distribution.probabilities if p > p_star)
    tied = sum(1 for p in distribution.probabilities if p == p_star)
    return RankOfTruth(min_rank=strictly_greater + 1, max_rank=strictly_greater + tied)


def entropy(distribution: Distribution, *, base: float | None = None) -> float:
    r"""Return the Shannon entropy of the predicted distribution.

    .. math::

        H(P) = -\sum_{k} p_k \ln p_k ,

    with the convention :math:`0 \ln 0 = 0`. By default the natural logarithm is
    used (entropy in *nats*); pass ``base`` for another unit (e.g. ``base=2`` for
    bits).

    .. warning::

        Entropy is **not** a measure of correctness. It describes only the
        spread of the prediction and does not involve the truth. A confidently
        wrong prediction has *low* entropy. Use the proper scoring rules in
        :mod:`immunomath.scoring` to assess forecast quality.

    Residual mass, if present, is treated as a single additional block of
    probability contributing :math:`-m \ln m` for residual mass :math:`m`.

    Parameters
    ----------
    distribution : Distribution
        The predicted distribution.
    base : float, optional
        The logarithm base. ``None`` (default) means natural log (nats).

    Returns
    -------
    float
        The entropy, in nats (or the requested base's unit).

    Examples
    --------
    A point mass has zero entropy:

    >>> from immunomath import Distribution, entropy
    >>> entropy(Distribution({"G1": 1.0}))
    0.0

    A uniform distribution over ``n`` outcomes has entropy ``ln n`` nats:

    >>> import math
    >>> round(entropy(Distribution({"A": 0.5, "B": 0.5})), 10) == round(math.log(2), 10)
    True
    """
    masses = list(distribution.probabilities)
    if not distribution.is_complete:
        masses.append(distribution.residual_mass)
    total = math.fsum(-p * math.log(p) for p in masses if p > 0.0)
    if base is not None:
        total /= math.log(base)
    return total


def effective_number_of_outcomes(distribution: Distribution, *, base: float | None = None) -> float:
    r"""Return the effective number of outcomes, :math:`\exp(H)`.

    This is the exponential of the Shannon entropy (the *perplexity*):

    .. math::

        N_\mathrm{eff} = \exp\bigl(H(P)\bigr),

    using natural-log entropy. It equals the number of outcomes of a uniform
    distribution with the same entropy, giving an interpretable "effective
    candidate count". A point mass has :math:`N_\mathrm{eff} = 1`; a uniform
    distribution over :math:`n` outcomes has :math:`N_\mathrm{eff} = n`.

    Like entropy, this is a descriptive property of the prediction and says
    nothing about correctness.

    Parameters
    ----------
    distribution : Distribution
        The predicted distribution.
    base : float, optional
        If given, entropy is computed in this base and the effective number is
        ``base ** H``. The result is mathematically identical to the default.

    Returns
    -------
    float
        The effective number of outcomes, :math:`\ge 1`.

    Examples
    --------
    >>> from immunomath import Distribution, effective_number_of_outcomes
    >>> effective_number_of_outcomes(Distribution({"G1": 1.0}))
    1.0
    >>> round(effective_number_of_outcomes(Distribution({"A": 0.5, "B": 0.5})), 10)
    2.0
    """
    if base is None:
        return math.exp(entropy(distribution))
    return float(base ** entropy(distribution, base=base))


def num_candidates(distribution: Distribution, *, positive_only: bool = False) -> int:
    """Return the number of candidate outcomes in the distribution.

    Parameters
    ----------
    distribution : Distribution
        The predicted distribution.
    positive_only : bool, optional
        If ``True``, count only outcomes with strictly positive probability
        (the support size). If ``False`` (default), count all explicitly listed
        outcomes, including those with explicit zero probability. The residual
        block is never counted as a candidate because it enumerates no specific
        outcome.

    Returns
    -------
    int
        The candidate count.

    Examples
    --------
    >>> from immunomath import Distribution, num_candidates
    >>> d = Distribution({"G1": 0.8, "G2": 0.2, "G3": 0.0})
    >>> num_candidates(d)
    3
    >>> num_candidates(d, positive_only=True)
    2
    """
    if positive_only:
        return len(distribution.support)
    return len(distribution)
