"""Quantiles of a loss distribution, with explicit infinite-value handling.

A cohort of per-subject losses induces an empirical loss *distribution*. Its
quantiles summarize the spread of forecast penalties: the typical (median)
penalty, the upper tail where catastrophic predictions live, and so on.

Quantile convention
-------------------
The default is **Hyndman-Fan type 7** (``"linear"``), the piecewise-linear
interpolation used by NumPy, R, pandas, and Julia by default. For a sorted
sample ``x_(1) <= ... <= x_(n)`` and probability ``q``, type 7 places the
``k``-th order statistic at plotting position ``(k-1)/(n-1)`` and linearly
interpolates between the two bracketing order statistics. Alternatives from the
Hyndman-Fan (1996) taxonomy are exposed through :class:`QuantileMethod` and map
onto NumPy's ``method=`` argument, so the reported quantile is never ambiguous
about which estimator produced it.

Infinite losses
---------------
The logarithmic score assigns ``+inf`` to an identifiable-zero truth. Such
infinities are real data, not errors, and are handled explicitly: infinities
sort to the top of the sample; a quantile whose interpolation bracket reaches
the infinite tail is returned as ``+inf`` exactly (never ``inf - inf = nan``);
and NaN inputs are rejected rather than silently producing NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

import numpy as np

__all__ = [
    "QuantileMethod",
    "loss_quantile",
    "loss_quantiles",
]


class QuantileMethod(StrEnum):
    """Quantile-estimator conventions, mapped onto NumPy ``method=`` names.

    ``LINEAR`` is Hyndman-Fan type 7 (the default). ``WEIBULL`` is type 6,
    ``MEDIAN_UNBIASED`` is type 8. The discontinuous rules (``LOWER``,
    ``HIGHER``, ``NEAREST``) return sample values without interpolation.
    """

    LINEAR = "linear"  # Hyndman-Fan type 7 (default)
    WEIBULL = "weibull"  # type 6
    MEDIAN_UNBIASED = "median_unbiased"  # type 8
    LOWER = "lower"
    HIGHER = "higher"
    NEAREST = "nearest"


def _validate_and_sort(losses: Sequence[float]) -> np.ndarray:
    """Return a sorted float array, rejecting empty input and NaNs."""
    if len(losses) == 0:
        raise ValueError("Cannot compute a quantile of an empty loss sample.")
    arr = np.asarray(losses, dtype=float)
    if np.isnan(arr).any():
        n_nan = int(np.isnan(arr).sum())
        raise ValueError(
            f"Loss sample contains {n_nan} NaN value(s). A NaN loss is a defect, "
            "not a legitimate value to summarize; fix the upstream scoring rather "
            "than quantiling over NaN."
        )
    return np.sort(arr)  # +inf sorts to the top, which is correct for losses


def _quantile_with_inf(sorted_losses: np.ndarray, q: float, method: QuantileMethod) -> float:
    """Compute one quantile on pre-sorted losses with explicit +inf handling."""
    n = sorted_losses.size
    n_inf = int(np.isinf(sorted_losses).sum())
    n_finite = n - n_inf

    if n_inf == 0:
        return float(np.quantile(sorted_losses, q, method=method.value))
    if n_finite == 0:
        return math.inf

    positions = np.arange(n, dtype=float)
    h = float(np.quantile(positions, q, method=method.value))
    first_inf_index = n_finite

    lo = math.floor(h)
    frac = h - lo
    # The result is +inf whenever the interpolation bracket reaches the infinite
    # tail. With no interpolation (frac == 0) the bracket is the single order
    # statistic ``lo``; with interpolation it is ``[lo, lo + 1]``.
    upper_index = lo if frac == 0.0 else lo + 1
    if upper_index >= first_inf_index:
        return math.inf

    lo_val = float(sorted_losses[lo])
    if frac == 0.0:
        return lo_val
    hi_val = float(sorted_losses[lo + 1])
    return lo_val + frac * (hi_val - lo_val)


def loss_quantile(
    losses: Sequence[float],
    q: float,
    *,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> float:
    """Return a single quantile of a loss sample (Hyndman-Fan type 7 default).

    Parameters
    ----------
    losses : sequence of float
        Per-subject losses. May contain ``math.inf``; must not contain NaN.
    q : float
        The probability level in ``[0, 1]``.
    method : QuantileMethod, optional
        The estimator convention; defaults to Hyndman-Fan type 7.

    Returns
    -------
    float
        The quantile. Returns ``math.inf`` when the level falls within the
        infinite upper tail; never returns an unexplained ``nan``.

    Raises
    ------
    ValueError
        If ``losses`` is empty, contains NaN, or ``q`` is outside ``[0, 1]``.

    Examples
    --------
    >>> from immunomath.quantiles import loss_quantile
    >>> loss_quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.5)
    0.3
    >>> import math
    >>> loss_quantile([0.1, 0.2, math.inf], 0.9)
    inf
    >>> loss_quantile([0.1, 0.2, math.inf], 0.25)
    0.15000000000000002
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile level q must lie in [0, 1], got {q!r}.")
    return _quantile_with_inf(_validate_and_sort(losses), q, method)


def loss_quantiles(
    losses: Sequence[float],
    qs: Sequence[float],
    *,
    method: QuantileMethod = QuantileMethod.LINEAR,
) -> tuple[float, ...]:
    """Return several quantiles of a loss sample, sharing one sort.

    Parameters
    ----------
    losses : sequence of float
        Per-subject losses. May contain ``math.inf``; must not contain NaN.
    qs : sequence of float
        The probability levels, each in ``[0, 1]``.
    method : QuantileMethod, optional
        The estimator convention; defaults to Hyndman-Fan type 7.

    Returns
    -------
    tuple of float
        One quantile per level in ``qs``, in order.

    Raises
    ------
    ValueError
        If ``losses`` is empty, contains NaN, or any level is outside ``[0, 1]``.

    Examples
    --------
    >>> from immunomath.quantiles import loss_quantiles
    >>> loss_quantiles([0.1, 0.2, 0.3, 0.4, 0.5], [0.25, 0.5, 0.75])
    (0.2, 0.3, 0.4)
    """
    for q in qs:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"Quantile level q must lie in [0, 1], got {q!r}.")
    sorted_losses = _validate_and_sort(losses)
    return tuple(_quantile_with_inf(sorted_losses, q, method) for q in qs)
