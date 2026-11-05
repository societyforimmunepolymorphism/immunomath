r"""Optional adapter: typing-resolution score (TRS) → observation weights.

.. warning::

    This module is an **optional adapter**, not part of the statistical kernel.
    It encodes one specific, domain-motivated choice of weighting function. The
    kernel (:mod:`immunomath.weighting`) consumes plain positive weights and is
    unaware of TRS, HLA, or any biological covariate. Nothing here performs HLA
    parsing, TRS *calculation*, or haplotype-frequency estimation; TRS is
    computed elsewhere and supplied as a normalized number in ``[0, 1]``.

The weighting function
----------------------
For a normalized typing-resolution score :math:`r_i \in [0, 1]` (higher means
better resolved),

.. math::

    w_i = 1 + \lambda\,(1 - r_i)^{\gamma},
    \qquad \lambda \ge 0,\ \gamma > 0.

* :math:`\lambda = 0` makes every weight ``1`` (uniform; recovers ordinary
  scoring exactly).
* Larger :math:`\lambda` upweights poorly resolved observations more.
* :math:`\gamma` controls the curvature: :math:`\gamma > 1` concentrates the
  extra weight on the *lowest*-resolution observations.

The parameters are **explicit and mandatory** — this adapter imposes no
"scientific default" :math:`\lambda` or :math:`\gamma`. Choose and pre-register
them before looking at predictions.

Why this preserves conditional propriety
-----------------------------------------
The weight depends only on :math:`r_i`, a covariate fixed by the observed typing
resolution. It does **not** depend on the true genotype, on whether imputation
succeeded, or on any model's posterior. It is therefore a positive constant with
respect to the forecast, and scaling a strictly proper loss by a positive
constant leaves honest reporting as the unique minimizer. See the
:mod:`immunomath.weighting` module docstring for the full argument.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "trs_weights",
]


def trs_weights(
    trs: Sequence[float],
    *,
    lam: float,
    gamma: float,
) -> tuple[float, ...]:
    r"""Map normalized TRS values to positive observation weights.

    Implements :math:`w_i = 1 + \lambda (1 - r_i)^{\gamma}` for
    :math:`r_i \in [0, 1]`, :math:`\lambda \ge 0`, :math:`\gamma > 0`.

    The TRS scale is **not** silently normalized: values must already lie in
    ``[0, 1]`` (higher = better resolved). Supplying raw, unnormalized scores is
    an error rather than a quietly rescaled convenience, because the correct
    normalization is domain knowledge the kernel cannot infer.

    Parameters
    ----------
    trs : sequence of float
        Normalized typing-resolution scores in ``[0, 1]``.
    lam : float
        Upweighting strength :math:`\lambda \ge 0`. ``0`` yields all-ones
        weights (uniform).
    gamma : float
        Curvature exponent :math:`\gamma > 0`.

    Returns
    -------
    tuple of float
        One strictly-positive weight per observation (``>= 1``).

    Raises
    ------
    ValueError
        If ``trs`` is empty, any TRS value is non-finite or outside ``[0, 1]``,
        ``lam < 0``, or ``gamma <= 0``.

    Examples
    --------
    >>> from immunomath.trs import trs_weights
    >>> trs_weights([1.0, 0.5, 0.0], lam=4.0, gamma=2.0)
    (1.0, 2.0, 5.0)
    >>> trs_weights([0.2, 0.8], lam=0.0, gamma=2.0)  # lambda=0 -> uniform
    (1.0, 1.0)
    """
    if lam < 0.0:
        raise ValueError(f"lam (lambda) must be >= 0, got {lam!r}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}.")
    if len(trs) == 0:
        raise ValueError("trs must contain at least one value.")

    weights: list[float] = []
    for i, r in enumerate(trs):
        rf = float(r)
        if math.isnan(rf) or math.isinf(rf):
            raise ValueError(f"trs[{i}] is non-finite; TRS values must be finite.")
        if not (0.0 <= rf <= 1.0):
            raise ValueError(
                f"trs[{i}] = {rf!r} is outside [0, 1]. Normalize TRS to [0, 1] "
                "explicitly before calling; this adapter does not guess the scale."
            )
        weights.append(1.0 + lam * (1.0 - rf) ** gamma)
    return tuple(weights)
