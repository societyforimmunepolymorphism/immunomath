"""Property-based tests for fundamental probability identities.

These tests assert mathematical invariants that must hold for *any* valid
distribution, rather than re-checking hand-computed outputs. They guard against
implementation drift that happens to pass example-based tests.
"""

from __future__ import annotations

import math

from hypothesis import assume, given
from hypothesis import strategies as st

from immunomath import (
    Distribution,
    brier_score,
    effective_number_of_outcomes,
    entropy,
    negative_log_score,
    top_k_accuracy,
)


@st.composite
def simplex(draw: st.DrawFn, min_size: int = 1, max_size: int = 8) -> dict[str, float]:
    """Draw a normalized probability vector over string outcomes."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    weights = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    total = math.fsum(weights)
    assume(total > 0.0)
    probs = [w / total for w in weights]
    # Fix residual float error onto the largest entry so it sums to exactly 1.
    drift = 1.0 - math.fsum(probs)
    imax = max(range(n), key=lambda i: probs[i])
    probs[imax] += drift
    assume(all(p >= 0.0 for p in probs))
    return {f"G{i}": p for i, p in enumerate(probs)}


@given(simplex())
def test_probabilities_sum_to_one(probs: dict[str, float]) -> None:
    d = Distribution(probs, atol=1e-6)
    assert math.isclose(math.fsum(d.probabilities), 1.0, abs_tol=1e-6)


@given(simplex())
def test_brier_in_range(probs: dict[str, float]) -> None:
    d = Distribution(probs, atol=1e-6)
    for truth in d.outcomes:
        b = brier_score(truth, d)
        assert -1e-12 <= b <= 2.0 + 1e-12


@given(simplex())
def test_log_score_nonnegative(probs: dict[str, float]) -> None:
    d = Distribution(probs, atol=1e-6)
    for truth in d.outcomes:
        loss = negative_log_score(truth, d)
        # -ln p >= 0 since p <= 1.
        assert loss >= -1e-12


@given(simplex())
def test_entropy_bounds(probs: dict[str, float]) -> None:
    d = Distribution(probs, atol=1e-6)
    h = entropy(d)
    n = len(d)
    # 0 <= H <= ln n
    assert -1e-9 <= h <= math.log(n) + 1e-9


@given(simplex())
def test_effective_number_bounds(probs: dict[str, float]) -> None:
    d = Distribution(probs, atol=1e-6)
    neff = effective_number_of_outcomes(d)
    assert 1.0 - 1e-9 <= neff <= len(d) + 1e-9


@given(simplex())
def test_topk_accuracy_sums_to_min_k_n(probs: dict[str, float]) -> None:
    """Summed over all truths, fractional top-k credit equals min(k, n).

    This is the key order-invariance identity: exactly k slots are filled
    (or n if n < k), and the fractional credit is distributed among outcomes
    so it sums to that number. The identity holds over the *positive support*,
    since a zero-probability outcome is never counted as "predicted".
    """
    d = Distribution(probs, atol=1e-6)
    positive = [o for o in d.outcomes if d[o] > 0.0]
    n = len(positive)
    assume(n >= 1)
    for k in range(1, n + 1):
        total = math.fsum(top_k_accuracy(o, d, k=k) for o in positive)
        assert math.isclose(total, min(k, n), abs_tol=1e-9)


@st.composite
def projection_map(draw: st.DrawFn, outcomes: list[str]) -> dict[str, str]:
    """Draw a total projection mapping each outcome to a coarse class label."""
    n_classes = draw(st.integers(min_value=1, max_value=max(1, len(outcomes))))
    classes = [f"C{j}" for j in range(n_classes)]
    return {o: draw(st.sampled_from(classes)) for o in outcomes}


@given(simplex(min_size=1, max_size=6), st.data())
def test_projection_preserves_mass(probs: dict[str, float], data: st.DataObject) -> None:
    """A total projection preserves total probability mass exactly."""
    d = Distribution(probs, atol=1e-6)
    mapping = data.draw(projection_map(list(d.outcomes)))
    coarse = d.project(lambda o: mapping[o])
    assert math.isclose(math.fsum(coarse.probabilities) + coarse.residual_mass, 1.0, abs_tol=1e-6)
    # Total projection: no residual introduced beyond input (which had none).
    assert coarse.is_complete


@given(simplex(min_size=2, max_size=6))
def test_entropy_invariant_to_permutation(probs: dict[str, float]) -> None:
    items = list(probs.items())
    d1 = Distribution(dict(items), atol=1e-6)
    d2 = Distribution(dict(reversed(items)), atol=1e-6)
    assert math.isclose(entropy(d1), entropy(d2), abs_tol=1e-12)
