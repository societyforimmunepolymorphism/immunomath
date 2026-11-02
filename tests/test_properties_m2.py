"""Property-based tests for Milestone 2 statistics invariants."""

from __future__ import annotations

import itertools
import math

from hypothesis import given
from hypothesis import strategies as st

from immunomath import bootstrap_ci, trimmed_mean
from immunomath.quantiles import QuantileMethod, loss_quantile

finite_samples = st.lists(
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)


@given(finite_samples)
def test_bootstrap_point_estimate_is_mean(sample: list[float]) -> None:
    res = bootstrap_ci(sample, seed=0, n_resamples=100)
    assert math.isclose(
        res.point_estimate, math.fsum(sample) / len(sample), rel_tol=1e-9, abs_tol=1e-9
    )


@given(finite_samples, st.integers(min_value=0, max_value=10_000))
def test_bootstrap_is_deterministic(sample: list[float], seed: int) -> None:
    a = bootstrap_ci(sample, seed=seed, n_resamples=100)
    b = bootstrap_ci(sample, seed=seed, n_resamples=100)
    assert (a.low, a.high, a.point_estimate) == (b.low, b.high, b.point_estimate)


@given(finite_samples)
def test_quantile_monotnon_decreasing_in_q(sample: list[float]) -> None:
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    vals = [loss_quantile(sample, q) for q in qs]
    for lo, hi in itertools.pairwise(vals):
        assert lo <= hi + 1e-9


@given(finite_samples)
def test_quantile_endpoints_are_extremes(sample: list[float]) -> None:
    assert loss_quantile(sample, 0.0) == min(sample)
    assert loss_quantile(sample, 1.0) == max(sample)


@given(finite_samples)
def test_trimmed_mean_le_mean_for_upper_trim(sample: list[float]) -> None:
    # Removing the largest values cannot increase the mean.
    mean = math.fsum(sample) / len(sample)
    trimmed = trimmed_mean(sample, upper_fraction=0.1)
    assert trimmed <= mean + 1e-9


@given(finite_samples)
def test_quantile_within_sample_range(sample: list[float]) -> None:
    for q in (0.0, 0.3, 0.6, 1.0):
        v = loss_quantile(sample, q, method=QuantileMethod.LINEAR)
        assert min(sample) - 1e-9 <= v <= max(sample) + 1e-9
