"""Property-based tests for Milestone 4 invariants."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from immunomath import Cohort, Distribution, risk_coverage, top_label_calibration
from immunomath.calibration import wilson_interval


@st.composite
def cohorts(draw: st.DrawFn) -> Cohort:
    n = draw(st.integers(min_value=1, max_value=25))
    rows = []
    truths = []
    for _ in range(n):
        p = draw(st.floats(min_value=0.01, max_value=0.99))
        rows.append(Distribution({"G1": p, "G2": 1.0 - p}))
        truths.append(draw(st.sampled_from(["G1", "G2"])))
    return Cohort([f"s{i}" for i in range(n)], truths, rows)


@settings(max_examples=60)
@given(
    cohorts(),
    st.sampled_from(["fixed_width", "equal_frequency"]),
    st.integers(min_value=1, max_value=12),
)
def test_calibration_invariants(cohort, binning, n_bins) -> None:
    r = top_label_calibration(cohort, n_bins=n_bins, binning=binning)
    assert sum(b.count for b in r.bins) == r.n_evaluated == len(cohort)
    assert 0.0 <= r.ece <= 1.0 + 1e-12
    assert r.ece <= r.mce + 1e-12  # a weighted mean cannot exceed the maximum
    for b in r.bins:
        if b.count:
            assert b.ci_low - 1e-12 <= b.observed_rate <= b.ci_high + 1e-12
            assert 0.0 <= b.ci_low <= b.ci_high <= 1.0


@settings(max_examples=60)
@given(cohorts())
def test_equal_frequency_bins_are_balanced(cohort) -> None:
    r = top_label_calibration(cohort, n_bins=4, binning="equal_frequency")
    counts = [b.count for b in r.bins]
    assert max(counts) - min(counts) <= 1


@settings(max_examples=60)
@given(cohorts())
def test_risk_coverage_invariants(cohort) -> None:
    result = risk_coverage(cohort)
    coverage = [p.coverage for p in result.points]
    assert coverage == sorted(coverage)
    assert math.isclose(coverage[-1], 1.0)
    for p in result.points:
        assert 0.0 <= p.accuracy <= 1.0
        assert math.isclose(p.accuracy + p.error_rate, 1.0, abs_tol=1e-12)
    assert result.full_coverage is not None
    assert result.points[-1].accuracy == result.full_coverage.accuracy


@settings(max_examples=60)
@given(cohorts(), st.randoms(use_true_random=False))
def test_risk_coverage_order_invariant(cohort, rnd) -> None:
    order = list(range(len(cohort)))
    rnd.shuffle(order)
    shuffled = Cohort(
        [cohort.subject_ids[i] for i in order],
        [cohort.truths[i] for i in order],
        [cohort.distributions[i] for i in order],
    )
    a, b = risk_coverage(cohort).points, risk_coverage(shuffled).points
    assert len(a) == len(b)
    for pa, pb in zip(a, b, strict=True):
        assert pa.n_retained == pb.n_retained
        assert math.isclose(pa.accuracy, pb.accuracy, abs_tol=1e-12)


@given(st.integers(min_value=1, max_value=500), st.data())
def test_wilson_contains_point_estimate(n, data) -> None:
    s = data.draw(st.integers(min_value=0, max_value=n))
    low, high = wilson_interval(s, n)
    assert 0.0 <= low <= s / n <= high <= 1.0
