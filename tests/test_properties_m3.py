"""Property-based tests for paired comparison invariants (Milestone 3)."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from immunomath import Cohort, Distribution, compare_cohorts


@st.composite
def binary_cohorts(draw: st.DrawFn):
    """Draw two cohorts over the same subjects and outcome space {G1, G2}."""
    n = draw(st.integers(min_value=1, max_value=8))
    ids = [f"s{i}" for i in range(n)]
    truths = [draw(st.sampled_from(["G1", "G2"])) for _ in range(n)]

    def draw_rows() -> list[Distribution]:
        rows = []
        for _ in range(n):
            p = draw(st.floats(min_value=0.02, max_value=0.98))
            rows.append(Distribution({"G1": p, "G2": 1.0 - p}))
        return rows

    a = Cohort(ids, truths, draw_rows())
    b = Cohort(ids, truths, draw_rows())
    return a, b


@given(binary_cohorts())
def test_swapping_arms_negates_mean_diff(cohorts) -> None:
    a, b = cohorts
    ab = compare_cohorts(a, b, seed=0, n_resamples=100)
    ba = compare_cohorts(b, a, seed=0, n_resamples=100)
    assert math.isclose(ab.mean_log_loss_diff, -ba.mean_log_loss_diff, abs_tol=1e-9)


@given(binary_cohorts())
def test_swapping_arms_swaps_improved_worsened(cohorts) -> None:
    a, b = cohorts
    ab = compare_cohorts(a, b, seed=0, n_resamples=100)
    ba = compare_cohorts(b, a, seed=0, n_resamples=100)
    assert ab.n_improved == ba.n_worsened
    assert ab.n_worsened == ba.n_improved
    assert ab.n_tied == ba.n_tied


@given(binary_cohorts())
def test_improved_worsened_tied_sum_to_paired(cohorts) -> None:
    a, b = cohorts
    cmp = compare_cohorts(a, b, seed=0, n_resamples=100)
    assert cmp.n_improved + cmp.n_worsened + cmp.n_tied == cmp.n_paired


@given(binary_cohorts())
def test_self_comparison_is_zero(cohorts) -> None:
    a, _ = cohorts
    cmp = compare_cohorts(a, a, seed=0, n_resamples=100)
    assert cmp.mean_log_loss_diff == 0.0
    assert cmp.n_improved == 0
    assert cmp.n_worsened == 0
    assert cmp.concordance_diff_top1 == 0.0
