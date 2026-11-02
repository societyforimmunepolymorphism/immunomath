"""Tests for paired comparative assessment (Milestone 3)."""

from __future__ import annotations

import math

import pytest

from immunomath import Cohort, Distribution, Residual, compare_cohorts

D = Distribution


def cohort(ids, truths, rows):
    return Cohort(ids, truths, [D(r) for r in rows])


class TestPairedDifferenceCorrectness:
    def test_sign_convention_negative_means_a_better(self) -> None:
        a = cohort(["s1"], ["G1"], [{"G1": 0.8, "G2": 0.2}])
        b = cohort(["s1"], ["G1"], [{"G1": 0.6, "G2": 0.4}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        # L_A = -ln 0.8, L_B = -ln 0.6, d = L_A - L_B < 0 (A better).
        expected = (-math.log(0.8)) - (-math.log(0.6))
        assert cmp.mean_log_loss_diff == pytest.approx(expected)
        assert cmp.mean_log_loss_diff < 0

    def test_subject_differences_recorded(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        b = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert len(cmp.subject_differences) == 2
        assert {sid for sid, _ in cmp.subject_differences} == {"s1", "s2"}


class TestBothHeadlines:
    def test_counts_and_proportions(self) -> None:
        # s1: A better; s2: B better; s3: tie
        a = cohort(
            ["s1", "s2", "s3"],
            ["G1", "G1", "G1"],
            [{"G1": 0.9, "G2": 0.1}, {"G1": 0.4, "G2": 0.6}, {"G1": 0.5, "G2": 0.5}],
        )
        b = cohort(
            ["s1", "s2", "s3"],
            ["G1", "G1", "G1"],
            [{"G1": 0.5, "G2": 0.5}, {"G1": 0.8, "G2": 0.2}, {"G1": 0.5, "G2": 0.5}],
        )
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert cmp.n_improved == 1
        assert cmp.n_worsened == 1
        assert cmp.n_tied == 1
        assert cmp.proportion_improved == pytest.approx(1 / 3)
        assert cmp.proportion_worsened == pytest.approx(1 / 3)
        assert cmp.proportion_tied == pytest.approx(1 / 3)

    def test_mean_difference_has_ci(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        b = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=500)
        ci = cmp.mean_log_loss_diff_ci
        assert ci is not None
        assert ci.low <= cmp.mean_log_loss_diff <= ci.high

    def test_largest_changes(self) -> None:
        a = cohort(
            ["s1", "s2"],
            ["G1", "G1"],
            [{"G1": 0.99, "G2": 0.01}, {"G1": 0.2, "G2": 0.8}],
        )
        b = cohort(
            ["s1", "s2"],
            ["G1", "G1"],
            [{"G1": 0.5, "G2": 0.5}, {"G1": 0.9, "G2": 0.1}],
        )
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        # s1 improved (A much better), s2 worsened (A much worse).
        assert cmp.largest_improvement is not None
        assert cmp.largest_improvement[0] == "s1"
        assert cmp.largest_worsening is not None
        assert cmp.largest_worsening[0] == "s2"

    def test_no_worsening_gives_none(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G1"], [{"G1": 0.9, "G2": 0.1}, {"G1": 0.8, "G2": 0.2}])
        b = cohort(["s1", "s2"], ["G1", "G1"], [{"G1": 0.5, "G2": 0.5}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert cmp.n_worsened == 0
        assert cmp.largest_worsening is None


class TestConcordanceDifference:
    def test_top1_concordance_diff(self) -> None:
        # A: both top-1 correct (acc 1.0). B: s1 correct, s2 tie (0.5) -> acc 0.75.
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        b = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert cmp.concordance_diff_top1 == pytest.approx(0.25)


class TestSubjectOverlap:
    def test_exact_requires_identical_sets(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        b = cohort(["s1", "s9"], ["G1", "G2"], [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}])
        with pytest.raises(ValueError, match="same subjects"):
            compare_cohorts(a, b)

    def test_intersection_opt_in(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        b = cohort(["s1", "s9"], ["G1", "G2"], [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, on="intersection", seed=0, n_resamples=200)
        assert cmp.n_common == 1
        assert cmp.n_paired == 1

    def test_no_common_subjects_raises(self) -> None:
        a = cohort(["s1"], ["G1"], [{"G1": 0.8, "G2": 0.2}])
        b = cohort(["s9"], ["G1"], [{"G1": 0.6, "G2": 0.4}])
        with pytest.raises(ValueError, match="share no subjects"):
            compare_cohorts(a, b, on="intersection")

    def test_truth_mismatch_raises(self) -> None:
        a = cohort(["s1"], ["G1"], [{"G1": 0.8, "G2": 0.2}])
        b = cohort(["s1"], ["G2"], [{"G1": 0.6, "G2": 0.4}])
        with pytest.raises(ValueError, match="different truths"):
            compare_cohorts(a, b)

    def test_invalid_on(self) -> None:
        a = cohort(["s1"], ["G1"], [{"G1": 0.8, "G2": 0.2}])
        b = cohort(["s1"], ["G1"], [{"G1": 0.6, "G2": 0.4}])
        with pytest.raises(ValueError, match="on must be"):
            compare_cohorts(a, b, on="bogus")  # type: ignore[arg-type]


class TestStrictOutcomeSpace:
    def test_mismatched_spaces_raise(self) -> None:
        a = cohort(["s1"], ["A*01:01"], [{"A*01:01": 0.7, "A*01:02": 0.3}])
        b = cohort(["s1"], ["A*01"], [{"A*01": 0.7, "A*02": 0.3}])
        with pytest.raises(ValueError, match="different outcome spaces"):
            compare_cohorts(a, b)

    def test_projection_reconciles(self) -> None:
        a = cohort(
            ["s1", "s2"],
            ["A*01:01", "A*02:01"],
            [{"A*01:01": 0.7, "A*02:01": 0.3}, {"A*01:01": 0.4, "A*02:01": 0.6}],
        )
        b = cohort(
            ["s1", "s2"],
            ["A*01:01", "A*02:01"],
            [{"A*01:01": 0.5, "A*02:01": 0.5}, {"A*01:01": 0.5, "A*02:01": 0.5}],
        )
        cmp = compare_cohorts(a, b, projection=lambda x: x.split(":")[0], seed=0, n_resamples=200)
        assert cmp.n_paired == 2
        assert math.isfinite(cmp.mean_log_loss_diff)

    def test_projection_dropping_truth_raises(self) -> None:
        a = cohort(["s1"], ["A*01:01"], [{"A*01:01": 0.6, "A*02:01": 0.4}])
        b = cohort(["s1"], ["A*01:01"], [{"A*01:01": 0.5, "A*02:01": 0.5}])
        # Projection drops the truth's class -> error.
        with pytest.raises(ValueError, match="maps the truth"):
            compare_cohorts(a, b, projection=lambda x: None if x.startswith("A*01") else x)


class TestInfiniteAndUndefined:
    def test_inf_minus_inf_excluded(self) -> None:
        a = cohort(["s1", "s2"], ["G2", "G2"], [{"G1": 1.0, "G2": 0.0}, {"G1": 1.0, "G2": 0.0}])
        b = cohort(["s1", "s2"], ["G2", "G2"], [{"G1": 1.0, "G2": 0.0}, {"G1": 0.5, "G2": 0.5}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert cmp.n_common == 2
        assert cmp.n_paired == 1  # s1 is inf-inf, excluded
        assert cmp.n_both_undefined_or_infinite == 1
        assert "inf - inf" in cmp.exclusion_reasons[0][1]

    def test_inf_minus_finite_retained(self) -> None:
        # s1: A finite, B inf -> d = finite - inf = -inf (A hugely better).
        a = cohort(["s1"], ["G2"], [{"G1": 0.5, "G2": 0.5}])
        b = cohort(["s1"], ["G2"], [{"G1": 1.0, "G2": 0.0}])
        cmp = compare_cohorts(a, b, seed=0, n_resamples=200)
        assert cmp.n_paired == 1
        assert cmp.mean_log_loss_diff == -math.inf
        assert cmp.n_improved == 1

    def test_undefined_in_one_arm_excluded(self) -> None:
        # Same outcome space {G1, G2, Gz}. s1's truth Gz is unlisted under an
        # open residual in arm A -> undefined -> the pair is excluded.
        a3 = Cohort(
            ["s1", "s2"],
            ["Gz", "G1"],
            [
                Distribution({"G1": 0.5, "G2": 0.3, Residual(0.2): None}),
                Distribution({"G1": 0.7, "G2": 0.2, "Gz": 0.1}),
            ],
        )
        b3 = Cohort(
            ["s1", "s2"],
            ["Gz", "G1"],
            [
                Distribution({"G1": 0.4, "G2": 0.3, "Gz": 0.3}),
                Distribution({"G1": 0.6, "G2": 0.2, "Gz": 0.2}),
            ],
        )
        cmp = compare_cohorts(a3, b3, seed=0, n_resamples=200)
        # s1: A's truth Gz unlisted in an open distribution -> undefined -> excluded.
        assert cmp.n_common == 2
        assert cmp.n_paired == 1
        assert cmp.n_both_undefined_or_infinite == 1
        assert "not identifiable" in cmp.exclusion_reasons[0][1]


class TestDeterminism:
    def test_same_seed_same_ci(self) -> None:
        a = cohort(
            ["s1", "s2", "s3"],
            ["G1", "G2", "G1"],
            [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}, {"G1": 0.6, "G2": 0.4}],
        )
        b = cohort(
            ["s1", "s2", "s3"],
            ["G1", "G2", "G1"],
            [{"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}, {"G1": 0.5, "G2": 0.5}],
        )
        c1 = compare_cohorts(a, b, seed=11, n_resamples=500).mean_log_loss_diff_ci
        c2 = compare_cohorts(a, b, seed=11, n_resamples=500).mean_log_loss_diff_ci
        assert c1 is not None and c2 is not None
        assert (c1.low, c1.high) == (c2.low, c2.high)


class TestIdenticalCohorts:
    def test_identical_gives_zero_difference(self) -> None:
        a = cohort(["s1", "s2"], ["G1", "G2"], [{"G1": 0.8, "G2": 0.2}, {"G1": 0.3, "G2": 0.7}])
        cmp = compare_cohorts(a, a, seed=0, n_resamples=200)
        assert cmp.mean_log_loss_diff == pytest.approx(0.0)
        assert cmp.n_improved == 0
        assert cmp.n_worsened == 0
        assert cmp.n_tied == 2
        assert cmp.concordance_diff_top1 == pytest.approx(0.0)
