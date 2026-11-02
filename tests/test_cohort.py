"""Tests for the Cohort container and CohortSummary (Milestone 2)."""

from __future__ import annotations

import math

import pytest

from immunomath import Cohort, Distribution, Residual, ScoreClass

COMPLETE = Distribution({"G1": 0.5, "G2": 0.5})
CONFIDENT = Distribution({"G1": 0.9, "G2": 0.1})
ZERO = Distribution({"G1": 1.0, "G2": 0.0})  # identifiable zero on G2
OPEN = Distribution({"G1": 0.5, Residual(0.5): None})  # G99 unidentifiable


def make_cohort() -> Cohort:
    return Cohort(
        ["S1", "S2", "S3", "S4"],
        ["G1", "G2", "G2", "G99"],
        [CONFIDENT, Distribution({"G1": 0.999, "G2": 0.001}), ZERO, OPEN],
    )


class TestConstruction:
    def test_basic(self) -> None:
        c = Cohort(["S1", "S2"], ["G1", "G2"], [COMPLETE, COMPLETE])
        assert len(c) == 2
        assert c.subject_ids == ("S1", "S2")

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            Cohort(["S1"], ["G1", "G2"], [COMPLETE, COMPLETE])

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Cohort([], [], [])

    def test_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Cohort(["S1", "S1"], ["G1", "G2"], [COMPLETE, COMPLETE])

    def test_non_distribution(self) -> None:
        with pytest.raises(TypeError, match="Distribution"):
            Cohort(["S1"], ["G1"], [{"G1": 1.0}])  # type: ignore[list-item]


class TestClassificationAndDenominators:
    def test_counts(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=200)
        assert s.n_total == 4
        assert s.n_scored == 3  # finite + infinite
        assert s.n_undefined == 1
        assert s.n_infinite == 1
        assert s.n_finite == 2

    def test_primary_mean_includes_infinite(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=200)
        assert s.mean_log_score == math.inf

    def test_finite_only_cohort_mean(self) -> None:
        c = Cohort(["S1", "S2"], ["G1", "G1"], [COMPLETE, COMPLETE])
        s = c.summarize(seed=0, n_resamples=200)
        assert s.mean_log_score == pytest.approx(-math.log(0.5))
        assert s.n_infinite == 0

    def test_undefined_excluded_with_reason(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=200)
        assert len(s.undefined_reasons) == 1
        sid, reason = s.undefined_reasons[0]
        assert sid == "S4"
        assert reason is not None and "not identifiable" in reason

    def test_all_undefined_nan_mean(self) -> None:
        c = Cohort(["S1", "S2"], ["G99", "G98"], [OPEN, OPEN])
        s = c.summarize(seed=0, n_resamples=200)
        assert s.n_scored == 0
        assert math.isnan(s.mean_log_score)


class TestStrictMode:
    def test_strict_rejects_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            make_cohort().summarize(strict=True)

    def test_strict_allows_identifiable(self) -> None:
        c = Cohort(["S1", "S2"], ["G1", "G2"], [COMPLETE, ZERO])
        s = c.summarize(strict=True, seed=0, n_resamples=200)
        assert s.n_undefined == 0
        assert s.mean_log_score == math.inf  # identifiable zero still included


class TestDescriptiveStatistics:
    def test_median_std_max_finite(self) -> None:
        c = Cohort(
            ["A", "B", "C", "D", "E"],
            ["G1"] * 5,
            [Distribution({"G1": p, "G2": 1 - p}) for p in (0.9, 0.8, 0.7, 0.6, 0.5)],
        )
        s = c.summarize(seed=1, n_resamples=500)
        losses = sorted(-math.log(p) for p in (0.9, 0.8, 0.7, 0.6, 0.5))
        assert s.median_log_score == pytest.approx(losses[2])
        assert s.max_log_score == pytest.approx(max(losses))
        assert s.std_log_score > 0

    def test_max_infinite_when_zero_present(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=200)
        assert s.max_log_score == math.inf

    def test_mean_brier_bounded(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=200)
        # Brier is bounded in [0, 2] even when log loss is infinite.
        assert 0.0 <= s.mean_brier <= 2.0

    def test_quantiles_present(self) -> None:
        c = Cohort(
            ["A", "B", "C", "D"],
            ["G1"] * 4,
            [Distribution({"G1": p, "G2": 1 - p}) for p in (0.9, 0.8, 0.7, 0.6)],
        )
        s = c.summarize(quantile_levels=(0.5, 0.9), seed=0, n_resamples=200)
        assert set(s.loss_quantiles) == {0.5, 0.9}


class TestBootstrapIntegration:
    def test_ci_deterministic(self) -> None:
        c = Cohort(
            ["A", "B", "C", "D", "E"],
            ["G1"] * 5,
            [Distribution({"G1": p, "G2": 1 - p}) for p in (0.9, 0.8, 0.7, 0.6, 0.5)],
        )
        a = c.summarize(seed=7, n_resamples=800).mean_log_score_ci
        b = c.summarize(seed=7, n_resamples=800).mean_log_score_ci
        assert a is not None and b is not None
        assert (a.low, a.high) == (b.low, b.high)

    def test_ci_infinite_upper_when_zero_present(self) -> None:
        s = make_cohort().summarize(seed=0, n_resamples=500)
        assert s.mean_log_score_ci is not None
        assert s.mean_log_score_ci.high == math.inf
        assert s.mean_log_score_ci.n_infinite_resamples > 0


class TestSubjectScores:
    def test_classes(self) -> None:
        scores = make_cohort().subject_scores()
        classes = [s.score_class for s in scores]
        assert classes == [
            ScoreClass.FINITE,
            ScoreClass.FINITE,
            ScoreClass.INFINITE,
            ScoreClass.UNDEFINED,
        ]
