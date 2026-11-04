"""Tests for catastrophic-error diagnostics (Milestone 2)."""

from __future__ import annotations

import math

import pytest

from immunomath import Cohort, Distribution, Residual, catastrophic_error_report
from immunomath.catastrophic import DEFAULT_PROBABILITY_THRESHOLDS


def build() -> Cohort:
    # P(truth): S1=0.9, S2=0.005 (<0.01), S3=0.0005 (<0.001), S4=0.0 (zero),
    # S5 undefined.
    return Cohort(
        ["S1", "S2", "S3", "S4", "S5"],
        ["G1", "G2", "G2", "G2", "G99"],
        [
            Distribution({"G1": 0.9, "G2": 0.1}),
            Distribution({"G1": 0.995, "G2": 0.005}),
            Distribution({"G1": 0.9995, "G2": 0.0005}),
            Distribution({"G1": 1.0, "G2": 0.0}),
            Distribution({"G1": 0.5, Residual(0.5): None}),
        ],
    )


class TestThresholds:
    def test_default_thresholds(self) -> None:
        rep = catastrophic_error_report(build())
        assert rep.n_identifiable == 4  # S5 undefined excluded
        counts = {tc.threshold: tc.count for tc in rep.threshold_counts}
        assert set(counts) == set(DEFAULT_PROBABILITY_THRESHOLDS)
        # Below 0.01: S2 (0.005), S3 (0.0005), S4 (zero) = 3
        assert counts[0.01] == 3
        # Below 0.001: S3 (0.0005), S4 (zero) = 2
        assert counts[0.001] == 2
        # Below 0.0001: S4 (zero) only = 1
        assert counts[0.0001] == 1

    def test_zero_probability_separate_category(self) -> None:
        rep = catastrophic_error_report(build())
        assert rep.n_zero_probability == 1
        assert rep.fraction_zero_probability == pytest.approx(1 / 4)

    def test_configurable_thresholds(self) -> None:
        rep = catastrophic_error_report(build(), thresholds=(0.5,))
        counts = {tc.threshold: tc.count for tc in rep.threshold_counts}
        # Below 0.5: S2, S3, S4 (zero) = 3
        assert counts[0.5] == 3

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            catastrophic_error_report(build(), thresholds=(1.5,))

    def test_fraction_denominator_is_identifiable(self) -> None:
        rep = catastrophic_error_report(build())
        for tc in rep.threshold_counts:
            assert tc.fraction == pytest.approx(tc.count / rep.n_identifiable)


class TestWorstAndContribution:
    def test_worst_sorted_descending(self) -> None:
        rep = catastrophic_error_report(build(), n_worst=3)
        losses = [loss for _, loss in rep.worst_subjects]
        assert losses == sorted(losses, reverse=True)
        # The zero-probability subject (inf loss) is worst.
        assert rep.worst_subjects[0][1] == math.inf
        assert rep.worst_subjects[0][0] == "S4"

    def test_total_loss_infinite_with_zero(self) -> None:
        rep = catastrophic_error_report(build())
        assert rep.total_loss == math.inf
        assert math.isnan(rep.top_contributor_fraction)

    def test_contribution_finite_cohort(self) -> None:
        c = Cohort(
            ["A", "B", "C"],
            ["G1", "G1", "G1"],
            [Distribution({"G1": p, "G2": 1 - p}) for p in (0.9, 0.9, 0.01)],
        )
        rep = catastrophic_error_report(c)
        assert math.isfinite(rep.total_loss)
        # The 0.01 subject dominates the total.
        assert rep.top_contributor_fraction > 0.5


class TestTrimmedMeanSensitivity:
    def test_off_by_default(self) -> None:
        rep = catastrophic_error_report(build())
        assert rep.trimmed_mean_log_score is None
        assert rep.trimmed_fraction is None

    def test_opt_in(self) -> None:
        # 100 finite subjects + behaviour: trimming removes upper tail.
        c = Cohort(
            [f"S{i}" for i in range(100)],
            ["G1"] * 100,
            [Distribution({"G1": 0.5, "G2": 0.5}) for _ in range(100)],
        )
        rep = catastrophic_error_report(c, include_trimmed_mean=True)
        assert rep.trimmed_fraction == 0.01
        assert rep.trimmed_mean_log_score is not None
        # All identical, so trimmed mean equals the primary mean here.
        assert rep.trimmed_mean_log_score == pytest.approx(rep.primary_mean_log_score)

    def test_primary_mean_preserved_alongside_trimmed(self) -> None:
        rep = catastrophic_error_report(build(), include_trimmed_mean=True)
        # Primary mean is still infinite (zero-prob subject), never replaced.
        assert rep.primary_mean_log_score == math.inf


class TestEdgeCases:
    def test_all_undefined_raises(self) -> None:
        c = Cohort(
            ["S1", "S2"],
            ["G99", "G98"],
            [Distribution({"G1": 0.5, Residual(0.5): None})] * 2,
        )
        with pytest.raises(ValueError, match="No identifiable"):
            catastrophic_error_report(c)

    def test_quantiles_include_infinite_tail(self) -> None:
        rep = catastrophic_error_report(build(), quantile_levels=(0.5, 1.0))
        # The maximum (q=1.0) is infinite because of the zero-probability subject.
        assert rep.loss_quantiles[1.0] == math.inf
