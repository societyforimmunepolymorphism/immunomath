"""Tests for risk-coverage analysis with abstention (Milestone 4)."""

from __future__ import annotations

import math

import pytest

from immunomath import Cohort, Distribution, Residual, risk_coverage

D = Distribution


def base_cohort() -> Cohort:
    return Cohort(
        ["a", "b", "c", "d"],
        ["G1", "G1", "G2", "G1"],
        [
            D({"G1": 0.95, "G2": 0.05}),
            D({"G1": 0.8, "G2": 0.2}),
            D({"G1": 0.7, "G2": 0.3}),
            D({"G1": 0.6, "G2": 0.4}),
        ],
    )


class TestDefaultCurve:
    def test_hand_computed_points(self) -> None:
        result = risk_coverage(base_cohort())
        got = [(p.n_retained, p.coverage, p.accuracy) for p in result.points]
        assert got == [
            (1, 0.25, 1.0),
            (2, 0.5, 1.0),
            (3, 0.75, pytest.approx(2 / 3)),
            (4, 1.0, 0.75),
        ]

    def test_accuracy_and_log_loss_reported_together(self) -> None:
        result = risk_coverage(base_cohort())
        first = result.points[0]
        assert first.mean_log_loss == pytest.approx(-math.log(0.95))
        assert first.error_rate == pytest.approx(1 - first.accuracy)
        last = result.points[-1]
        expected = (-math.log(0.95) - math.log(0.8) - math.log(0.3) - math.log(0.6)) / 4
        assert last.mean_log_loss == pytest.approx(expected)

    def test_full_coverage_reference_point(self) -> None:
        result = risk_coverage(base_cohort())
        assert result.full_coverage is not None
        assert result.full_coverage.coverage == 1.0
        assert result.full_coverage.accuracy == pytest.approx(0.75)
        assert result.score_source == "top_probability"

    def test_coverage_is_non_decreasing_as_threshold_falls(self) -> None:
        points = risk_coverage(base_cohort()).points
        assert [p.threshold for p in points] == sorted((p.threshold for p in points), reverse=True)
        coverage = [p.coverage for p in points]
        assert coverage == sorted(coverage)


class TestTiesAndOrder:
    def test_equal_scores_retained_together(self) -> None:
        cohort = Cohort(
            ["a", "b", "c"],
            ["G1", "G2", "G1"],
            [D({"G1": 0.8, "G2": 0.2})] * 3,
        )
        result = risk_coverage(cohort)
        assert len(result.points) == 1  # one distinct score: all or nothing
        assert result.points[0].n_retained == 3

    def test_input_order_does_not_change_curve(self) -> None:
        cohort = base_cohort()
        order = [2, 0, 3, 1]
        shuffled = Cohort(
            [cohort.subject_ids[i] for i in order],
            [cohort.truths[i] for i in order],
            [cohort.distributions[i] for i in order],
        )
        assert risk_coverage(cohort).points == risk_coverage(shuffled).points


class TestUserDefinedConfidenceAndAbstention:
    def test_callable_score(self) -> None:
        # Score by margin between the top two probabilities.
        def margin(dist: Distribution) -> float:
            top = sorted(dist.probabilities, reverse=True)
            return top[0] - top[1]

        result = risk_coverage(base_cohort(), confidence=margin)
        assert result.score_source == "callable"
        assert result.points[-1].coverage == 1.0

    def test_mapping_with_algorithm_abstention(self) -> None:
        scores = {"a": 0.9, "b": None, "c": 0.5, "d": 0.7}
        result = risk_coverage(base_cohort(), confidence=scores)
        assert result.score_source == "mapping"
        assert result.n_abstained == 1
        assert result.n_evaluable == 4
        # Abstained subject never retained, so maximum coverage is 3/4.
        assert result.points[-1].coverage == pytest.approx(0.75)
        assert result.points[-1].n_retained == 3

    def test_mapping_missing_subject_rejected(self) -> None:
        with pytest.raises(ValueError, match="no entry"):
            risk_coverage(base_cohort(), confidence={"a": 0.9})

    def test_nan_score_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            risk_coverage(base_cohort(), confidence=lambda d: math.nan)

    def test_explicit_thresholds(self) -> None:
        result = risk_coverage(base_cohort(), thresholds=[0.65, 0.99])
        by_t = {p.threshold: p for p in result.points}
        assert by_t[0.65].n_retained == 3
        assert by_t[0.99].n_retained == 0
        assert by_t[0.99].coverage == 0.0
        assert math.isnan(by_t[0.99].accuracy)

    def test_uninformative_score_gives_single_point(self) -> None:
        result = risk_coverage(base_cohort(), confidence=lambda d: 1.0)
        assert len(result.points) == 1
        assert result.points[0].coverage == 1.0


class TestInfiniteAndUndefined:
    def test_zero_probability_truth_makes_log_loss_infinite_when_retained(self) -> None:
        cohort = Cohort(
            ["a", "b"],
            ["G1", "G2"],
            [D({"G1": 0.9, "G2": 0.1}), D({"G1": 1.0, "G2": 0.0})],
        )
        result = risk_coverage(cohort)
        # b is the most confident (1.0) but wrong with zero probability on the truth.
        assert result.points[0].n_retained == 1
        assert result.points[0].mean_log_loss == math.inf
        assert result.points[0].n_infinite_retained == 1
        assert result.points[0].accuracy == 0.0

    def test_undefined_excluded_from_denominator(self) -> None:
        cohort = Cohort(
            ["a", "b"],
            ["G1", "Gx"],
            [D({"G1": 0.8, "G2": 0.2}), D({"G1": 0.5, Residual(0.5): None})],
        )
        result = risk_coverage(cohort)
        assert (result.n_total, result.n_evaluable, result.n_undefined) == (2, 1, 1)
        assert result.points[-1].coverage == 1.0

    def test_invalid_floor(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            risk_coverage(base_cohort(), floor=2.0)
