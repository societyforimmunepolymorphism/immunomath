"""Tests for calibration diagnostics (Milestone 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from immunomath import (
    Cohort,
    Distribution,
    Residual,
    event_calibration,
    top_label_calibration,
)
from immunomath.calibration import wilson_interval

D = Distribution


def two_outcome_cohort(confidences, truths):
    n = len(confidences)
    return Cohort(
        [f"s{i}" for i in range(n)],
        truths,
        [D({"G1": c, "G2": 1.0 - c}) for c in confidences],
    )


class TestWilson:
    def test_known_value(self) -> None:
        low, high = wilson_interval(8, 10)
        assert low == pytest.approx(0.4902, abs=1e-4)
        assert high == pytest.approx(0.9433, abs=1e-4)

    def test_extremes_stay_in_unit_interval(self) -> None:
        assert wilson_interval(0, 5)[0] == 0.0
        assert wilson_interval(5, 5)[1] == 1.0
        low, high = wilson_interval(0, 5)
        assert 0.0 <= low <= high <= 1.0

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError):
            wilson_interval(0, 0)
        with pytest.raises(ValueError):
            wilson_interval(1, 2, confidence=1.0)


class TestTopLabelHandComputed:
    def setup_method(self) -> None:
        # confidences .9,.9,.9,.8 ; correct 1,1,0,1
        self.cohort = Cohort(
            ["a", "b", "c", "d"],
            ["G1", "G1", "G2", "G2"],
            [D({"G1": 0.9, "G2": 0.1})] * 3 + [D({"G1": 0.2, "G2": 0.8})],
        )
        self.result = top_label_calibration(self.cohort)

    def test_ece_mce(self) -> None:
        assert self.result.ece == pytest.approx(0.75 * abs(2 / 3 - 0.9) + 0.25 * abs(1.0 - 0.8))
        assert self.result.mce == pytest.approx(abs(2 / 3 - 0.9))

    def test_bins_and_counts(self) -> None:
        by_index = {b.index: b for b in self.result.bins}
        assert len(self.result.bins) == 10  # default, fixed width keeps empty bins
        assert by_index[9].count == 3
        assert by_index[8].count == 1
        assert sum(b.count for b in self.result.bins) == self.result.n_evaluated == 4
        assert by_index[0].count == 0
        assert math.isnan(by_index[0].observed_rate)

    def test_record_of_method(self) -> None:
        assert self.result.binning == "fixed_width"
        assert self.result.n_bins_requested == 10
        assert self.result.kind == "top_label"
        assert self.result.ci_confidence == 0.95


class TestBinning:
    def test_confidence_one_falls_in_last_bin(self) -> None:
        result = top_label_calibration(two_outcome_cohort([1.0, 0.55], ["G1", "G1"]))
        assert result.bins[-1].count == 1

    def test_equal_frequency_balanced_and_ordered(self) -> None:
        conf = [0.5 + 0.005 * i for i in range(97)]
        result = top_label_calibration(
            two_outcome_cohort(conf, ["G1"] * 97), binning="equal_frequency", n_bins=10
        )
        counts = [b.count for b in result.bins]
        assert sum(counts) == 97
        assert max(counts) - min(counts) <= 1
        means = [b.mean_predicted for b in result.bins]
        assert means == sorted(means)

    def test_equal_frequency_fewer_subjects_than_bins(self) -> None:
        result = top_label_calibration(
            two_outcome_cohort([0.6, 0.7, 0.9], ["G1"] * 3), binning="equal_frequency"
        )
        assert len(result.bins) == 3
        assert all(b.count == 1 for b in result.bins)

    def test_invalid_arguments(self) -> None:
        cohort = two_outcome_cohort([0.6], ["G1"])
        with pytest.raises(ValueError, match="binning"):
            top_label_calibration(cohort, binning="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_bins"):
            top_label_calibration(cohort, n_bins=0)


class TestKnownSyntheticProbabilities:
    """Calibration against data whose true probabilities are known by construction."""

    def _simulate(self, true_prob_of_top, seed=0, n=20000):
        rng = np.random.default_rng(seed)
        conf = rng.uniform(0.5, 1.0, size=n)
        top_is_truth = rng.random(n) < true_prob_of_top(conf)
        truths = ["G1" if hit else "G2" for hit in top_is_truth]
        return two_outcome_cohort(conf.tolist(), truths)

    def test_calibrated_forecaster_has_small_ece(self) -> None:
        result = top_label_calibration(self._simulate(lambda c: c))
        assert result.ece < 0.015

    def test_calibrated_bins_cover_the_diagonal(self) -> None:
        result = top_label_calibration(self._simulate(lambda c: c))
        covered = [
            b for b in result.bins if b.count > 0 and b.ci_low <= b.mean_predicted <= b.ci_high
        ]
        populated = [b for b in result.bins if b.count > 0]
        assert len(covered) >= len(populated) - 1

    def test_overconfident_forecaster_has_large_ece(self) -> None:
        # Reports 0.5-1.0 but is right only 50% of the time: gap is mean(conf)-0.5 = 0.25.
        result = top_label_calibration(self._simulate(lambda c: np.full_like(c, 0.5)))
        assert result.ece == pytest.approx(0.25, abs=0.02)
        assert result.mce > result.ece

    def test_underconfident_gap_is_positive(self) -> None:
        result = top_label_calibration(self._simulate(lambda c: np.full_like(c, 1.0)))
        assert all(b.gap > 0 for b in result.bins if b.count > 0 and b.mean_predicted < 0.999)


class TestTiesAndExclusions:
    def test_tied_top_gets_fractional_credit(self) -> None:
        cohort = Cohort(["a"], ["G1"], [D({"G1": 0.5, "G2": 0.5})])
        result = top_label_calibration(cohort)
        populated = next(b for b in result.bins if b.count)
        assert populated.observed_rate == pytest.approx(0.5)

    def test_undefined_excluded_and_reported(self) -> None:
        cohort = Cohort(
            ["a", "b"],
            ["G1", "Gx"],
            [D({"G1": 0.8, "G2": 0.2}), D({"G1": 0.5, Residual(0.5): None})],
        )
        result = top_label_calibration(cohort)
        assert (result.n_total, result.n_evaluated, result.n_undefined) == (2, 1, 1)
        assert result.undefined_reasons[0][0] == "b"

    def test_order_invariance(self) -> None:
        conf = [0.55, 0.65, 0.75, 0.85, 0.95, 0.6]
        truths = ["G1", "G2", "G1", "G1", "G1", "G2"]
        a = top_label_calibration(two_outcome_cohort(conf, truths))
        order = [3, 0, 5, 1, 4, 2]
        b = top_label_calibration(
            two_outcome_cohort([conf[i] for i in order], [truths[i] for i in order])
        )
        assert a.ece == pytest.approx(b.ece)
        assert a.mce == pytest.approx(b.mce)


class TestEventCalibration:
    def setup_method(self) -> None:
        self.cohort = Cohort(
            ["a", "b"],
            ["G1", "G3"],
            [D({"G1": 0.6, "G2": 0.3, "G3": 0.1}), D({"G1": 0.2, "G2": 0.1, "G3": 0.7})],
        )

    def test_brier_hand_computed(self) -> None:
        result = event_calibration(self.cohort, {"G1", "G2"}, label="G1 or G2")
        assert result.brier == pytest.approx(((0.9 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
        assert result.kind == "event"
        assert result.label == "G1 or G2"

    def test_aggregates_probabilities_of_event_members(self) -> None:
        result = event_calibration(self.cohort, {"G1", "G2"}, n_bins=10)
        pred = sorted(b.mean_predicted for b in result.bins if b.count)
        assert pred == pytest.approx([0.3, 0.9])

    def test_open_distribution_excluded_not_scored_as_lower_bound(self) -> None:
        cohort = Cohort(
            ["a", "b"],
            ["G1", "G1"],
            [D({"G1": 0.5, "G2": 0.3, Residual(0.2): None}), D({"G1": 0.6, "G2": 0.4})],
        )
        result = event_calibration(cohort, {"G1"})
        assert result.n_evaluated == 1
        assert result.n_undefined == 1
        assert "residual" in result.undefined_reasons[0][1]

    def test_empty_event_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            event_calibration(self.cohort, set())

    def test_event_calibrated_on_known_probabilities(self) -> None:
        rng = np.random.default_rng(5)
        n = 20000
        p = rng.uniform(0.05, 0.95, size=n)
        happened = rng.random(n) < p
        cohort = Cohort(
            [f"s{i}" for i in range(n)],
            ["pos" if h else "neg" for h in happened],
            [D({"pos": float(pi), "neg": float(1 - pi)}) for pi in p],
        )
        result = event_calibration(cohort, {"pos"})
        assert result.ece < 0.015
        # Brier of a calibrated forecaster is E[p(1-p)] = 1/6-ish for U(.05,.95)
        assert result.brier == pytest.approx(float(np.mean(p * (1 - p))), abs=0.01)
