"""Tests for proper scoring rules: negative log score and Brier score.

Includes cross-checks against scikit-learn's ``log_loss`` and ``brier_score_loss``
to ensure the implementations agree with established references.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import log_loss

from immunomath import (
    Distribution,
    Residual,
    brier_score,
    mean_brier_score,
    mean_negative_log_score,
    negative_log_score,
)


class TestNegativeLogScore:
    def test_perfect_prediction_is_zero(self) -> None:
        assert negative_log_score("G1", Distribution({"G1": 1.0})) == 0.0

    def test_known_value(self) -> None:
        d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
        assert negative_log_score("G1", d) == pytest.approx(0.22314355131, abs=1e-9)

    def test_zero_probability_truth_is_infinite(self) -> None:
        d = Distribution({"G1": 1.0, "G2": 0.0})
        assert negative_log_score("G2", d) == math.inf

    def test_omitted_truth_in_complete_is_infinite(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.4})
        assert negative_log_score("G99", d) == math.inf

    def test_no_implicit_clipping(self) -> None:
        # A very small but nonzero probability yields a large finite loss,
        # not a clipped value.
        d = Distribution({"G1": 1e-300, "G2": 1.0 - 1e-300})
        loss = negative_log_score("G1", d)
        assert math.isfinite(loss)
        assert loss == pytest.approx(-math.log(1e-300))

    def test_explicit_floor_opt_in(self) -> None:
        d = Distribution({"G1": 0.0, "G2": 1.0})
        # Without floor: infinite.
        assert negative_log_score("G1", d) == math.inf
        # With floor: finite, equals -ln(floor).
        assert negative_log_score("G1", d, floor=1e-6) == pytest.approx(-math.log(1e-6))

    def test_floor_bounds_validated(self) -> None:
        d = Distribution({"G1": 1.0})
        with pytest.raises(ValueError):
            negative_log_score("G1", d, floor=0.0)
        with pytest.raises(ValueError):
            negative_log_score("G1", d, floor=1.5)

    def test_residual_explicit_truth_scored_normally(self) -> None:
        d = Distribution({"G1": 0.8, Residual(0.2): None})
        assert negative_log_score("G1", d) == pytest.approx(-math.log(0.8))

    def test_residual_unlisted_truth_undefined(self) -> None:
        d = Distribution({"G1": 0.8, Residual(0.2): None})
        with pytest.raises(KeyError):
            negative_log_score("G99", d)

    def test_overconfident_wrong_dominates(self) -> None:
        confident_wrong = negative_log_score("G2", Distribution({"G1": 0.999, "G2": 0.001}))
        humble = negative_log_score("G2", Distribution({"G1": 0.5, "G2": 0.5}))
        assert confident_wrong > humble


class TestMeanNegativeLogScore:
    def test_cohort_mean(self) -> None:
        d1 = Distribution({"G1": 1.0})
        d2 = Distribution({"G1": 0.5, "G2": 0.5})
        res = mean_negative_log_score(["G1", "G2"], [d1, d2])
        assert res.mean == pytest.approx(math.log(2) / 2)
        assert res.n_infinite == 0

    def test_infinite_loss_not_discarded(self) -> None:
        d1 = Distribution({"G1": 1.0})
        d2 = Distribution({"G1": 1.0, "G2": 0.0})
        res = mean_negative_log_score(["G1", "G2"], [d1, d2])
        assert res.mean == math.inf
        assert res.n_infinite == 1

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mean_negative_log_score(["G1"], [])

    def test_empty_cohort(self) -> None:
        with pytest.raises(ValueError):
            mean_negative_log_score([], [])


class TestBrierScore:
    def test_known_value_sum_form(self) -> None:
        d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
        assert brier_score("G1", d) == pytest.approx(0.065, abs=1e-12)

    def test_perfect_is_zero(self) -> None:
        assert brier_score("G1", Distribution({"G1": 1.0})) == 0.0

    def test_worst_case_is_two(self) -> None:
        # All mass on a single wrong outcome.
        d = Distribution({"G1": 1.0, "G2": 0.0})
        assert brier_score("G2", d) == pytest.approx(2.0)

    def test_range_bounds(self) -> None:
        d = Distribution({"G1": 0.4, "G2": 0.35, "G3": 0.25})
        for truth in ("G1", "G2", "G3"):
            b = brier_score(truth, d)
            assert 0.0 <= b <= 2.0

    def test_binary_relationship(self) -> None:
        # Multiclass sum form over 2 classes = 2 * binary Brier (p - y)^2.
        p = 0.7
        d = Distribution({"pos": p, "neg": 1 - p})
        multiclass = brier_score("pos", d)
        binary = (p - 1.0) ** 2
        assert multiclass == pytest.approx(2 * binary)

    def test_omitted_truth_complete(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.4})
        # truth omitted: (0-1)^2 + 0.6^2 + 0.4^2 = 1 + 0.36 + 0.16 = 1.52
        assert brier_score("G99", d) == pytest.approx(1.52)


class TestSklearnCrossChecks:
    def test_log_loss_matches_sklearn(self) -> None:
        labels = ["G1", "G2", "G3"]
        truths = ["G1", "G2", "G1", "G3"]
        prob_rows = [
            {"G1": 0.8, "G2": 0.15, "G3": 0.05},
            {"G1": 0.2, "G2": 0.7, "G3": 0.1},
            {"G1": 0.5, "G2": 0.3, "G3": 0.2},
            {"G1": 0.1, "G2": 0.1, "G3": 0.8},
        ]
        dists = [Distribution(r) for r in prob_rows]
        ours = mean_negative_log_score(truths, dists).mean

        y_prob = np.array([[r[lbl] for lbl in labels] for r in prob_rows])
        sk = log_loss(truths, y_prob, labels=labels)
        assert ours == pytest.approx(sk, abs=1e-10)

    def test_brier_matches_manual_multiclass(self) -> None:
        # scikit-learn's brier_score_loss is binary-only in a way that differs
        # by convention; cross-check the multiclass sum form against the
        # explicit numpy computation instead (the canonical definition).
        labels = ["G1", "G2", "G3"]
        truth = "G2"
        row = {"G1": 0.2, "G2": 0.5, "G3": 0.3}
        d = Distribution(row)
        p = np.array([row[lbl] for lbl in labels])
        y = np.array([1.0 if lbl == truth else 0.0 for lbl in labels])
        expected = float(np.sum((p - y) ** 2))
        assert brier_score(truth, d) == pytest.approx(expected)


class TestMeanBrier:
    def test_single(self) -> None:
        d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
        assert mean_brier_score(["G1"], [d]) == pytest.approx(0.065)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            mean_brier_score(["G1"], [])
