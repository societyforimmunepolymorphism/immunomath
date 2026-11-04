"""Tests for descriptive diagnostics: accuracy, rank, entropy, effective N.

Includes cross-checks against scipy.stats.entropy.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import entropy as scipy_entropy

from immunomath import (
    Distribution,
    Residual,
    effective_number_of_outcomes,
    entropy,
    num_candidates,
    probability_of_truth,
    rank_of_truth,
    top_k_accuracy,
)


class TestProbabilityOfTruth:
    def test_basic(self) -> None:
        assert probability_of_truth("G1", Distribution({"G1": 0.8, "G2": 0.2})) == 0.8

    def test_omitted_complete(self) -> None:
        assert probability_of_truth("Z", Distribution({"G1": 1.0})) == 0.0


class TestTopKAccuracy:
    def test_top1_hit(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1})
        assert top_k_accuracy("G1", d, k=1) == 1.0

    def test_top1_miss(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1})
        assert top_k_accuracy("G2", d, k=1) == 0.0

    def test_topk_includes(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1})
        assert top_k_accuracy("G2", d, k=2) == 1.0
        assert top_k_accuracy("G3", d, k=2) == 0.0

    def test_fractional_two_way_tie(self) -> None:
        d = Distribution({"G1": 0.5, "G2": 0.5})
        assert top_k_accuracy("G1", d, k=1) == pytest.approx(0.5)
        assert top_k_accuracy("G2", d, k=1) == pytest.approx(0.5)

    def test_fractional_three_way_tie(self) -> None:
        # Three tied for two slots -> 2/3 credit each.
        d = Distribution({"G1": 1 / 3, "G2": 1 / 3, "G3": 1 / 3})
        assert top_k_accuracy("G1", d, k=2) == pytest.approx(2 / 3)

    def test_order_invariance(self) -> None:
        a = Distribution({"G1": 0.5, "G2": 0.5})
        b = Distribution({"G2": 0.5, "G1": 0.5})
        assert top_k_accuracy("G1", a, k=1) == top_k_accuracy("G1", b, k=1)

    def test_sure_winner_above_tie(self) -> None:
        # G0 is a sure winner; G1,G2 tie for the 2nd slot at k=2.
        d = Distribution({"G0": 0.5, "G1": 0.25, "G2": 0.25})
        assert top_k_accuracy("G0", d, k=2) == 1.0
        assert top_k_accuracy("G1", d, k=2) == pytest.approx(0.5)

    def test_zero_prob_truth_never_in_topk(self) -> None:
        d = Distribution({"G1": 1.0, "G2": 0.0})
        assert top_k_accuracy("G2", d, k=1) == 0.0

    def test_invalid_k(self) -> None:
        d = Distribution({"G1": 1.0})
        with pytest.raises(ValueError):
            top_k_accuracy("G1", d, k=0)
        with pytest.raises(ValueError):
            top_k_accuracy("G1", d, k=-1)


class TestRankOfTruth:
    def test_untied_rank(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.3, "G3": 0.1})
        r = rank_of_truth("G2", d)
        assert (r.min_rank, r.max_rank) == (2, 2)
        assert not r.is_tied

    def test_tied_rank_interval(self) -> None:
        d = Distribution({"G1": 0.5, "G2": 0.5})
        r = rank_of_truth("G1", d)
        assert (r.min_rank, r.max_rank) == (1, 2)
        assert r.is_tied

    def test_rank_order_invariant(self) -> None:
        a = Distribution({"G1": 0.5, "G2": 0.3, "G3": 0.2})
        b = Distribution({"G3": 0.2, "G2": 0.3, "G1": 0.5})
        assert rank_of_truth("G2", a) == rank_of_truth("G2", b)

    def test_zero_prob_truth_undefined(self) -> None:
        d = Distribution({"G1": 1.0, "G2": 0.0})
        with pytest.raises(ValueError):
            rank_of_truth("G2", d)


class TestEntropy:
    def test_point_mass_zero(self) -> None:
        assert entropy(Distribution({"G1": 1.0})) == 0.0

    def test_uniform_is_log_n(self) -> None:
        d = Distribution({"A": 0.5, "B": 0.5})
        assert entropy(d) == pytest.approx(math.log(2))

    def test_base2_bits(self) -> None:
        d = Distribution({"A": 0.5, "B": 0.5})
        assert entropy(d, base=2) == pytest.approx(1.0)

    def test_matches_scipy(self) -> None:
        probs = [0.5, 0.3, 0.15, 0.05]
        d = Distribution({f"G{i}": p for i, p in enumerate(probs)})
        assert entropy(d) == pytest.approx(scipy_entropy(probs))

    def test_residual_contributes(self) -> None:
        d = Distribution({"G1": 0.5, Residual(0.5): None})
        # -0.5 ln 0.5 - 0.5 ln 0.5 = ln 2
        assert entropy(d) == pytest.approx(math.log(2))

    def test_explicit_zero_contributes_nothing(self) -> None:
        a = Distribution({"G1": 0.5, "G2": 0.5})
        b = Distribution({"G1": 0.5, "G2": 0.5, "G3": 0.0})
        assert entropy(a) == pytest.approx(entropy(b))


class TestEffectiveNumber:
    def test_point_mass_one(self) -> None:
        assert effective_number_of_outcomes(Distribution({"G1": 1.0})) == 1.0

    def test_uniform_equals_n(self) -> None:
        d = Distribution({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        assert effective_number_of_outcomes(d) == pytest.approx(4.0)

    def test_base_invariance(self) -> None:
        d = Distribution({"A": 0.6, "B": 0.4})
        assert effective_number_of_outcomes(d) == pytest.approx(
            effective_number_of_outcomes(d, base=2)
        )


class TestNumCandidates:
    def test_counts_all_listed(self) -> None:
        d = Distribution({"G1": 0.8, "G2": 0.2, "G3": 0.0})
        assert num_candidates(d) == 3

    def test_positive_only(self) -> None:
        d = Distribution({"G1": 0.8, "G2": 0.2, "G3": 0.0})
        assert num_candidates(d, positive_only=True) == 2

    def test_residual_not_counted(self) -> None:
        d = Distribution({"G1": 0.8, Residual(0.2): None})
        assert num_candidates(d) == 1
