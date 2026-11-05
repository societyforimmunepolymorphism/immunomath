"""Deterministic tests for TRS-aware observation-weighted scoring.

Covers the eleven required properties of the weighted-scoring milestone:

1.  Uniform weights reproduce ordinary scoring.
2.  lambda = 0 reproduces unweighted results (via the TRS adapter).
3.  Lower TRS produces greater-or-equal weight.
4.  Scaling all weights by a positive constant leaves the weighted mean fixed.
5.  Weighted calculations agree with hand-computed examples.
6.  Paired score differences have the correct sign.
7.  Permuting aligned records does not change aggregate results.
8.  Invalid weights, TRS values, and parameters are rejected.
9.  Macro-averages are correct for unequal group sizes.
10. Existing zero-probability and residual behavior is preserved.
11. Existing public APIs and serialization remain backward compatible.

Also checks explicit infinity handling (no silent inf - inf).
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from immunomath import (
    Distribution,
    LossMetric,
    Residual,
    compare_methods_weighted,
    mean_brier_score,
    mean_negative_log_score,
    per_observation_losses,
    stratified_comparison,
    trs_weights,
    validate_weights,
    weighted_mean_loss,
)


def _dist(mapping: dict[str, float]) -> Distribution:
    return Distribution(mapping)


# --------------------------------------------------------------------------- #
# Property 1: uniform weights reproduce ordinary scoring.
# --------------------------------------------------------------------------- #
class TestUniformReproducesOrdinary:
    truths: ClassVar = ["G1", "G2", "G1"]
    dists: ClassVar = [
        _dist({"G1": 0.9, "G2": 0.1}),
        _dist({"G1": 0.4, "G2": 0.6}),
        _dist({"G1": 0.7, "G2": 0.3}),
    ]

    def test_log_none_weights_match_mean_negative_log_score(self) -> None:
        ws = weighted_mean_loss(self.truths, self.dists, metric=LossMetric.LOG)
        ref = mean_negative_log_score(self.truths, self.dists).mean
        assert ws.weighted_mean == pytest.approx(ref)
        assert ws.unweighted_mean == pytest.approx(ref)
        assert ws.weighted_mean == pytest.approx(ws.unweighted_mean)

    def test_brier_none_weights_match_mean_brier_score(self) -> None:
        ws = weighted_mean_loss(self.truths, self.dists, metric="brier")
        ref = mean_brier_score(self.truths, self.dists)
        assert ws.weighted_mean == pytest.approx(ref)
        assert ws.unweighted_mean == pytest.approx(ref)

    def test_explicit_uniform_weights_match_none(self) -> None:
        a = weighted_mean_loss(self.truths, self.dists)
        b = weighted_mean_loss(self.truths, self.dists, [1.0, 1.0, 1.0])
        assert a.weighted_mean == pytest.approx(b.weighted_mean)

    def test_effective_sample_size_uniform_equals_n(self) -> None:
        ws = weighted_mean_loss(self.truths, self.dists)
        assert ws.effective_sample_size == pytest.approx(3.0)
        # Any constant weight also gives N_eff == n.
        ws2 = weighted_mean_loss(self.truths, self.dists, [5.0, 5.0, 5.0])
        assert ws2.effective_sample_size == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Property 2: lambda = 0 reproduces unweighted results.
# --------------------------------------------------------------------------- #
def test_lambda_zero_gives_uniform_weights_and_results() -> None:
    truths = ["G1", "G2"]
    dists = [_dist({"G1": 0.8, "G2": 0.2}), _dist({"G1": 0.3, "G2": 0.7})]
    trs = [0.2, 0.9]
    w = trs_weights(trs, lam=0.0, gamma=2.0)
    assert w == (1.0, 1.0)
    weighted = weighted_mean_loss(truths, dists, w)
    uniform = weighted_mean_loss(truths, dists)
    assert weighted.weighted_mean == pytest.approx(uniform.weighted_mean)


# --------------------------------------------------------------------------- #
# Property 3: lower TRS -> greater-or-equal weight.
# --------------------------------------------------------------------------- #
def test_lower_trs_gives_greater_or_equal_weight() -> None:
    trs = [0.0, 0.25, 0.5, 0.75, 1.0]
    w = trs_weights(trs, lam=4.0, gamma=2.0)
    # Weights are monotone non-increasing as resolution increases.
    assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))
    # Lowest resolution gets the largest weight; highest gets exactly 1.
    assert w[0] == max(w)
    assert w[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Property 4: scaling all weights by a positive constant is invariant.
# --------------------------------------------------------------------------- #
def test_positive_scaling_of_weights_is_invariant() -> None:
    truths = ["G1", "G2", "G1"]
    dists = [
        _dist({"G1": 0.9, "G2": 0.1}),
        _dist({"G1": 0.4, "G2": 0.6}),
        _dist({"G1": 0.7, "G2": 0.3}),
    ]
    base = [1.0, 2.0, 3.0]
    scaled = [7.5 * x for x in base]
    a = weighted_mean_loss(truths, dists, base)
    b = weighted_mean_loss(truths, dists, scaled)
    assert a.weighted_mean == pytest.approx(b.weighted_mean)
    # Effective sample size is also scale-invariant.
    assert a.effective_sample_size == pytest.approx(b.effective_sample_size)


# --------------------------------------------------------------------------- #
# Property 5: agreement with hand-computed examples.
# --------------------------------------------------------------------------- #
class TestHandComputed:
    def test_weighted_log_mean_by_hand(self) -> None:
        # losses: -ln(0.5)=0.6931..., -ln(0.25)=1.3862...
        truths = ["G1", "G1"]
        dists = [_dist({"G1": 0.5, "G2": 0.5}), _dist({"G1": 0.25, "G2": 0.75})]
        weights = [1.0, 3.0]
        l1 = -math.log(0.5)
        l2 = -math.log(0.25)
        expected = (1.0 * l1 + 3.0 * l2) / 4.0
        ws = weighted_mean_loss(truths, dists, weights)
        assert ws.weighted_mean == pytest.approx(expected)
        assert ws.total_weight == pytest.approx(4.0)
        # N_eff = (1+3)^2 / (1+9) = 16/10 = 1.6
        assert ws.effective_sample_size == pytest.approx(1.6)

    def test_trs_weight_values_by_hand(self) -> None:
        # w = 1 + 4*(1-r)^2 ; r=1 -> 1 ; r=0.5 -> 2 ; r=0 -> 5
        assert trs_weights([1.0, 0.5, 0.0], lam=4.0, gamma=2.0) == (1.0, 2.0, 5.0)

    def test_weighted_brier_by_hand(self) -> None:
        # Brier sum form for {G1:0.8,G2:0.2}, truth G1: (0.8-1)^2+(0.2)^2=0.04+0.04=0.08
        truths = ["G1", "G2"]
        dists = [_dist({"G1": 0.8, "G2": 0.2}), _dist({"G1": 0.5, "G2": 0.5})]
        b1 = 0.08
        b2 = (0.5) ** 2 + (0.5 - 1.0) ** 2  # 0.25 + 0.25 = 0.5
        weights = [2.0, 1.0]
        expected = (2.0 * b1 + 1.0 * b2) / 3.0
        ws = weighted_mean_loss(truths, dists, weights, metric="brier")
        assert ws.weighted_mean == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Property 6: paired score differences have the correct sign.
# --------------------------------------------------------------------------- #
class TestPairedSign:
    truths: ClassVar = ["G1", "G2"]
    incumbent: ClassVar = [_dist({"G1": 0.5, "G2": 0.5}), _dist({"G1": 0.5, "G2": 0.5})]
    candidate: ClassVar = [_dist({"G1": 0.9, "G2": 0.1}), _dist({"G1": 0.1, "G2": 0.9})]

    def test_candidate_better_positive_improvement(self) -> None:
        cmp = compare_methods_weighted(self.truths, self.incumbent, self.candidate)
        # delta = L_inc - L_cand > 0 means candidate is better.
        assert cmp.mean_improvement > 0
        assert cmp.n_improved == 2
        assert cmp.n_worsened == 0

    def test_swapping_arms_flips_sign(self) -> None:
        forward = compare_methods_weighted(self.truths, self.incumbent, self.candidate)
        backward = compare_methods_weighted(self.truths, self.candidate, self.incumbent)
        assert backward.mean_improvement == pytest.approx(-forward.mean_improvement)
        assert backward.n_worsened == forward.n_improved

    def test_per_observation_delta_matches_losses(self) -> None:
        cmp = compare_methods_weighted(self.truths, self.incumbent, self.candidate)
        la = per_observation_losses(self.truths, self.incumbent)
        lb = per_observation_losses(self.truths, self.candidate)
        for idx, delta in cmp.improvements:
            assert la[idx].loss is not None and lb[idx].loss is not None
            assert delta == pytest.approx(la[idx].loss - lb[idx].loss)


# --------------------------------------------------------------------------- #
# Property 7: permuting aligned records does not change aggregates.
# --------------------------------------------------------------------------- #
def test_permutation_invariance() -> None:
    truths = ["G1", "G2", "G1", "G2"]
    inc = [
        _dist({"G1": 0.6, "G2": 0.4}),
        _dist({"G1": 0.4, "G2": 0.6}),
        _dist({"G1": 0.7, "G2": 0.3}),
        _dist({"G1": 0.2, "G2": 0.8}),
    ]
    cand = [
        _dist({"G1": 0.8, "G2": 0.2}),
        _dist({"G1": 0.3, "G2": 0.7}),
        _dist({"G1": 0.9, "G2": 0.1}),
        _dist({"G1": 0.1, "G2": 0.9}),
    ]
    weights = [1.0, 2.0, 3.0, 4.0]
    perm = [2, 0, 3, 1]

    base = weighted_mean_loss(truths, inc, weights)
    permd = weighted_mean_loss(
        [truths[i] for i in perm],
        [inc[i] for i in perm],
        [weights[i] for i in perm],
    )
    assert base.weighted_mean == pytest.approx(permd.weighted_mean)
    assert base.effective_sample_size == pytest.approx(permd.effective_sample_size)

    cmp_base = compare_methods_weighted(truths, inc, cand, weights)
    cmp_perm = compare_methods_weighted(
        [truths[i] for i in perm],
        [inc[i] for i in perm],
        [cand[i] for i in perm],
        [weights[i] for i in perm],
    )
    assert cmp_base.mean_improvement == pytest.approx(cmp_perm.mean_improvement)
    assert cmp_base.weighted_mean_improvement == pytest.approx(
        cmp_perm.weighted_mean_improvement
    )
    assert cmp_base.n_improved == cmp_perm.n_improved
    assert cmp_base.n_worsened == cmp_perm.n_worsened


# --------------------------------------------------------------------------- #
# Property 8: invalid weights, TRS values, and parameters are rejected.
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_validate_weights_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="one entry per observation"):
            validate_weights([1.0, 2.0], n=3)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_validate_weights_rejects_nonpositive_or_nonfinite(self, bad: float) -> None:
        with pytest.raises(ValueError):
            validate_weights([1.0, bad], n=2)

    def test_weighted_mean_loss_rejects_bad_weights(self) -> None:
        truths = ["G1", "G2"]
        dists = [_dist({"G1": 1.0}), _dist({"G1": 0.5, "G2": 0.5})]
        with pytest.raises(ValueError):
            weighted_mean_loss(truths, dists, [1.0, 0.0])

    def test_unknown_metric_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_mean_loss(["G1"], [_dist({"G1": 1.0})], metric="spherical")

    @pytest.mark.parametrize("r", [-0.01, 1.01, math.nan, math.inf])
    def test_trs_values_out_of_range_rejected(self, r: float) -> None:
        with pytest.raises(ValueError):
            trs_weights([0.5, r], lam=1.0, gamma=1.0)

    def test_trs_parameters_rejected(self) -> None:
        with pytest.raises(ValueError, match="lam"):
            trs_weights([0.5], lam=-1.0, gamma=1.0)
        with pytest.raises(ValueError, match="gamma"):
            trs_weights([0.5], lam=1.0, gamma=0.0)

    def test_empty_inputs_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_mean_loss([], [])
        with pytest.raises(ValueError):
            trs_weights([], lam=1.0, gamma=1.0)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            compare_methods_weighted(["G1"], [_dist({"G1": 1.0})], [])


# --------------------------------------------------------------------------- #
# Property 9: macro-averages correct for unequal group sizes.
# --------------------------------------------------------------------------- #
class TestMacroAverage:
    def test_macro_average_equal_group_weight(self) -> None:
        # 'high': 3 near-perfect obs (tiny loss). 'low': 1 obs with larger loss.
        # Overall (donor-weighted) mean is dominated by 'high'; macro-average
        # gives 'low' equal aggregate weight, so it is larger.
        truths = ["G1", "G1", "G1", "G1"]
        inc = [
            _dist({"G1": 0.99, "G2": 0.01}),
            _dist({"G1": 0.99, "G2": 0.01}),
            _dist({"G1": 0.99, "G2": 0.01}),
            _dist({"G1": 0.5, "G2": 0.5}),
        ]
        cand = inc  # identical: improvement is zero everywhere
        strata = ["high", "high", "high", "low"]
        result = stratified_comparison(
            truths, inc, cand, strata, stratum_order=["low", "high"]
        )
        # Per-stratum means computed by hand.
        high_mean = -math.log(0.99)
        low_mean = -math.log(0.5)
        macro = (low_mean + high_mean) / 2.0
        overall = (3 * high_mean + low_mean) / 4.0
        assert result.macro_mean_log_incumbent == pytest.approx(macro)
        assert result.overall_mean_log_incumbent == pytest.approx(overall)
        assert result.macro_mean_log_incumbent > result.overall_mean_log_incumbent

    def test_empty_stratum_excluded_from_macro(self) -> None:
        truths = ["G1", "G1"]
        inc = [_dist({"G1": 0.9, "G2": 0.1}), _dist({"G1": 0.8, "G2": 0.2})]
        cand = inc
        strata = ["high", "high"]
        result = stratified_comparison(
            truths, inc, cand, strata, stratum_order=["low", "intermediate", "high"]
        )
        assert set(result.empty_strata) == {"low", "intermediate"}
        # Macro-average is over the single nonempty stratum only.
        high_mean = (-math.log(0.9) - math.log(0.8)) / 2.0
        assert result.macro_mean_log_incumbent == pytest.approx(high_mean)

    def test_unknown_stratum_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="not present in stratum_order"):
            stratified_comparison(
                ["G1"], [_dist({"G1": 1.0})], [_dist({"G1": 1.0})], ["mystery"],
                stratum_order=["low", "high"],
            )

    def test_candidate_helps_overall_but_hurts_low_trs(self) -> None:
        # The motivating case: a large high-TRS group each improves slightly,
        # dominating the donor-weighted (overall) mean so it is positive; a
        # single low-TRS donor is substantially worsened. The TRS-balanced
        # macro-average, giving the low stratum equal aggregate weight, turns
        # negative and exposes the degradation the overall mean hides.
        truths = ["G1"] * 20 + ["G1"]
        inc = [_dist({"G1": 0.90, "G2": 0.10})] * 20 + [_dist({"G1": 0.60, "G2": 0.40})]
        cand = [_dist({"G1": 0.95, "G2": 0.05})] * 20 + [_dist({"G1": 0.30, "G2": 0.70})]
        strata = ["high"] * 20 + ["low"]
        result = stratified_comparison(
            truths, inc, cand, strata, stratum_order=["low", "high"]
        )
        low = next(s for s in result.strata if s.label == "low")
        high = next(s for s in result.strata if s.label == "high")
        assert high.mean_improvement_log > 0  # candidate better in high
        assert low.mean_improvement_log < 0  # candidate worse in low
        # Donor-weighted overall improvement is positive (high group dominates)...
        assert result.overall_mean_improvement_log > 0
        # ...but the TRS-balanced macro improvement is negative, flagging the
        # low-TRS degradation that the overall mean conceals.
        assert result.macro_mean_improvement_log < 0


# --------------------------------------------------------------------------- #
# Property 10: zero-probability and residual behavior preserved.
# --------------------------------------------------------------------------- #
class TestInfinityAndResidual:
    def test_zero_probability_gives_infinite_weighted_mean(self) -> None:
        truths = ["G1", "G2"]
        dists = [_dist({"G1": 0.5, "G2": 0.5}), _dist({"G1": 1.0, "G2": 0.0})]
        ws = weighted_mean_loss(truths, dists, [1.0, 1.0])
        assert ws.n_infinite == 1
        assert ws.weighted_mean == math.inf
        assert ws.unweighted_mean == math.inf

    def test_residual_truth_is_undefined_and_excluded(self) -> None:
        truths = ["G1", "G99"]
        dists = [_dist({"G1": 0.9, "G2": 0.1}), Distribution({"G1": 0.5, Residual(0.5): None})]
        ws = weighted_mean_loss(truths, dists, [1.0, 1.0])
        assert ws.n_undefined == 1
        assert ws.n_scored == 1
        assert len(ws.undefined_reasons) == 1
        # Only the defined observation contributes.
        assert ws.weighted_mean == pytest.approx(-math.log(0.9))

    def test_paired_inf_minus_inf_excluded(self) -> None:
        # Both methods assign the truth zero probability -> inf - inf excluded.
        truths = ["G2"]
        inc = [_dist({"G1": 1.0, "G2": 0.0})]
        cand = [_dist({"G1": 1.0, "G2": 0.0})]
        cmp = compare_methods_weighted(truths, inc, cand)
        assert cmp.n_paired == 0
        assert cmp.n_excluded == 1
        assert "inf - inf" in cmp.exclusion_reasons[0][1]

    def test_one_sided_infinity_retained_as_defined(self) -> None:
        # Incumbent assigns zero (inf), candidate finite -> defined improvement.
        truths = ["G2"]
        inc = [_dist({"G1": 1.0, "G2": 0.0})]
        cand = [_dist({"G1": 0.5, "G2": 0.5})]
        cmp = compare_methods_weighted(truths, inc, cand)
        assert cmp.n_paired == 1
        assert cmp.n_improved == 1  # candidate rescues an infinite loss
        assert cmp.mean_improvement == math.inf


# --------------------------------------------------------------------------- #
# Property 11: backward compatibility of existing API and serialization.
# --------------------------------------------------------------------------- #
class TestBackwardCompatibility:
    def test_existing_scorers_unchanged(self) -> None:
        d = _dist({"G1": 0.80, "G2": 0.15, "G3": 0.05})
        from immunomath import negative_log_score

        assert negative_log_score("G1", d) == pytest.approx(0.22314355131)

    def test_new_symbols_exported(self) -> None:
        import immunomath as im

        for name in (
            "weighted_mean_loss",
            "WeightedScore",
            "compare_methods_weighted",
            "PairedWeightedComparison",
            "stratified_comparison",
            "StratifiedComparison",
            "StratumComparison",
            "LossMetric",
            "ObservationLoss",
            "per_observation_losses",
            "validate_weights",
            "trs_weights",
        ):
            assert name in im.__all__
            assert hasattr(im, name)

    def test_analyze_cohort_serialization_still_works(self) -> None:
        from immunomath import Cohort, analyze_cohort

        cohort = Cohort(
            ["a", "b"],
            ["G1", "G2"],
            [_dist({"G1": 0.8, "G2": 0.2}), _dist({"G1": 0.3, "G2": 0.7})],
        )
        result = analyze_cohort(cohort, label="bc", seed=0, n_resamples=50)
        record = result.to_dict()
        assert record["denominators"]["n_scored"] == 2
