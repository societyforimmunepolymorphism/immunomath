"""Tests for resampling statistics: bootstrap CIs and the trimmed mean."""

from __future__ import annotations

import math

import numpy as np
import pytest

from immunomath import (
    bootstrap_ci,
    paired_bootstrap_ci,
    trimmed_mean,
)
from immunomath.statistics import _inf_aware_quantile


class TestTrimmedMean:
    def test_trims_upper_tail(self) -> None:
        assert trimmed_mean([1.0, 2.0, 3.0, 100.0], upper_fraction=0.25) == pytest.approx(2.0)

    def test_small_cohort_no_trim_at_one_percent(self) -> None:
        # n=3, floor(0.01*3)=0 -> no trimming, equals ordinary mean.
        assert trimmed_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_trims_infinity_from_upper_tail(self) -> None:
        # 100 values, 1% trims the single +inf, leaving a finite mean.
        vals = [float(i) for i in range(99)] + [math.inf]
        result = trimmed_mean(vals, upper_fraction=0.01)
        assert math.isfinite(result)

    def test_infinity_remains_if_not_trimmed(self) -> None:
        # n=3 at 1% trims nothing, so +inf remains and the mean is +inf.
        assert trimmed_mean([1.0, 2.0, math.inf]) == math.inf

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            trimmed_mean([])

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            trimmed_mean([1.0, math.nan])

    def test_rejects_bad_fraction(self) -> None:
        with pytest.raises(ValueError):
            trimmed_mean([1.0, 2.0], upper_fraction=1.0)


class TestBootstrapCI:
    def test_point_estimate_is_sample_mean(self) -> None:
        res = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=0, n_resamples=500)
        assert res.point_estimate == pytest.approx(0.3)

    def test_deterministic_with_seed(self) -> None:
        a = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=42, n_resamples=1000)
        b = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=42, n_resamples=1000)
        assert (a.low, a.high) == (b.low, b.high)

    def test_different_seeds_differ(self) -> None:
        a = bootstrap_ci([0.1, 0.5, 0.9, 0.2, 0.7], seed=1, n_resamples=1000)
        b = bootstrap_ci([0.1, 0.5, 0.9, 0.2, 0.7], seed=2, n_resamples=1000)
        assert (a.low, a.high) != (b.low, b.high)

    def test_ci_brackets_point(self) -> None:
        res = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=0, n_resamples=1000)
        assert res.low <= res.point_estimate <= res.high

    def test_ci_covers_true_mean_frequently(self) -> None:
        # Coverage sanity check: a 95% CI for the mean of a known population
        # should cover the truth in most of many repetitions.
        rng = np.random.default_rng(0)
        true_mean = 5.0
        covered = 0
        trials = 200
        for t in range(trials):
            sample = rng.normal(true_mean, 2.0, size=80).tolist()
            res = bootstrap_ci(sample, seed=t, n_resamples=400)
            if res.low <= true_mean <= res.high:
                covered += 1
        # Allow slack for bootstrap + finite trials; expect well above 0.8.
        assert covered / trials >= 0.85

    def test_infinite_values_counted_and_bound_infinite(self) -> None:
        res = bootstrap_ci([0.1, 0.2, math.inf], seed=0, n_resamples=500)
        assert res.point_estimate == math.inf
        assert res.high == math.inf
        assert res.n_infinite_resamples > 0

    def test_bca_runs_and_brackets(self) -> None:
        res = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 5.0], seed=1, n_resamples=1000, method="bca")
        assert res.method == "bca"
        assert res.low <= res.point_estimate <= res.high

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_rejects_bad_confidence(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci([0.1, 0.2], confidence=1.5)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            bootstrap_ci([0.1, math.nan])


class TestPairedBootstrap:
    def test_mean_difference(self) -> None:
        a = [0.5, 0.4, 0.6, 0.3]
        b = [0.4, 0.5, 0.4, 0.2]
        res = paired_bootstrap_ci(a, b, seed=1, n_resamples=500)
        assert res.point_estimate == pytest.approx(0.075)

    def test_pairing_preserved_deterministic(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [0.5, 1.5, 2.5, 3.5]
        r1 = paired_bootstrap_ci(a, b, seed=3, n_resamples=500)
        r2 = paired_bootstrap_ci(a, b, seed=3, n_resamples=500)
        assert (r1.low, r1.high) == (r2.low, r2.high)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            paired_bootstrap_ci([1.0], [1.0, 2.0])

    def test_inf_minus_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="inf - inf"):
            paired_bootstrap_ci([math.inf], [math.inf], seed=0)

    def test_inf_minus_finite_is_inf_difference(self) -> None:
        # a=inf, b=finite -> difference +inf, handled (not indeterminate).
        res = paired_bootstrap_ci([math.inf, 0.5], [0.1, 0.2], seed=0, n_resamples=200)
        assert res.point_estimate == math.inf


class TestInfAwareQuantile:
    def test_matches_numpy_when_finite(self) -> None:
        stats = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.7])
        for q in (0.025, 0.5, 0.975):
            assert _inf_aware_quantile(stats, q) == pytest.approx(
                float(np.quantile(stats, q, method="linear"))
            )

    def test_upper_bound_infinite_when_tail_infinite(self) -> None:
        stats = np.array([0.1, 0.2, 0.3, math.inf, math.inf])
        assert _inf_aware_quantile(stats, 0.975) == math.inf
        assert math.isfinite(_inf_aware_quantile(stats, 0.1))
