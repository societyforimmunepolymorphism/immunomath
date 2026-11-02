# Uncertainty: Bootstrap Confidence Intervals

A cohort mean is an *estimate*. ImmunoMath quantifies its sampling uncertainty
with a **subject-level nonparametric bootstrap**, implemented in
[`bootstrap_ci`][immunomath.bootstrap_ci] and surfaced in every
[`CohortSummary`][immunomath.CohortSummary].

## The nonparametric bootstrap

Given per-subject losses $x_1, \dots, x_n$, each **bootstrap resample** draws
$n$ subjects *with replacement* and recomputes the statistic (here, the mean).
Repeating this $B$ times yields the bootstrap distribution of the statistic,
whose spread estimates the sampling uncertainty of the original estimate.

ImmunoMath resamples at the **subject** level, so the unit of resampling matches
the unit of observation. Defaults follow the scientific specification:

| Parameter | Default |
| --- | --- |
| resamples $B$ | `2000` |
| confidence | `0.95` |
| method | `percentile` |

## Reproducibility

Every bootstrap is driven by a seeded NumPy
`numpy.random.Generator`. **The same `seed` gives identical
bounds**, which is required for reproducible analyses:

```python
>>> from immunomath import bootstrap_ci
>>> a = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=42, n_resamples=1000)
>>> b = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=42, n_resamples=1000)
>>> (a.low, a.high) == (b.low, b.high)
True
```

## Percentile intervals

The percentile interval takes the empirical $\alpha/2$ and $1-\alpha/2$
quantiles of the bootstrap statistics. It is simple and transformation-respecting
but can be biased for skewed statistics.

## BCa intervals (optional)

The **bias-corrected and accelerated** (BCa) interval adjusts the percentile
endpoints using

- a bias-correction $\hat{z}_0$, from the fraction of bootstrap statistics below
  the point estimate, and
- an acceleration $\hat{a}$, from the jackknife skewness of the sample.

The adjusted level for a target normal quantile $z$ is

$$
\Phi\!\left( \hat{z}_0 + \frac{\hat{z}_0 + z}{1 - \hat{a}\,(\hat{z}_0 + z)} \right),
$$

where $\Phi$ is the standard normal CDF (Efron & Tibshirani, 1993). BCa is more
accurate for the skewed log-loss distributions typical of imputation evaluation.

```python
>>> res = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 5.0], seed=1,
...                     n_resamples=1000, method="bca")
>>> res.method
'bca'
>>> res.low <= res.point_estimate <= res.high
True
```

## Infinite losses are never laundered into finite intervals

This is the critical safety property. If some subjects have infinite loss (zero
probability on the truth), a bootstrap resample that includes one has an
**infinite** mean. ImmunoMath:

1. counts such resamples in
   [`n_infinite_resamples`][immunomath.BootstrapResult], and
2. reports any confidence bound that falls within the infinite block as `+inf`,
   never as a misleadingly finite quantile.

```python
>>> import math
>>> res = bootstrap_ci([0.1, 0.2, math.inf], seed=0, n_resamples=500)
>>> res.point_estimate
inf
>>> res.high
inf
>>> res.n_infinite_resamples > 0
True
```

A finite lower bound alongside an infinite upper bound is the honest
representation of "the mean is probably large and possibly unbounded."

## Paired resampling for comparisons

To compare two measurements on the **same** individuals,
[`paired_bootstrap_ci`][immunomath.paired_bootstrap_ci] resamples subject
*indices* once per replicate and applies them to both arms, preserving the
pairing. The statistic is the mean of the paired differences $d_i = a_i - b_i$.

```python
>>> from immunomath import paired_bootstrap_ci
>>> a = [0.5, 0.4, 0.6, 0.3]
>>> b = [0.4, 0.5, 0.4, 0.2]
>>> res = paired_bootstrap_ci(a, b, seed=1, n_resamples=500)
>>> round(res.point_estimate, 4)
0.075
```

An `inf − inf` paired difference (both arms assigned the same subject's truth
zero probability) is **indeterminate** and raises, rather than silently
producing `nan`. This is the foundation for the Milestone 3 comparison
framework.

## Assumptions and failure modes

- The bootstrap assumes subjects are exchangeable draws from the population of
  interest; it does not correct for a biased or non-representative cohort.
- Coverage is approximate and improves with cohort size; very small cohorts give
  wide, unstable intervals.
- NaN inputs are rejected — a NaN loss is a defect, not a datum.

## References

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
- DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals.
  *Statistical Science*, 11(3), 189–228.
