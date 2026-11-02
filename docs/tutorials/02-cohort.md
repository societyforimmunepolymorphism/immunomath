# Tutorial 2 — Evaluating 1,000 Simulated Individuals

This tutorial scales up from a single prediction to a **cohort** of 1,000
subjects. It demonstrates cohort summaries, explicit denominators, bootstrap
confidence intervals, loss-distribution quantiles, and catastrophic-error
diagnostics. All data is synthetic and reproducible — no imputation is run.

## Generate a reproducible synthetic cohort

We draw 1,000 predictions over five candidate genotypes from a seeded generator.
About 3% of subjects are made *confidently wrong* (the truth receives a tiny
probability) so the cohort has a realistic catastrophic tail.

```python
import numpy as np
from immunomath import Cohort, Distribution

rng = np.random.default_rng(20240531)
N = 1000
candidates = [f"G{i}" for i in range(5)]

subject_ids, truths, distributions = [], [], []
for i in range(N):
    logits = rng.normal(0.0, 1.5, size=5)
    p = np.exp(logits)
    p /= p.sum()
    truth_idx = int(rng.integers(0, 5))
    if rng.random() < 0.03:  # 3% confidently wrong
        p = p * 0.01
        p[truth_idx] = 1e-4
        p /= p.sum()
    subject_ids.append(f"S{i:04d}")
    truths.append(candidates[truth_idx])
    distributions.append(Distribution({c: float(pi) for c, pi in zip(candidates, p)}))

cohort = Cohort(subject_ids, truths, distributions)
```

## Summarize the cohort

```python
summary = cohort.summarize(seed=0, n_resamples=2000)
```

The denominators are explicit — all 1,000 subjects are identifiable here, with
no undefined or infinite scores:

```text
n_total     = 1000
n_scored    = 1000     # denominator of the primary mean
n_undefined = 0
n_infinite  = 0
```

The headline proper-score summary and its 95% bootstrap confidence interval:

```text
mean_log_score        ≈ 2.32  nats
mean_log_score_ci     ≈ (2.24, 2.41)   # percentile, 2000 resamples, seed=0
median_log_score      ≈ 2.20  nats
mean_brier            ≈ 1.03
```

The mean exceeds the median, the signature of a right-skewed loss distribution
dragged up by the confidently-wrong tail. The loss quantiles make the tail
explicit:

```text
loss_quantiles ≈ {0.25: 1.26, 0.50: 2.20, 0.75: 3.18, 0.95: 4.67, 0.99: 5.97}
```

## Find the catastrophic errors

```python
from immunomath import catastrophic_error_report

report = catastrophic_error_report(cohort, include_trimmed_mean=True)
```

```text
n_zero_probability = 0            # no exactly-zero truths in this cohort
P(truth) < 0.01    : 54  (5.4%)
P(truth) < 0.001   :  2  (0.2%)
P(truth) < 0.0001  :  0  (0.0%)
```

Roughly 5% of subjects received less than 1% probability on the truth — the
injected tail. The worst few subjects dominate:

```text
worst_subjects[:3] ≈ [("S0229", 7.09), ("S0230", 7.02), ("S0617", 6.82)]
```

## The trimmed mean is a sensitivity check, not a replacement

```text
primary_mean_log_score ≈ 2.324     # the proper score (untrimmed)
trimmed_mean_log_score ≈ 2.282     # 1% upper-tail trimmed (descriptive only)
```

Trimming the worst 1% barely moves the mean here (from 2.324 to 2.282), telling
us the summary is *not* dominated by a single subject in this cohort. That is a
useful diagnosis — but the trimmed value must never be reported *in place of*
the proper score. Had the cohort contained a zero-probability subject, the
primary mean would be `+inf` while the trimmed mean could be finite; the proper
score still governs.

## What you learned

- A [`Cohort`][immunomath.Cohort] reports **explicit denominators**
  (`n_total`, `n_scored`, `n_undefined`, `n_infinite`).
- The primary mean is a proper score over identifiable subjects, with a
  **reproducible** bootstrap CI (fixed `seed`).
- Loss quantiles and the
  [catastrophic-error report](../mathematical-foundations/catastrophic-error.md)
  expose the tail that the mean alone hides.
- The trimmed mean is a labelled sensitivity statistic, never a substitute for
  the proper score.

Next (Milestone 3): comparing two algorithms on the same individuals with paired
bootstrap confidence intervals.
```
