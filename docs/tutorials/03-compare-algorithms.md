# Tutorial 3 — Comparing Two Algorithms on Identical Individuals

This tutorial performs a **paired comparison** of two algorithms, A and B,
evaluated on the same 500 synthetic subjects. It shows why ImmunoMath reports
*two* headline summaries — the mean difference and the proportion improved — and
why they can tell different stories. All data is synthetic and reproducible.

## Build two aligned cohorts

B is a baseline; A nudges probability mass toward the truth by a *random* amount
(positive on average, but sometimes negative), so A is better overall yet not
uniformly better.

```python
import numpy as np
from immunomath import Cohort, Distribution, compare_cohorts

rng = np.random.default_rng(31415)
N = 500
candidates = [f"G{i}" for i in range(4)]

ids, truths, rows_a, rows_b = [], [], [], []
for i in range(N):
    truth = candidates[int(rng.integers(0, 4))]
    ti = candidates.index(truth)
    base = rng.normal(0.0, 1.0, size=4)
    p_b = np.exp(base)
    p_b /= p_b.sum()
    nudge = rng.normal(0.4, 0.8)  # mean +0.4, frequently negative
    a = base.copy()
    a[ti] += nudge
    p_a = np.exp(a)
    p_a /= p_a.sum()
    ids.append(f"s{i:03d}")
    truths.append(truth)
    rows_a.append(Distribution({c: float(x) for c, x in zip(candidates, p_a)}))
    rows_b.append(Distribution({c: float(x) for c, x in zip(candidates, p_b)}))

cohort_a = Cohort(ids, truths, rows_a)
cohort_b = Cohort(ids, truths, rows_b)
```

## Run the comparison

```python
cmp = compare_cohorts(cohort_a, cohort_b, label_a="A", label_b="B", seed=0, n_resamples=2000)
```

Both cohorts share identical subject ids and the same outcome space `{G0..G3}`,
so the default `on="exact"` policy and the strict outcome-space check both pass.

## Headline 1 — mean paired difference

```text
mean_log_loss_diff    ≈ -0.171 nats
mean_log_loss_diff_ci ≈ (-0.222, -0.122)    # 95% paired bootstrap, seed=0
```

The mean difference is negative and its 95% confidence interval lies **entirely
below zero**: on average, A assigns the truth higher probability than B, and the
effect is statistically distinguishable from zero. (Recall the sign convention:
$d = L_A - L_B$, negative favors A.)

## Headline 2 — proportion improved vs worsened

```text
n_improved = 329   (65.8%)
n_worsened = 171   (34.2%)
n_tied     =   0
```

Here is the lesson: although A wins *on average*, it only improves about
**two-thirds** of subjects and actively **worsens a third**. A single mean would
hide that a substantial minority is hurt by switching to A. Reporting both
headlines keeps that tradeoff visible.

## Per-subject diagnostics

```text
largest_improvement ≈ ("s076", -2.59)   # A dramatically better here
largest_worsening   ≈ ("s305", +2.12)   # A dramatically worse here
concordance_diff_top1 ≈ +0.112          # A's top-1 accuracy exceeds B's by ~11 pts
```

`subject_differences` holds every $(\text{id}, d_i)$ pair, ready for a paired
scatter plot or a histogram of per-subject differences.

## What you learned

- [`compare_cohorts`][immunomath.compare_cohorts] pairs two cohorts by subject
  and reports the **mean difference with a paired bootstrap CI** *and* the
  **improved/worsened proportions** as co-equal headlines.
- The two can disagree in emphasis: a favorable mean can coexist with many
  worsened subjects. Read both.
- The sign convention is $d = L_A - L_B$; negative favors A.

Next: comparing a population-specific panel against a global panel, including a
resolution change handled by an explicit projection.
```
