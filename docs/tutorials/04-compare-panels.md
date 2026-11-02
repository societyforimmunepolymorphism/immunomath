# Tutorial 4 — Population-Specific vs Global Reference Panel

This tutorial compares a **population-specific** reference panel against a
**global** panel on the same 300 subjects, and then demonstrates the strict
common-outcome-space rule by re-comparing at a coarser resolution using an
explicit projection. All data is synthetic and reproducible.

## Build two aligned cohorts over two-field alleles

The population-specific panel concentrates probability more sharply around the
truth than the global panel.

```python
import numpy as np
from immunomath import Cohort, Distribution, compare_cohorts

rng = np.random.default_rng(2718)
M = 300
alleles = ["A*01:01", "A*01:02", "A*02:01", "A*02:02"]

ids, truths, rows_pop, rows_glob = [], [], [], []
for i in range(M):
    truth = alleles[int(rng.integers(0, 4))]
    ti = alleles.index(truth)
    g = rng.normal(0.0, 0.8, size=4)
    p_glob = np.exp(g)
    p_glob /= p_glob.sum()
    pop = g.copy()
    pop[ti] += 0.6  # population panel sharper
    p_pop = np.exp(pop)
    p_pop /= p_pop.sum()
    ids.append(f"p{i:03d}")
    truths.append(truth)
    rows_pop.append(Distribution({c: float(x) for c, x in zip(alleles, p_pop)}))
    rows_glob.append(Distribution({c: float(x) for c, x in zip(alleles, p_glob)}))

population = Cohort(ids, truths, rows_pop)
global_panel = Cohort(ids, truths, rows_glob)
```

## Compare at two-field resolution

```python
cmp2 = compare_cohorts(
    population, global_panel, label_a="population", label_b="global", seed=1, n_resamples=2000
)
```

```text
mean_log_loss_diff  ≈ -0.417 nats   (negative favors the population panel)
proportion_improved ≈ 1.000
```

The population-specific panel is better here by both headlines: a clearly
negative mean difference and improvement for essentially all subjects.

## Changing resolution requires an explicit projection

Suppose we instead want to compare at **one-field** resolution (collapsing
`A*01:01` and `A*01:02` into `A*01`, etc.). We must *not* mix resolutions
silently. If the two cohorts were defined over different spaces,
`compare_cohorts` would refuse. To evaluate the induced one-field distributions,
pass a `projection`; it is applied to both cohorts and to their truths, and the
projection is mass-preserving:

```python
one_field = lambda a: a.split(":")[0]

cmp1 = compare_cohorts(
    population,
    global_panel,
    label_a="population",
    label_b="global",
    projection=one_field,
    seed=1,
    n_resamples=2000,
)
```

```text
n_paired            = 300
mean_log_loss_diff  ≈ -0.150 nats
proportion_improved ≈ 1.000
```

The population panel still wins at one-field resolution, but by a **smaller**
margin (−0.150 vs −0.417 nats). This is expected: collapsing to one field
discards the two-field distinctions where the population panel had its sharpest
advantage. Evaluating at the resolution that matters for your application is a
scientific choice — and ImmunoMath forces you to make it explicitly rather than
comparing incommensurable numbers.

## The strict rule in action

If you attempt to compare cohorts whose outcome spaces differ *without* a
projection, the comparison raises:

```python
from immunomath import Cohort, Distribution, compare_cohorts

two_field = Cohort(["x"], ["A*01:01"], [Distribution({"A*01:01": 0.7, "A*02:01": 0.3})])
one_fielded = Cohort(["x"], ["A*01"], [Distribution({"A*01": 0.7, "A*02": 0.3})])

# Raises ValueError: different outcome spaces — supply a projection.
# compare_cohorts(two_field, one_fielded)
```

## What you learned

- Paired comparison extends directly to **reference-panel** comparisons.
- A **resolution change** must be reconciled with an explicit `projection`;
  ImmunoMath refuses to compare different outcome spaces silently.
- Projecting to a coarser space is mass-preserving but discards fine-grained
  signal, which typically shrinks a panel's measured advantage — a result to
  interpret, not an artifact to hide.

Next (Milestone 4): calibration curves, event-level calibration, risk–coverage
analysis, and reproducible reports.
```
