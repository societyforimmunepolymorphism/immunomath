# Comparative Assessment

Milestone 3 compares two evaluations on the **same individuals** — algorithm A
versus B, reference panel A versus B, or two software versions.
[`compare_cohorts`][immunomath.compare_cohorts] returns a
[`PairedComparison`][immunomath.PairedComparison] with two co-equal headline
summaries.

## The paired difference

For each subject $i$ present in both arms, the comparison statistic is the
difference in negative log loss:

$$
d_i = L_A(i) - L_B(i) = -\ln P_A\!\left(g_i\right) + \ln P_B\!\left(g_i\right).
$$

**Sign convention.** A **negative** $d_i$ means arm A assigned the truth higher
probability (lower loss) than arm B — *A is better for subject $i$*. The same
convention applies to the mean.

Pairing is what gives the comparison its power: because both arms score the
*same* subject against the *same* truth, subject-level difficulty cancels, and
the difference isolates the effect of changing the algorithm or panel.

## Two co-equal headlines

### Headline 1 — mean paired difference (a proper-score contrast)

$$
\overline{d} = \frac{1}{n_\text{paired}} \sum_i d_i ,
$$

reported with a **paired bootstrap confidence interval** (see
[Uncertainty](uncertainty.md)). Because it is built from a strictly proper
scoring rule, $\overline{d}$ rewards honest probability, not just ranking. A
confidence interval lying entirely below zero is evidence that A is better.

### Headline 2 — proportion improved versus worsened

Counts of subjects for which A improves on B, worsens, or ties:

$$
n_\text{improved} = \#\{ i : d_i < -\tau \}, \quad
n_\text{worsened} = \#\{ i : d_i > \tau \}, \quad
n_\text{tied} = \#\{ i : |d_i| \le \tau \},
$$

with a small tie tolerance $\tau$ (default $10^{-12}$), reported both as counts
and as proportions of $n_\text{paired}$.

!!! note "Why both headlines"
    The mean and the proportion answer different questions. The mean can be
    dominated by a few subjects with large differences; the proportion shows how
    *broadly* one arm wins. A method might improve the mean while helping only a
    minority of subjects (a few big wins), or help most subjects only slightly.
    ImmunoMath reports both so neither story can hide the other.

## Worked example

```python
>>> from immunomath import Cohort, Distribution, compare_cohorts
>>> a = Cohort(["s1", "s2"], ["G1", "G2"],
...            [Distribution({"G1": 0.8, "G2": 0.2}),
...             Distribution({"G1": 0.3, "G2": 0.7})])
>>> b = Cohort(["s1", "s2"], ["G1", "G2"],
...            [Distribution({"G1": 0.6, "G2": 0.4}),
...             Distribution({"G1": 0.5, "G2": 0.5})])
>>> cmp = compare_cohorts(a, b, label_a="A", label_b="B", seed=0, n_resamples=2000)
>>> cmp.mean_log_loss_diff < 0      # A assigns the truth higher probability
True
>>> (cmp.n_improved, cmp.n_worsened, cmp.n_tied)
(2, 0, 0)
```

## Same subjects: overlap policy

Subjects are aligned by identifier.

- `on="exact"` (default): the subject-id sets must be **identical**; any
  mismatch raises. This is the safe default — a silent subset comparison can
  badly mislead.
- `on="intersection"`: restrict to the subjects present in both arms, reporting
  `n_common`.

A subject with **different truths** in the two arms raises an error: it cannot
be the same individual observed two ways.

## Strict common outcome space

Two evaluations at different genotype resolutions produce probabilities over
*different* outcome spaces; their log losses are not directly comparable.
`compare_cohorts` therefore **refuses** to proceed when the cohorts' outcome
universes differ:

```python
>>> a2 = Cohort(["s1"], ["A*01:01"], [Distribution({"A*01:01": 0.7, "A*01:02": 0.3})])
>>> b2 = Cohort(["s1"], ["A*01"], [Distribution({"A*01": 0.7, "A*02": 0.3})])
>>> compare_cohorts(a2, b2)                      # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
ValueError: The two cohorts are defined over different outcome spaces ...
```

Reconcile explicitly by supplying a `projection` that maps both cohorts (and
their truths) onto a shared coarser space (see
[Probability Distributions](probability-distributions.md) on mass-preserving
projection):

```python
>>> a3 = Cohort(["s1", "s2"], ["A*01:01", "A*02:01"],
...             [Distribution({"A*01:01": 0.7, "A*02:01": 0.3}),
...              Distribution({"A*01:01": 0.4, "A*02:01": 0.6})])
>>> b3 = Cohort(["s1", "s2"], ["A*01:01", "A*02:01"],
...             [Distribution({"A*01:01": 0.5, "A*02:01": 0.5}),
...              Distribution({"A*01:01": 0.5, "A*02:01": 0.5})])
>>> cmp = compare_cohorts(a3, b3, projection=lambda x: x.split(":")[0],
...                       seed=0, n_resamples=500)
>>> cmp.n_paired
2
```

## Infinite and undefined differences

A paired difference is defined only when **both** arms yield an identifiable
score. ImmunoMath handles the problem cases explicitly, excluding them from the
paired statistics and reporting them under
[`n_both_undefined_or_infinite`][immunomath.PairedComparison] with reasons:

| Situation | Treatment |
| --- | --- |
| both arms finite | paired, $d_i$ computed |
| one arm $+\infty$, other finite | retained; $d_i = \pm\infty$ (a decisive win/loss) |
| **both** arms $+\infty$ | excluded — $\infty - \infty$ is indeterminate |
| truth undefined in either arm | excluded with reason |

This mirrors the paired-bootstrap rule that an `inf − inf` difference is
indeterminate and must be resolved, never silently set to zero.

## Other reported diagnostics

- `largest_improvement` / `largest_worsening` — the subjects with the most
  negative / most positive $d_i$ (or `None` if none improved / worsened).
- `subject_differences` — the full per-subject $(\text{id}, d_i)$ list, for
  scatter and per-subject difference plots.
- `concordance_diff_top1` — $\text{acc}_A - \text{acc}_B$ using fractional
  top-1 accuracy (positive favors A). A *descriptive* contrast, complementing
  the proper-score headline.

## No multiplicity correction

Confidence intervals are **raw** paired bootstrap intervals. ImmunoMath does not
apply any multiple-comparison adjustment; if you compare many algorithms or
panels, account for multiplicity in your own analysis plan.

## Assumptions and failure modes

- The comparison assumes genuine pairing (same individuals, same truths). It
  does not detect label leakage or non-independent training/validation splits.
- When many subjects fall into the `inf − inf` or undefined categories, the
  paired denominator shrinks; always read `n_paired` alongside the headlines.
- The mean difference can be $\pm\infty$ when some subjects have one-sided
  infinite loss; this is reported honestly rather than hidden.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules.
  *JASA*, 102(477), 359–378.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall. (Paired bootstrap.)
