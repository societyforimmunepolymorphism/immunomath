# Catastrophic-Error Analysis

Because the negative log score penalizes small probabilities on the truth
*without bound*, a handful of subjects can dominate the cohort mean.
[`catastrophic_error_report`][immunomath.catastrophic_error_report] finds and
describes those subjects — it never trims or excludes them from the primary
proper-score summary.

## Why the tail matters

If one subject in a thousand is assigned probability $10^{-6}$ on the truth, its
log loss is $\approx 13.8$ nats. A single such subject can raise a cohort mean
that is otherwise near $0.3$ by over an order of magnitude. This sensitivity is
a *feature* of a proper scoring rule — it is what discourages overconfidence —
but it means the mean must be read alongside tail diagnostics.

## Low-probability fractions

The report counts the fraction of **identifiable** subjects whose true-outcome
probability falls below each of a set of thresholds. The scientific defaults are

$$
P(\text{truth}) < 0.01, \quad < 0.001, \quad < 0.0001,
$$

and the thresholds are configurable. Because $P = e^{-L}$, "$P < t$" is
equivalent to "$L > -\ln t$", which is how the counts are computed for finite
subjects.

### Exactly-zero probability is a separate category

A subject assigned **exactly** zero probability on the truth (infinite loss) is
reported separately as
[`n_zero_probability`][immunomath.CatastrophicReport], not folded anonymously
into the smallest threshold. It is also included in each below-threshold count
(since $0 < t$ for every threshold), so the fractions remain honest "below
threshold" measures, while the zero category stays visible.

```python
>>> from immunomath import Cohort, Distribution, catastrophic_error_report
>>> cohort = Cohort(
...     ["S1", "S2", "S3", "S4"],
...     ["G1", "G2", "G2", "G2"],
...     [
...         Distribution({"G1": 0.9, "G2": 0.1}),       # P(truth)=0.1
...         Distribution({"G1": 0.995, "G2": 0.005}),   # P(truth)=0.005
...         Distribution({"G1": 0.9995, "G2": 0.0005}), # P(truth)=0.0005
...         Distribution({"G1": 1.0, "G2": 0.0}),        # P(truth)=0  (zero category)
...     ],
... )
>>> report = catastrophic_error_report(cohort)
>>> report.n_zero_probability
1
>>> {tc.threshold: tc.count for tc in report.threshold_counts}
{0.01: 3, 0.001: 2, 0.0001: 1}
```

The denominator of every fraction is `n_identifiable` (finite + infinite);
undefined subjects are excluded and reported through the cohort summary, never
counted as zero-probability.

## Worst subjects and contribution

The report also lists the worst individual losses and quantifies how
concentrated the total loss is:

- `worst_subjects` — `(subject_id, loss)` pairs, largest first.
- `loss_quantiles` — quantiles of the individual-loss distribution (upper
  quantiles are `+inf` when zero-probability subjects exist).
- `total_loss` — the summed loss (`+inf` if any subject is infinite).
- `top_contributor_fraction` — the share of the (finite) total contributed by
  the single worst subject; `nan` when the total is infinite.

These support subject-level diagnostic tables and subgroup-enrichment analysis
without altering the proper score.

## The trimmed mean is descriptive, never a replacement

As an **optional sensitivity analysis**, the report can include an upper-tail
trimmed mean (default 1%):

```python
>>> report = catastrophic_error_report(cohort, include_trimmed_mean=True,
...                                     trimmed_fraction=0.01)
>>> report.trimmed_fraction
0.01
```

!!! danger "Do not substitute the trimmed mean for the proper score"
    Trimming discards exactly the catastrophic cases a proper scoring rule is
    designed to penalize. The trimmed mean is useful to *diagnose* how
    sensitive the mean is to the worst subjects, but it is **not** a proper
    scoring rule. The report always preserves
    [`primary_mean_log_score`][immunomath.CatastrophicReport] — the untrimmed
    mean — as the headline figure, and it remains `+inf` when a zero-probability
    subject is present even when the trimmed mean is finite.

Robust descriptive statistics aid diagnosis; they are not interchangeable with
proper scores. This distinction is the same one drawn in
[Proper Scoring vs. Descriptive Metrics](proper-vs-descriptive.md).

## Design commitments

- **No automatic trimming or exclusion.** Rare alleles and outliers are never
  dropped by default; the primary mean sees every identifiable subject.
- **Zero probability stays visible.** It is a distinct category, not a quietly
  large finite number.
- **Thresholds are explicit and configurable**, so a report always states the
  cutoffs it used.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules.
  *JASA*, 102(477), 359–378.
- Huber, P. J., & Ronchetti, E. M. (2009). *Robust Statistics* (2nd ed.). Wiley.
  (On trimmed means as robust *descriptive* statistics.)
