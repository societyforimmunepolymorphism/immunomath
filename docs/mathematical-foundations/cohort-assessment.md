# Cohort Assessment

Milestone 2 evaluates *collections* of predictions. A
[`Cohort`][immunomath.Cohort] pairs stable subject identifiers with observed
truths and predicted distributions; [`summarize`][immunomath.cohort.Cohort.summarize]
returns a [`CohortSummary`][immunomath.CohortSummary] with explicit denominators.

## The three-way classification of a subject

Every subject is classified into exactly one of three mutually exclusive,
jointly exhaustive cases, determined by whether the truth's probability is
**identifiable** (see the
[probability distributions](probability-distributions.md) page):

| Classification | Condition | Log score |
| --- | --- | --- |
| `finite` | truth has identifiable, strictly positive probability | a finite real number |
| `infinite` | truth has identifiable **zero** probability | `+inf` |
| `undefined` | truth's probability is **not identifiable** (open-world residual) | does not exist |

The classification is performed by
[`score_cohort_accounting`][immunomath.score_cohort_accounting], and the
per-subject result is a [`SubjectScore`][immunomath.SubjectScore].

!!! important "Why `undefined` is not `infinite`"
    A subject whose truth might be hidden in residual mass is **not** the same
    as a subject whose truth was assigned zero probability. The former has *no
    identifiable probability at all*; treating it as zero would fabricate an
    infinite penalty the forecaster never earned. ImmunoMath keeps these cases
    strictly separate.

## Explicit denominators

The summary reports four counts so no observation is ever silently dropped:

- `n_total` — all subjects.
- `n_scored` — identifiable subjects (`finite` + `infinite`). **This is the
  denominator of the primary mean.**
- `n_undefined` — subjects excluded from the mean, each with a recorded reason.
- `n_infinite` — identifiable subjects with infinite loss.

A convenience `n_finite = n_scored − n_infinite` is also provided.

## The primary mean is a proper score over identifiable subjects

$$
\overline{L}_{\text{identifiable}}
  = \frac{1}{n_\text{scored}} \sum_{i \,\in\, \text{identifiable}} L_i .
$$

- If **any** identifiable subject has infinite loss, the primary mean is
  `+inf`. Infinite losses are never discarded — that is the whole point of a
  proper score.
- If **every** subject is undefined (`n_scored == 0`), the mean is `nan`,
  reported explicitly rather than hidden.
- When undefined subjects are present, the mean is labelled as applying to the
  identifiable subpopulation; the undefined rows are counted, not averaged.

## Worked example

```python
>>> from immunomath import Cohort, Distribution, Residual
>>> cohort = Cohort(
...     subject_ids=["S1", "S2", "S3", "S4"],
...     truths=["G1", "G2", "G2", "G99"],
...     distributions=[
...         Distribution({"G1": 0.9, "G2": 0.1}),       # finite
...         Distribution({"G1": 0.999, "G2": 0.001}),   # finite (bad)
...         Distribution({"G1": 1.0, "G2": 0.0}),        # infinite (zero on truth)
...         Distribution({"G1": 0.5, Residual(0.5): None}),  # undefined
...     ],
... )
>>> summary = cohort.summarize(seed=0, n_resamples=500)
>>> (summary.n_total, summary.n_scored, summary.n_undefined, summary.n_infinite)
(4, 3, 1, 1)
>>> summary.mean_log_score            # +inf: an identifiable zero is present
inf
>>> summary.undefined_reasons[0][0]   # the excluded subject id
'S4'
```

## What the summary reports

| Field | Meaning |
| --- | --- |
| `mean_log_score` | primary proper-score summary over identifiable subjects |
| `median_log_score`, `std_log_score` | dispersion over **finite** subjects |
| `max_log_score` | `+inf` if any identifiable subject is infinite |
| `loss_quantiles` | quantiles of the identifiable loss distribution (see [quantiles](quantiles.md)) |
| `mean_brier` | mean Brier score over identifiable subjects (bounded, always finite) |
| `mean_log_score_ci` | bootstrap CI for the mean (see [uncertainty](uncertainty.md)) |
| `subject_scores` | per-subject classification and score |
| `undefined_reasons` | `(subject_id, reason)` for each undefined subject |

## Strict mode

When a cohort should contain no unidentifiable truths, pass `strict=True` to
reject the cohort instead of excluding undefined subjects:

```python
>>> cohort.summarize(strict=True)   # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
ValueError: strict mode: cohort contains 1 undefined subject(s) ...
```

Invalid input data (mismatched lengths, duplicate subject ids, non-distribution
predictions) always raises regardless of strict mode.

## Assumptions and failure modes

- Subject identifiers must be unique; they anchor the per-subject resampling and
  diagnostic tables.
- The Brier mean remains finite even when the log-score mean is infinite,
  because the Brier score is bounded. The two are complementary, not
  interchangeable.
- Dispersion statistics (median, std) describe the finite subpopulation and must
  not be read as the proper-score summary.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules.
  *JASA*, 102(477), 359–378.
