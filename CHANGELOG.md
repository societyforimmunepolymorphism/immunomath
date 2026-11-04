# Changelog

All notable changes to ImmunoMath are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — Milestone 3: comparative assessment

### Added

- **`compare_cohorts`** and **`PairedComparison`** (`immunomath.comparison`) —
  paired comparison of two cohorts (algorithm A vs B, reference panel A vs B,
  software versions) on the **same subjects**, with two co-equal headline
  summaries:
  - the **mean paired log-loss difference** (`d = L_A - L_B`, negative favors A)
    with a raw **paired bootstrap confidence interval** (no multiplicity
    correction), and
  - the **proportion improved vs worsened vs tied**, with counts and
    proportions.
- Per-subject diagnostics: `subject_differences`, `largest_improvement` /
  `largest_worsening` (or `None` when none), and a descriptive
  `concordance_diff_top1` (top-1 accuracy contrast).
- **Subject-overlap policy**: `on="exact"` (default) requires identical
  subject-id sets and raises on any mismatch; `on="intersection"` restricts to
  common subjects. A subject with differing truths across arms raises.
- **Strict common-outcome-space enforcement**: comparisons over different
  outcome spaces (e.g. different genotype resolutions) are **refused** unless a
  reconciling `projection` is supplied, which is applied to both cohorts and
  their truths (mass-preserving).
- **Explicit infinite/undefined handling** in paired differences: one-sided
  infinite losses are retained as decisive `±inf` differences; indeterminate
  `inf − inf` pairs and subjects undefined in either arm are excluded with
  recorded reasons and counted in `n_both_undefined_or_infinite`.
- **Documentation**: a comparative-assessment mathematical-foundations page and
  two executable tutorials — Tutorial 3 (comparing two algorithms) and Tutorial
  4 (population-specific vs global panel, including a projection-based
  resolution change) — plus a new API reference page and navigation entries.
- **Tests**: paired-difference correctness and sign convention, both headlines,
  overlap policy and `intersection` opt-in, truth-mismatch errors, strict
  outcome-space enforcement and projection reconciliation, infinite/undefined
  handling, bootstrap determinism, and property-based antisymmetry and
  self-comparison invariants.

### Changed

- `__version__` and the packaged version bumped to `0.3.0`; the public API now
  exports `compare_cohorts` and `PairedComparison`.

## [0.2.0] — Milestone 2: cohort assessment

### Added

- **`Cohort`** and **`CohortSummary`** (`immunomath.cohort`) — a
  subject-identified cohort container and a summary with **explicit
  denominators** (`n_total`, `n_scored`, `n_undefined`, `n_infinite`,
  `n_finite`), dispersion statistics, loss quantiles, mean Brier, and a
  bootstrap confidence interval.
- **Finite / infinite / undefined accounting** (`score_cohort_accounting`,
  `CohortAccounting`, `SubjectScore`, `ScoreClass`) — classifies each subject by
  whether its truth probability is identifiable-positive (finite),
  identifiable-zero (infinite `+inf`), or unidentifiable (undefined). The
  primary mean is a proper score over identifiable subjects, includes infinite
  losses, and excludes undefined subjects with recorded reasons. An optional
  **strict mode** rejects cohorts containing undefined subjects.
- **Bootstrap confidence intervals** (`immunomath.statistics`): seeded,
  reproducible subject-level `bootstrap_ci` (percentile default, BCa optional;
  2000 resamples, 95% by default) and `paired_bootstrap_ci` for comparing two
  measurements on the same individuals. Infinite losses are never laundered into
  misleading finite intervals; affected bounds are reported as `+inf` with an
  infinite-resample count.
- **Loss-distribution quantiles** (`immunomath.quantiles`): `loss_quantile` /
  `loss_quantiles` with a configurable convention (`QuantileMethod`, default
  Hyndman-Fan type 7) and explicit infinite-value handling (upper-tail
  quantiles return `+inf`, never `nan`).
- **Catastrophic-error analysis** (`immunomath.catastrophic`):
  `catastrophic_error_report` reporting the fraction of subjects below
  configurable `P(truth)` thresholds (default `0.01`, `0.001`, `0.0001`), with
  **exactly-zero probability as a separate category**, worst-subject tables,
  loss-contribution concentration, and an **optional** 1% upper-tail trimmed
  mean offered strictly as a descriptive sensitivity statistic — never a
  replacement for the proper-score mean.
- **Documentation**: four new mathematical-foundations pages (cohort assessment,
  uncertainty/bootstrap, loss-distribution quantiles, catastrophic-error
  analysis), Tutorial 2 (evaluating 1,000 simulated individuals), and new API
  reference pages. Navigation and the trimmed-mean/proper-score distinction are
  documented throughout.
- **Tests**: cohort classification and denominators, strict mode, bootstrap
  determinism and coverage, BCa, quantile cross-checks against NumPy,
  catastrophic thresholds and the zero category, trimmed-mean behavior, explicit
  infinite handling, and property-based invariants.

### Changed

- `__version__` bumped to `0.2.0`; the public API in `immunomath.__init__`
  now exports the Milestone 2 surface.
- Statistical dependencies: `scipy` is used for the BCa normal quantiles (a
  scoped mypy override allows its untyped import without relaxing strictness
  elsewhere).

## [0.1.0] — Milestone 1: the mathematical kernel

### Added

- **`Distribution`** — a validated, immutable finite discrete probability
  distribution over hashable outcomes (strings, integers, tuples, genotype
  identifiers). Rejects non-finite, negative, or unnormalized inputs; never
  silently renormalizes.
  - Explicit three-way mass distinction: omitted outcome, explicit zero, and
    unknown residual mass via the **`Residual`** sentinel.
  - `probability()` with closed-/open-world semantics and identifiability
    checks; `event_probability()` for identifiable event aggregation.
  - Opt-in `renormalize()`; mass-preserving `project()` with explicit
    complete-vs-partial handling.
  - JSON/dict serialization that round-trips strings, ints, and tuples.
- **`Projection`** — a general-purpose, HLA-free protocol for mapping fine
  outcomes onto coarser equivalence classes.
- **Proper scoring rules** (`immunomath.scoring`):
  - `negative_log_score` (natural log, nats; `+inf` for zero-probability truth;
    no implicit clipping; opt-in `floor`).
  - `brier_score` (multiclass sum form, range `[0, 2]`).
  - `mean_negative_log_score` and `mean_brier_score` cohort means, with
    `CohortScore` reporting infinite-loss counts.
- **Descriptive diagnostics** (`immunomath.diagnostics`):
  - `top_k_accuracy` with order-invariant fractional-credit tie handling.
  - `rank_of_truth` reporting a `(min_rank, max_rank)` interval for ties.
  - `probability_of_truth`, `entropy`, `effective_number_of_outcomes`,
    `num_candidates`.
- **Documentation site** (MkDocs + Material): mathematical-foundations pages
  (distributions, logarithmic score, Brier score, proper-vs-descriptive),
  getting-started, architecture/design rationale, tutorial 1, and auto-generated
  API reference.
- **Test suite**: edge-case coverage from the testing checklist, property-based
  tests (Hypothesis), and cross-checks against scikit-learn (`log_loss`) and
  SciPy (`entropy`).

### Notes

- Licensed under **Apache-2.0**.
- Milestones 2–5 (cohort assessment, comparison, calibration/reporting, HLA
  integration) are intentionally out of scope for this release.
