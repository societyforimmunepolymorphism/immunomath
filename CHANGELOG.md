# Changelog

All notable changes to ImmunoMath are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — TRS-aware weighted scoring

### Added

- **Observation-weighted proper scoring** (`weighted_mean_loss`,
  `WeightedScore`): a covariate-weighted mean of the existing log or Brier
  per-observation losses, $L_w = \sum_i w_i L_i / \sum_i w_i$, with the
  unweighted mean always reported alongside. Weights are externally supplied and
  validated (finite, strictly positive, correct length) but never normalized or
  clipped. Reports sample count, total weight, and the Kish effective sample
  size $N_\mathrm{eff} = (\sum w)^2 / \sum w^2$. Uniform (or `None`) weights
  reproduce ordinary scoring exactly.
- **Paired model comparison** (`compare_methods_weighted`,
  `PairedWeightedComparison`): compares an incumbent and a candidate on
  identical observations, truths and weights. Reports conventional and weighted
  mean loss for each method, conventional and weighted mean improvement
  ($\Delta_i = L_\mathrm{incumbent} - L_\mathrm{candidate}$; positive favors the
  candidate), improved/worsened/unchanged counts, and the full distribution of
  per-observation changes. Undefined-in-either and indeterminate $\infty-\infty$
  observations are excluded and reported; one-sided infinities are retained.
- **TRS-stratified validation** (`stratified_comparison`, `StratifiedComparison`,
  `StratumComparison`): per-stratum mean log/Brier scores and paired improvement,
  plus a **group-balanced macro-average** $L_\mathrm{balanced} = \frac1K \sum_k
  \bar L_k$ over nonempty strata, reported alongside the conventional
  donor-weighted cohort mean. Empty declared strata are listed and excluded from
  the macro denominator.
- **Shared loss front end** (`per_observation_losses`, `ObservationLoss`,
  `LossMetric`, `validate_weights`): reuses `negative_log_score` and
  `brier_score`, so residual/zero-probability semantics are defined in one place.
- **Optional TRS → weight adapter** (`immunomath.trs.trs_weights`): the
  domain-specific mapping $w_i = 1 + \lambda (1 - r_i)^\gamma$ for normalized
  $r_i \in [0, 1]$, $\lambda \ge 0$, $\gamma > 0$, kept **outside** the
  statistical kernel. Parameters are explicit (no scientific default); the TRS
  scale is not silently normalized.
- Documentation: a mathematical-foundations page (conditional propriety, the
  weighting function, empty-strata and infinity handling) and an API page, plus
  the worked example `examples/trs_weighted_comparison.py`.

### Notes

- Fully backward compatible: no existing public API, behavior, or serialization
  changed. The core remains free of biological dependencies; no HLA parsing,
  haplotype-frequency estimation, or donor-ranking logic was added.

## [0.4.0] — Milestone 4: calibration and reporting

### Added

- **Top-label calibration** (`top_label_calibration`) and **event-level
  calibration** (`event_calibration`) with a calibration table per bin
  (count, mean predicted, observed rate, Wilson interval), **ECE** (weighted by
  bin count) and **MCE**. Fixed-width (default) and equal-frequency binning,
  10 bins by default; the method is recorded in every result. Event calibration
  also reports the event Brier score and excludes subjects whose event
  probability is not identifiable (residual mass), counting them.
- **Risk-coverage analysis with abstention** (`risk_coverage`): user-defined
  confidence score (callable, per-subject mapping, or the top probability by
  default), explicit abstention, and **both accuracy and mean log loss** reported
  against coverage. Tied scores are retained together, so the curve is
  order-invariant.
- **Analysis record** (`analyze_cohort`, `AnalysisResult`): one serializable,
  schema-versioned record of the cohort, tail, calibration, event and
  risk-coverage results with every parameter. The library never stamps the clock.
- **`immunomath.reporting`** (optional extra `reporting`: Plotly + PyArrow):
  lossless JSON export (non-finite floats are tagged, not nulled), CSV/Parquet
  tables, Plotly reliability / risk-coverage / loss-distribution figures, and a
  **self-contained HTML report** with metadata, denominators, exclusions, methods,
  assumptions and generated warnings. Reports are rebuilt from the saved JSON
  without re-running anything. All dynamic text is HTML-escaped.
- **Documentation**: calibration and risk-coverage foundations pages, a reporting
  user guide, a reproducibility checklist, Tutorial 6 (calibration diagram) and
  Tutorial 7 (allele-presence probabilities), API pages, and a test that executes
  every `>>>` example in the Markdown docs.

### Fixed

- `bootstrap_ci(method="bca")` imported SciPy, which is not a runtime
  dependency, so BCa failed on a plain `pip install immunomath`. It now uses the
  standard library (`statistics.NormalDist`); results are unchanged.
- `getting-started.md` showed unrounded floats (`0.065`) that the code does not
  actually print; the examples are now rounded and executed in CI.

### Changed

- Version is now defined in `immunomath/_version.py` (no circular import).
- Version `0.4.0`.

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
