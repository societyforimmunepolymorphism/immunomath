# Changelog

All notable changes to ImmunoMath are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
