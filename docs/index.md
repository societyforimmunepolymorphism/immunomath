# ImmunoMath

**Metrognome — keeping probabilistic genetics honest.**

ImmunoMath is an open-source Python library, maintained by the Society for
Immune Polymorphism (SIP), for **evaluating** probabilistic genetic inference.
It provides rigorous, reproducible methods for assessing the accuracy,
calibration, and reliability of probability distributions produced by inference
engines — initially HLA genotype imputation, but applicable to probabilistic
classification in general.

## ImmunoMath is an evaluation library, not an inference library

ImmunoMath does **not** generate synthetic genotypes, simulate laboratory
typing, estimate haplotype frequencies, or perform imputation. Those
responsibilities belong to independent packages:

| Package | Responsibility |
| --- | --- |
| **DiplotypeForge** | Generate synthetic, phase-resolved genetic truth. |
| **TissueTypeForge** | Simulate observed typing information and ambiguity. |
| **py-graph-imputation** and others | Generate candidate genotypes and probabilities. |
| **ImmunoMath** | Compare predicted probability distributions against independent truth. |

This separation keeps each package independently usable and keeps ImmunoMath's
statistical core free of any dependence on HLA nomenclature, GL Strings, or a
particular imputation algorithm.

## The fundamental contract

The statistical core accepts:

1. A known outcome (or an appropriately represented uncertain observation).
2. A **normalized** probability distribution over possible outcomes.
3. An optional equivalence or projection function defining the outcome space
   being evaluated.

ImmunoMath **never silently normalizes** an incomplete or invalid distribution.

## This release: Milestone 1 — the mathematical kernel

- Validated discrete probability distributions ([`Distribution`][immunomath.Distribution])
- Correct-outcome probability extraction, with explicit residual-mass semantics
- Negative logarithmic score ([`negative_log_score`][immunomath.negative_log_score])
- Multiclass Brier score ([`brier_score`][immunomath.brier_score])
- Top-1 / top-K accuracy, rank of truth, entropy, effective number of outcomes
- A general-purpose [`Projection`][immunomath.Projection] protocol for outcome
  equivalence (HLA-free in the kernel)
- A comprehensive, cross-validated test suite and this documentation site

Start with [Getting Started](getting-started.md), or read the
[Mathematical Foundations](mathematical-foundations/probability-distributions.md).

## License

ImmunoMath is licensed under the **Apache License 2.0**.
