# Architecture and Design

This page explains the design decisions behind ImmunoMath. These are
deliberate, load-bearing choices, not accidents of implementation.

## Why ImmunoMath does not perform imputation

ImmunoMath is an **evaluation** library. Mixing inference and evaluation in one
package creates a conflict of interest: an evaluator that also generates
predictions is tempted to tune its metrics to flatter its own method. By
accepting predictions from *any* external engine and comparing them against
*independent* truth, ImmunoMath can serve as a neutral referee suitable for
benchmarking and, eventually, clinical validation.

Generation of truth, simulation of typing, and imputation are delegated to
independent packages (DiplotypeForge, TissueTypeForge, py-graph-imputation, and
others), each independently usable.

## Why the normalized distribution is the foundational abstraction

Every metric in ImmunoMath operates on a single abstraction — the
[`Distribution`][immunomath.Distribution] — rather than on raw arrays or
dictionaries. Centralizing validation there means:

- No downstream metric ever sees a negative, non-finite, or unnormalized
  probability vector.
- The "never silently normalize" guarantee is enforced in exactly one place.
- New scoring rules can be added without re-implementing validation, and
  without changing the distribution API.

A distribution is validated at construction and is immutable thereafter.

## Why normalization is explicit, never automatic

Silently renormalizing a distribution that does not sum to one hides bugs:
a truncated or mis-exported prediction would be "repaired" into something that
looks valid but misrepresents the inference engine's actual output. ImmunoMath
therefore **rejects** such inputs. Renormalization is available only through the
explicit, opt-in [`renormalize`][immunomath.distributions.Distribution.renormalize] method,
documented with its information-destroying implications.

## The three-way mass distinction

A probability of "not observed" is ambiguous unless we are precise about what
"not observed" means. ImmunoMath distinguishes:

| Case | Representation | Meaning |
| --- | --- | --- |
| Omitted | key absent | In a *complete* distribution, probability is exactly `0`. |
| Explicit zero | `{"G": 0.0}` | The outcome is declared possible to name but impossible to occur. |
| Residual | `Residual(m)` | Mass `m` is distributed over unspecified, unlisted outcomes. |

The residual case makes a distribution **open-world**. This controls scoring of
an unlisted truth (see the
[logarithmic score](mathematical-foundations/logarithmic-score.md) and
[distributions](mathematical-foundations/probability-distributions.md) pages).

## Why biological equivalence is a separate layer

The statistical kernel knows nothing about HLA nomenclature, GL Strings, allele
resolution, or G/P groups. Instead it defines a general
[`Projection`][immunomath.Projection] protocol: a function mapping fine outcomes
onto coarse equivalence classes, with probability aggregated (summed) across
each class.

HLA-specific adapters (a later milestone) *implement* this protocol. This keeps
the kernel reusable for any probabilistic classification problem and lets
biological nomenclature evolve — using versioned reference mappings — without
touching the mathematics.

> **Milestone note.** HLA-specific development must not delay the general
> statistical kernel. This release ships the kernel and the projection
> abstraction only.

## Why probabilistic quality cannot be reduced to top-genotype concordance

Reporting only "how often was the top prediction correct?" discards the
information that makes a *probabilistic* method valuable: its confidence.

- Two methods can have identical top-1 accuracy while one is well-calibrated and
  the other wildly overconfident.
- Top-1 accuracy cannot distinguish a prediction that assigned the truth 51%
  from one that assigned it 99%.
- Proper scoring rules (the logarithmic and Brier scores) reward honest
  confidence and penalize overconfidence, so they capture quality that accuracy
  cannot.

ImmunoMath therefore treats proper scoring rules as primary and accuracy/rank as
descriptive diagnostics. See
[Proper Scoring vs. Descriptive Metrics](mathematical-foundations/proper-vs-descriptive.md).

## How ImmunoMath relates to the surrounding ecosystem

```text
 DiplotypeForge  ──►  truth (phase-resolved genotypes)
 TissueTypeForge ──►  simulated typing / ambiguity
 py-graph-imputation / other engines ──►  predicted Distribution
                                             │
                                             ▼
                                        ImmunoMath
                              (scoring, calibration, cohort stats)
```

ImmunoMath consumes predicted distributions and independent truth; it produces
scores, diagnostics, and (in later milestones) calibration curves, cohort
statistics, comparisons, and reports.
