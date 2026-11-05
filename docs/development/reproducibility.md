# Reproducibility Checklist

ImmunoMath makes its own computations reproducible: given the same cohort, the
same parameters and the same seed, `analyze_cohort` returns the same record. It
cannot make the *upstream* of an evaluation reproducible. That is your
responsibility, and this page lists what to keep.

!!! note "Scope"
    Structured experiment manifests and manuscript-export helpers are planned but
    not yet implemented. Until then, record the items below alongside each saved
    analysis record, for example in a small `manifest.json` of your own.

## What ImmunoMath guarantees

- **Seeded resampling.** Bootstrap intervals come from a NumPy `Generator` seeded
  with the `seed` you pass, so the same seed gives identical bounds. With
  `seed=None` the result is *not* reproducible and the report says so.
- **No hidden clock.** The record contains a timestamp only if you supply
  `created=`.
- **Parameters are stored.** Every setting that influences a number is saved in
  `metadata.parameters`.
- **Order-invariant statistics.** Top-k credit, rank intervals, risk–coverage
  ties and calibration do not depend on the order of subjects or candidates.
- **Lossless round trip.** `inf` and `nan` survive JSON export exactly.

## What you must record yourself

| Item | Why it matters |
| --- | --- |
| Inference engine and version | different versions give different probabilities |
| Reference panel / frequency model and version | the dominant source of change in imputation |
| Nomenclature release (for HLA) | allele names and equivalence classes change between releases |
| How the prediction file was exported | truncation or renormalization upstream changes the distribution |
| Cohort definition and subject-selection rules | defines the denominator |
| Source of the truth and how it was determined | a "truth" from lower-resolution typing is not genotype truth |
| Definition of any `confidence_score` callable | the record stores only its kind |
| Library version | recorded automatically in `metadata.immunomath_version` |
| Random seed | recorded in `metadata.parameters.seed` |

## Independent training and validation cohorts

Evaluation is only meaningful if the truth is independent of what the engine saw:

1. **No subject overlap.** Subjects used to build or tune the reference panel must
   not appear in the evaluation cohort. ImmunoMath cannot detect this.
2. **No relatives across the split.** Close relatives share haplotypes; leaking
   them inflates accuracy and distorts calibration.
3. **Choose operating thresholds on training data.** A risk–coverage threshold or
   a calibration fix selected on the evaluation cohort gives optimistic results.
4. **Do not tune on the evaluation score.** Repeatedly adjusting an engine to
   improve an evaluation metric turns the evaluation set into a training set.

The Wilson and bootstrap intervals assume independent subjects. Relatives or
repeated individuals break that assumption.

## Minimal reproducible run

```python
from immunomath import analyze_cohort
from immunomath.reporting import write_json

result = analyze_cohort(cohort, label="engine X 1.4 / panel P7", seed=20241104)
write_json(result, "analysis.json")
```

Then store, next to `analysis.json`: the engine and panel versions, the
nomenclature release, the cohort definition, the truth source, and the
ImmunoMath version (already inside the record).
