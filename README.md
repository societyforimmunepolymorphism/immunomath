# ImmunoMath

**Metrognome — keeping probabilistic genetics honest.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

ImmunoMath is an open-source Python library, maintained by the Society for
Immune Polymorphism (SIP), for **evaluating** probabilistic genetic inference —
initially HLA genotype imputation, but applicable to probabilistic
classification in general. It provides rigorous, reproducible methods for
assessing the accuracy, calibration, and reliability of predicted probability
distributions.

## ImmunoMath is an evaluation library, not an inference library

ImmunoMath does **not** generate synthetic genotypes, simulate typing, estimate
haplotype frequencies, or perform imputation. It compares predicted probability
distributions against independent truth. Generation and inference belong to
separate, independently usable packages (DiplotypeForge, TissueTypeForge,
py-graph-imputation, and others).

The statistical core depends on **no** HLA nomenclature, GL Strings, or
particular imputation algorithm. Biological adapters are a separate layer.

## Installation

```bash
pip install immunomath            # requires Python 3.12+
```

## Quick start

```python
import immunomath as im

prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})

im.negative_log_score("G1", prediction)  # 0.2231...  (nats; lower is better)
im.brier_score("G1", prediction)  # 0.065      (sum form, range [0, 2])
im.top_k_accuracy("G1", prediction, k=1)  # 1.0
```

ImmunoMath **never silently normalizes** an invalid distribution: a prediction
that does not sum to one (within tolerance) is rejected at construction.

## Calibration and reports (Milestone 4)

```python
from immunomath import analyze_cohort
from immunomath.reporting import write_json, render_report

result = analyze_cohort(cohort, label="engine X", seed=1)  # needs a Cohort
write_json(result, "analysis.json")  # lossless record
render_report(result.to_dict(), "report.html")  # pip install 'immunomath[reporting]'
```

Top-label and event calibration (ECE/MCE, Wilson intervals), risk-coverage with
abstention, and a self-contained Plotly HTML report rebuilt from the saved JSON.

## Milestone 1: the mathematical kernel

- `Distribution`: validated finite discrete distributions, with an explicit
  three-way distinction between omitted outcomes, explicit zeros, and unknown
  residual mass.
- Proper scoring rules: `negative_log_score`, `brier_score` (+ cohort means).
- Descriptive diagnostics: `top_k_accuracy` (order-invariant fractional ties),
  `rank_of_truth` (min/max), `probability_of_truth`, `entropy`,
  `effective_number_of_outcomes`, `num_candidates`.
- A general-purpose `Projection` protocol for mass-preserving outcome
  equivalence (HLA-free in the kernel).
- A cross-validated test suite and a full documentation site.

Later milestones add cohort assessment, comparative analysis, calibration and
reporting, and HLA integration.

## Documentation

Build the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

See the mathematical-foundations pages for formal definitions, worked examples,
proper-scoring-rule theory, and failure modes.

## Development

```bash
pip install -e ".[dev,docs]"
ruff check .
mypy
pytest --doctest-modules src/immunomath tests
mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use ImmunoMath in academic work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).
