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
uv add immunomath                 # in a uv project
pip install immunomath            # or pip, inside a virtual environment (Python 3.12+)
```

Optional extras: `immunomath[reporting]` adds Plotly (HTML reports) and PyArrow (Parquet).

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

## Observation-weighted scoring (TRS-aware)

Conventional means weight every observation equally. When a minority of
observations are poorly resolved (low HLA Typing Resolution Score, TRS), they
can be diluted even though modelling choices matter most there. ImmunoMath adds
**covariate-weighted** proper scoring, a **paired** comparison of two methods,
and **TRS-stratified** validation with a group-balanced macro-average — all
domain-independent, with the unweighted result always reported alongside.

```python
import immunomath as im
from immunomath.trs import trs_weights  # optional, non-kernel adapter

truths = ["G1", "G1", "G1"]
incumbent = [im.Distribution({"G1": 0.90, "G2": 0.10})] * 2 + [
    im.Distribution({"G1": 0.60, "G2": 0.40})
]
candidate = [im.Distribution({"G1": 0.95, "G2": 0.05})] * 2 + [
    im.Distribution({"G1": 0.30, "G2": 0.70})
]  # worse for the low-TRS donor

trs = [0.95, 0.95, 0.10]  # normalized, higher = better resolved
weights = trs_weights(trs, lam=4.0, gamma=2.0)  # w = 1 + λ(1 − r)^γ (explicit params)

cmp = im.compare_methods_weighted(truths, incumbent, candidate, weights)
cmp.mean_improvement  # conventional: Δ = L_incumbent − L_candidate (>0 favors candidate)
cmp.weighted_mean_improvement  # TRS-weighted improvement
cmp.effective_sample_size  # Kish N_eff = (Σw)² / Σw²

strat = im.stratified_comparison(
    truths, incumbent, candidate, strata=["high", "high", "low"], stratum_order=["low", "high"]
)
strat.overall_mean_improvement_log  # donor-weighted (conventional) improvement
strat.macro_mean_improvement_log  # group-balanced: (1/K) Σ_k mean_k over nonempty strata
```

**Statistical guarantee.** Weights must be fixed functions of pre-registered
covariates (e.g. observed typing resolution) and must **not** depend on the
truth or on any model's predictions. Scaling a strictly proper loss by a fixed
positive weight preserves its unique minimizer, so the weighted mean stays a
proper criterion *conditional on the covariates*. See
[Weighted Scoring](docs/mathematical-foundations/weighted-scoring.md) and
`examples/trs_weighted_comparison.py`.

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
uv run --extra docs mkdocs serve
```

`uv` creates and manages the virtual environment for you. Without uv, create one
first (a bare `pip install` is refused by system and Homebrew Pythons):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[docs]" && mkdocs serve
```

See the mathematical-foundations pages for formal definitions, worked examples,
proper-scoring-rule theory, and failure modes.

## Development

```bash
uv sync --extra dev --extra docs          # creates .venv from uv.lock
uv run ruff check .
uv run mypy
uv run pytest --doctest-modules src/immunomath tests
uv run mkdocs build --strict
```

Or with plain pip: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,docs]"`.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use ImmunoMath in academic work, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE).
