# Getting Started

## Installation

ImmunoMath requires Python 3.12 or newer.

```bash
pip install immunomath
```

For development (tests, linting, type checking) and documentation:

```bash
pip install -e ".[dev,docs]"
```

## A first evaluation

Every ImmunoMath evaluation begins with a
[`Distribution`][immunomath.Distribution]: a validated, finite, discrete
probability distribution over hashable outcomes.

```python
import immunomath as im

prediction = im.Distribution(
    {
        "G1": 0.80,
        "G2": 0.15,
        "G3": 0.05,
    }
)
```

If the probabilities did not sum to one (within a documented tolerance),
construction would fail immediately — ImmunoMath never silently repairs a
distribution.

Now score the prediction against a known truth:

```python
>>> import immunomath as im
>>> prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> round(im.negative_log_score("G1", prediction), 6)   # natural log, in nats
0.223144
>>> round(im.brier_score("G1", prediction), 10)         # sum form, range [0, 2]
0.065
>>> im.top_k_accuracy("G1", prediction, k=1)   # fractional-credit ties
1.0
```

## Interpreting the numbers

- **Negative log score** (`0.223`): lower is better; `0` is perfect. It uses the
  natural logarithm and penalizes confidently-wrong predictions severely (an
  outcome assigned probability zero that then occurs costs `+inf`).
- **Brier score** (`0.065`): lower is better; `0` is perfect, `2` is the worst
  possible (all mass on a single wrong outcome).
- **Top-1 accuracy** (`1.0`): the truth was the single most probable outcome.

See [Mathematical Foundations](mathematical-foundations/logarithmic-score.md)
for the formal definitions, worked examples, and failure modes.

## Outcomes can be any hashable identifier

```python
im.Distribution({0: 0.5, 1: 0.5})  # integers
im.Distribution({("A*01:01", "A*02:01"): 1.0})  # tuples / genotypes
im.Distribution({"A*01": 0.7, "A*02": 0.3})  # allele strings
```

## The three-way mass distinction

ImmunoMath carefully distinguishes:

1. an **omitted** outcome,
2. an outcome with **explicit zero** probability, and
3. **unknown residual** probability mass.

```python
from immunomath import Distribution, Residual

complete = Distribution({"G1": 0.8, "G2": 0.2})  # closed world
with_zero = Distribution({"G1": 0.8, "G2": 0.2, "G3": 0.0})  # G3 explicitly impossible
open_world = Distribution({"G1": 0.8, Residual(0.2): None})  # 0.2 unspecified
```

This distinction is not cosmetic — it changes how an unlisted truth is scored.
See the [Probability Distributions](mathematical-foundations/probability-distributions.md)
foundations page.
