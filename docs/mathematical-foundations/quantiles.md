# Loss-Distribution Quantiles

A cohort of per-subject losses induces an empirical **loss distribution**. Its
quantiles summarize the spread of forecast penalties — the typical (median)
penalty, and the upper tail where catastrophic predictions live.
[`loss_quantile`][immunomath.loss_quantile] and
[`loss_quantiles`][immunomath.loss_quantiles] compute them with a documented,
configurable convention and explicit handling of infinite losses.

## The quantile convention: Hyndman–Fan type 7 (default)

There is no single definition of a sample quantile; estimators differ in how
they place order statistics and interpolate. ImmunoMath defaults to
**Hyndman–Fan type 7** (`QuantileMethod.LINEAR`), the piecewise-linear rule used
by NumPy, R, pandas, and Julia by default.

For a sorted sample $x_{(1)} \le \dots \le x_{(n)}$ and level $q$, type 7 places
the $k$-th order statistic at plotting position $(k-1)/(n-1)$ and sets

$$
Q(q) = x_{(\lfloor h \rfloor)} + \bigl(h - \lfloor h \rfloor\bigr)
       \left( x_{(\lceil h \rceil)} - x_{(\lfloor h \rfloor)} \right),
\qquad h = (n-1)\,q + 1 .
$$

It is continuous in $q$ and returns sample extremes at $q = 0$ and $q = 1$.

```python
>>> from immunomath import loss_quantile
>>> loss_quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5)
2.0
>>> loss_quantile([1.0, 2.0, 3.0, 4.0], 0.25)
1.75
```

This matches `numpy.quantile(..., method="linear")` exactly for finite data, so
ImmunoMath quantiles are directly comparable to the wider scientific-Python
ecosystem.

## Alternative conventions

The convention is an explicit argument so a reported quantile is never ambiguous
about which estimator produced it. Available members of
[`QuantileMethod`][immunomath.QuantileMethod] map onto NumPy's `method=`:

| Member | NumPy `method` | Hyndman–Fan type |
| --- | --- | --- |
| `LINEAR` (default) | `"linear"` | type 7 |
| `WEIBULL` | `"weibull"` | type 6 |
| `MEDIAN_UNBIASED` | `"median_unbiased"` | type 8 |
| `LOWER` | `"lower"` | discontinuous (nearest-rank down) |
| `HIGHER` | `"higher"` | discontinuous (nearest-rank up) |
| `NEAREST` | `"nearest"` | discontinuous (nearest order statistic) |

Type 8 (`MEDIAN_UNBIASED`) is recommended by Hyndman & Fan (1996) as
approximately median-unbiased across distributions; type 7 is kept as the
default for consistency with prevailing tools.

## Infinite losses are handled explicitly

An identifiable-zero truth produces a `+inf` log score. These infinities are
**real data**, not errors, and the quantile routines handle them without
dropping them or returning an unexplained `nan`:

- infinities sort to the **top** of the sample (they are the largest losses);
- a quantile whose interpolation bracket is entirely finite is computed
  normally;
- a quantile that falls **within the infinite upper tail** returns `+inf`
  exactly — never the indeterminate `inf − inf = nan` that naïve interpolation
  would produce.

```python
>>> import math
>>> loss_quantile([0.1, 0.2, math.inf], 0.9)    # upper tail is infinite
inf
>>> loss_quantile([0.1, 0.2, math.inf], 0.25)   # lower quantile still finite
0.15000000000000002
```

NaN inputs are rejected with a clear error: a NaN loss indicates an upstream
scoring defect that should be fixed, not summarized over.

## Interpreting loss quantiles

- The **median** ($q = 0.5$) is the typical penalty, robust to a few extreme
  subjects.
- Upper quantiles ($q = 0.95, 0.99$) expose the tail: if the 99th percentile is
  `+inf`, at least 1% of subjects were assigned zero probability on the truth.
- Quantiles are **descriptive**; they summarize the loss distribution but are
  not themselves proper scoring rules.

## Assumptions and failure modes

- Quantiles require a non-empty sample.
- With few subjects, high quantiles are coarse (there simply are not enough
  order statistics to resolve the far tail).
- The convention matters at small $n$; always report which `QuantileMethod` was
  used.

## References

- Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages.
  *The American Statistician*, 50(4), 361–365.
