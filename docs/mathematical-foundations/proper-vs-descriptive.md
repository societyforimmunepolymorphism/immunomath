# Proper Scoring Rules vs. Descriptive Metrics

This page explains a distinction that is central to using ImmunoMath correctly:
the difference between **proper scoring rules** and **descriptive diagnostics**,
and the documented behavior for ties and zero probabilities.

## Two categories of metric

| Proper scoring rules | Descriptive diagnostics |
| --- | --- |
| [`negative_log_score`][immunomath.negative_log_score] | [`top_k_accuracy`][immunomath.top_k_accuracy] |
| [`brier_score`][immunomath.brier_score] | [`rank_of_truth`][immunomath.rank_of_truth] |
| | [`entropy`][immunomath.entropy] |
| | [`effective_number_of_outcomes`][immunomath.effective_number_of_outcomes] |
| | [`num_candidates`][immunomath.num_candidates] |
| | [`probability_of_truth`][immunomath.probability_of_truth] |

A **strictly proper scoring rule** is uniquely minimized in expectation by
reporting one's true belief. Proper scoring rules therefore measure *forecast
quality*: they reward both correctness and honest calibration and cannot be
gamed by hedging or overconfidence.

**Descriptive diagnostics** summarize a prediction or its relationship to the
truth, but they are *not* proper scoring rules and must not be read as measures
of quality.

## Why accuracy and rank are not enough

Accuracy and rank depend only on the **ordering** of probabilities, not their
magnitudes:

- A prediction that assigns the truth $0.51$ and one that assigns it $0.99$ have
  identical top-1 accuracy and identical rank.
- Two methods can tie on accuracy while one is well-calibrated and the other
  dangerously overconfident.

Only a proper scoring rule distinguishes these cases. This is why ImmunoMath
treats proper scoring rules as primary and reports accuracy/rank as context.

## Entropy is not correctness

!!! warning
    [Entropy][immunomath.entropy] measures the **spread** of a prediction. It is
    a property of the distribution *alone* and never involves the truth. A
    confidently wrong prediction has **low** entropy. Entropy must never be
    interpreted as a measure of correctness.

Formally, the Shannon entropy is

$$
H(P) = -\sum_k p_k \ln p_k, \qquad 0\ln 0 := 0,
$$

in nats (pass `base=2` for bits). A point mass has $H = 0$; a uniform
distribution over $n$ outcomes has $H = \ln n$, the maximum.

```python
>>> import math, immunomath as im
>>> im.entropy(im.Distribution({"G1": 1.0}))                 # point mass
0.0
>>> round(im.entropy(im.Distribution({"A": 0.5, "B": 0.5})), 6) == round(math.log(2), 6)
True
```

The **effective number of outcomes** is $N_\text{eff} = \exp(H)$ (the
perplexity): the size of a uniform distribution with the same entropy. It is an
interpretable "effective candidate count" but, like entropy, says nothing about
correctness.

## Tie handling (documented, deterministic, order-invariant)

Evaluation statistics must not depend on the arbitrary order in which equally
probable candidates happen to be listed. ImmunoMath guarantees order-invariance
by construction.

### Top-K accuracy: fractional credit

Let $p^\*$ be the probability of the truth, $s$ the number of outcomes strictly
more probable, and $t$ the number tied at exactly $p^\*$ (including the truth).
Then top-$k$ credit is

$$
\text{credit} =
\begin{cases}
0 & \text{if } s \ge k,\\[2pt]
1 & \text{if } s + t \le k,\\[2pt]
\dfrac{k - s}{t} & \text{otherwise.}
\end{cases}
$$

The fractional middle case is exactly the expected inclusion if the $t$ tied
candidates were ordered at random, so the statistic is invariant to tie-break
order.

```python
>>> import immunomath as im
>>> a = im.Distribution({"G1": 0.5, "G2": 0.5})
>>> b = im.Distribution({"G2": 0.5, "G1": 0.5})   # reversed order
>>> im.top_k_accuracy("G1", a, k=1) == im.top_k_accuracy("G1", b, k=1) == 0.5
True
```

### Rank of truth: minimum and maximum

With $s$ outcomes strictly above the truth and $t$ tied, the truth's rank lies
in $[\,s+1,\; s+t\,]$. ImmunoMath reports both endpoints rather than committing
to an arbitrary order:

```python
>>> r = im.rank_of_truth("G1", im.Distribution({"G1": 0.5, "G2": 0.5}))
>>> (r.min_rank, r.max_rank, r.is_tied)
(1, 2, True)
```

A canonical sort key may be used for *presentation* (e.g. tables), but it does
not change these statistics.

## Zero probabilities

- **Log score:** zero probability on the truth gives `+inf` (see the
  [logarithmic score](logarithmic-score.md) page). No clipping.
- **Top-K / rank:** a zero-probability truth is never counted as "predicted";
  top-K credit is `0`, and its rank is undefined (treated as "not ranked").
- **Entropy:** outcomes with probability zero contribute nothing, by the
  convention $0\ln 0 = 0$.

## When robust descriptive statistics are appropriate

Robust summaries (trimmed means, medians, quantiles of individual loss) are
valuable for **diagnosis** — understanding which subjects drive a result. They
are **not** interchangeable with proper scoring rules: trimming discards exactly
the catastrophic cases a proper rule is designed to penalize. ImmunoMath (from
Milestone 2) supports such summaries as explicit sensitivity analyses while
preserving the ordinary mean log loss as the primary proper-scoring summary.

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
- Jost, L. (2006). Entropy and diversity. *Oikos*, 113(2), 363–375.
  (Effective number of outcomes / Hill numbers.)
