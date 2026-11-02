# The Negative Logarithmic Score

The negative logarithmic score (log loss) is the primary proper scoring rule in
ImmunoMath.

## Formal definition

For an observed true outcome $g$ and a predicted distribution $P$, the negative
logarithmic score is

$$
L(P, g) = -\ln P(g).
$$

ImmunoMath uses the **natural logarithm**, so $L$ is measured in **nats**. For a
cohort of $N$ subjects the mean is

$$
\bar{L} = \frac{1}{N} \sum_{i=1}^{N} L(P_i, g_i)
         = \frac{1}{N} \sum_{i=1}^{N} -\ln P_i(g_i).
$$

## Plain-English explanation

The log score asks a single question of your prediction: *how surprised were you
by the truth?* If you assigned the true outcome high probability you are barely
surprised and the penalty is small; if you assigned it a tiny probability you
are very surprised and the penalty is large. Assigning the truth **zero**
probability is infinite surprise — an unbounded penalty.

## Worked numerical examples

Using the running example $P = (0.80, 0.15, 0.05)$ over $(G_1, G_2, G_3)$:

| Truth | $P(\text{truth})$ | $L = -\ln P(\text{truth})$ |
| --- | --- | --- |
| $G_1$ | $0.80$ | $0.2231$ |
| $G_2$ | $0.15$ | $1.8971$ |
| $G_3$ | $0.05$ | $2.9957$ |

```python
>>> import immunomath as im
>>> d = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> round(im.negative_log_score("G1", d), 4)
0.2231
>>> im.negative_log_score("G1", im.Distribution({"G1": 1.0}))   # perfect
0.0
>>> im.negative_log_score("G2", im.Distribution({"G1": 1.0, "G2": 0.0}))  # zero prob
inf
```

## Interpreting higher and lower values

- **Lower is better.** $0$ is a perfect prediction (the truth was assigned
  probability $1$).
- There is **no upper bound**: as $P(g) \to 0^+$, $L \to +\infty$.
- A difference of $\ln 2 \approx 0.693$ nats corresponds to a factor-of-two
  difference in the probability assigned to truth.

## Why the logarithmic score is strictly proper

A scoring rule is **proper** if a forecaster minimizes their *expected* penalty
by reporting their true belief, and **strictly proper** if that true belief is
the unique minimizer.

Let the forecaster's true belief be $Q$ and suppose they report $P$. Their
expected log loss is the cross-entropy

$$
\mathbb{E}_{g \sim Q}\bigl[-\ln P(g)\bigr]
 = -\sum_g Q(g)\ln P(g)
 = \underbrace{-\sum_g Q(g)\ln Q(g)}_{H(Q)}
 + \underbrace{\sum_g Q(g)\ln\frac{Q(g)}{P(g)}}_{D_{\mathrm{KL}}(Q \,\|\, P)}.
$$

The Kullback–Leibler divergence $D_{\mathrm{KL}}(Q \,\|\, P) \ge 0$, with
equality **iff** $P = Q$ (Gibbs' inequality). The expected penalty is therefore
minimized uniquely at $P = Q$: honest reporting is strictly optimal, and no
hedging or overconfidence can do better in expectation.

## Sensitivity to confidently incorrect predictions

Because $-\ln p \to \infty$ as $p \to 0^+$, a single confidently wrong
prediction can dominate a cohort mean:

```python
>>> import immunomath as im
>>> confident_wrong = im.negative_log_score("G2", im.Distribution({"G1": 0.999, "G2": 0.001}))
>>> humble = im.negative_log_score("G2", im.Distribution({"G1": 0.5, "G2": 0.5}))
>>> round(confident_wrong, 3), round(humble, 3)
(6.908, 0.693)
```

This is a **feature** for safety-critical evaluation — it is exactly the
behavior that discourages overconfidence — but it means the cohort mean is
sensitive to a few catastrophic errors. Milestone 2 adds catastrophic-error
analysis to diagnose these cases without discarding them.

## Zero probabilities and the (optional) floor

ImmunoMath performs **no implicit epsilon clipping**. A zero probability
assigned to the truth yields exactly `+inf`. For numerical experiments where a
finite surrogate is explicitly desired, an opt-in `floor` may be supplied:

```python
>>> import immunomath as im
>>> d = im.Distribution({"G1": 0.0, "G2": 1.0})
>>> im.negative_log_score("G1", d)                 # no clipping
inf
>>> round(im.negative_log_score("G1", d, floor=1e-6), 4)
13.8155
```

The floor is never applied by default, and its use should be disclosed in any
reported analysis, because it changes the score from a strictly proper rule into
a bounded approximation.

## Residual-mass semantics

- **Explicit truth, residual present:** scored normally using the reported
  probability.
- **Truth omitted from a complete distribution:** probability is $0$, loss is
  `+inf`.
- **Truth possibly inside residual mass:** probability is unidentifiable; exact
  log loss is **undefined** and a `KeyError` is raised. Use event-level scoring
  instead.

## Assumptions and failure modes

- Requires the probability of the *exact* truth to be identifiable. For
  ambiguous truth, score an identifiable event instead.
- The mean is undefined (reported as `+inf`) whenever any subject has infinite
  loss; ImmunoMath reports the count of infinite losses separately rather than
  dropping them.

## Relationship to other metrics

- The log score is the **negative log-likelihood** of the truth under the
  prediction; minimizing mean log loss is maximum-likelihood evaluation.
- Its expectation decomposes into entropy plus KL divergence (above), linking it
  to information theory.
- Unlike accuracy or rank, it depends on the *magnitude* of probabilities, so it
  detects miscalibration that ordering-based metrics cannot.

## References

- Good, I. J. (1952). Rational decisions. *JRSS B*, 14(1), 107–114.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
