# The Multiclass Brier Score

The Brier score is a second strictly proper scoring rule, complementary to the
logarithmic score.

## Formal definition

Using the complete probability distribution, ImmunoMath computes the **sum form**
multiclass Brier score

$$
B(P, g) = \sum_{k} \bigl( p_k - y_k \bigr)^2,
\qquad
y_k = \begin{cases} 1 & \text{if } \omega_k = g, \\ 0 & \text{otherwise.}\end{cases}
$$

Here $y$ is the one-hot indicator of the true outcome. The score is the squared
Euclidean distance between the predicted probability vector and the vertex of
the simplex corresponding to the truth.

## Plain-English explanation

The Brier score measures how far your probability vector sits from the "perfect"
vector that would put all its weight on the true outcome. A perfect prediction
sits exactly on that vertex and scores $0$; a prediction that confidently backs
the wrong outcome sits as far away as possible and scores $2$.

## Range and the binary relationship

The sum form ranges over $[0, 2]$:

- $B = 0$ — perfect prediction (all mass on the truth).
- $B = 2$ — worst case (all mass on a single wrong outcome).

!!! note "Relationship to binary Brier scoring"
    The conventional **binary** Brier score is $(p - y)^2$, with range $[0, 1]$.
    Restricting the multiclass sum form to two complementary classes gives

    $$
    (p - y)^2 + \bigl((1-p) - (1-y)\bigr)^2 = 2\,(p - y)^2,
    $$

    exactly **twice** the binary score. ImmunoMath reports the sum form by
    default so the convention is consistent for any number of classes. Divide by
    $2$ to recover the familiar binary scale.

## Worked numerical example

For $P = (0.80, 0.15, 0.05)$ with truth $G_1$:

$$
B = (0.80 - 1)^2 + (0.15 - 0)^2 + (0.05 - 0)^2
  = 0.04 + 0.0225 + 0.0025
  = 0.065.
$$

```python
>>> import immunomath as im
>>> d = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> round(im.brier_score("G1", d), 10)
0.065
>>> im.brier_score("G1", im.Distribution({"G1": 1.0}))           # perfect
0.0
>>> im.brier_score("G2", im.Distribution({"G1": 1.0, "G2": 0.0}))  # worst case
2.0
```

## Why the Brier score is strictly proper

With true belief $Q$, the expected Brier score under report $P$ is

$$
\mathbb{E}_{g \sim Q}[B(P, g)]
 = \sum_k \bigl(p_k - q_k\bigr)^2 + \sum_k q_k(1 - q_k).
$$

The second term does not depend on $P$; the first is a sum of squares minimized
uniquely at $p_k = q_k$. Honest reporting is therefore strictly optimal.

## Interpreting higher and lower values

- **Lower is better**, with $0$ perfect.
- Unlike the logarithmic score, the Brier score is **bounded**, so a single
  catastrophic prediction contributes at most $2$ to the sum. This makes the
  Brier score less sensitive to rare, confidently-wrong predictions — useful as
  a robustness cross-check, but it also means the Brier score under-penalizes
  the overconfidence that the log score flags.

## Residual-mass semantics

Only explicitly listed outcomes contribute to the squared-error sum; residual
mass is not expanded into hypothetical outcomes. If the truth is unlisted in a
*complete* distribution, the missing truth class contributes $(0 - 1)^2 = 1$. If
the truth is unidentifiable in an *open* distribution, a `KeyError` is raised.

## Assumptions and failure modes

- The sum form assumes you want a single number over the full distribution; if
  you only care about a specific event, score that event instead.
- Because it is bounded, the Brier score should **not** replace the logarithmic
  score as the primary summary when overconfidence is the key risk.

## Relationship to other metrics

- Both the Brier and logarithmic scores are strictly proper, but they weight
  errors differently: the log score penalizes small probabilities on the truth
  far more steeply.
- The Brier score decomposes (Murphy, 1973) into reliability, resolution, and
  uncertainty components, linking it directly to calibration analysis
  (Milestone 4).

## References

- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. *Monthly Weather Review*, 78(1), 1–3.
- Murphy, A. H. (1973). A new vector partition of the probability score.
  *Journal of Applied Meteorology*, 12(4), 595–600.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
