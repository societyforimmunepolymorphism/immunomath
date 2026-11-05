# Risk–Coverage and Abstention

A selective predictor may **decline to answer**. Sweeping a confidence threshold
traces the trade-off between how many cases it answers (*coverage*) and how often
the answered cases are wrong (*risk*).
[`risk_coverage`][immunomath.risk_coverage] computes the full curve.

## Definitions

For a threshold $t$, a subject is **retained** if its confidence score is
$\ge t$ and the predictor did not abstain. Among $N$ evaluable subjects with
$n_t$ retained:

$$
\text{coverage}(t) = \frac{n_t}{N}, \qquad
\text{accuracy}(t) = \frac{1}{n_t}\sum_{i \,\text{retained}} y_i, \qquad
\text{risk}(t) = 1 - \text{accuracy}(t),
$$

where $y_i$ is the fractional top-1 credit. Alongside accuracy, each point
reports the **mean negative log score** over retained subjects:

$$
\overline{L}(t) = \frac{1}{n_t}\sum_{i \,\text{retained}} -\ln P_i(g_i),
$$

which is $+\infty$ if any retained subject has an identifiable-zero truth.

## Why report both accuracy and log loss

Selecting on confidence usually raises accuracy. But accuracy ignores *how
wrong* the remaining errors are. A confidence score that retains only high-top-
probability cases can improve accuracy while still keeping the confidently wrong
predictions that dominate the log loss. Reporting the two together exposes that:

- accuracy up, log loss flat or rising $\Rightarrow$ the score is selecting
  easy cases but not removing the dangerous ones;
- both improving $\Rightarrow$ abstention is genuinely removing bad predictions.

## The confidence score is user-defined

The default score is the **top predicted probability**. Because that is not
always the best choice, the score can be supplied:

```python
>>> from immunomath import Cohort, Distribution, risk_coverage
>>> cohort = Cohort(
...     ["a", "b", "c", "d"], ["G1", "G1", "G2", "G1"],
...     [Distribution({"G1": 0.95, "G2": 0.05}), Distribution({"G1": 0.8, "G2": 0.2}),
...      Distribution({"G1": 0.7, "G2": 0.3}), Distribution({"G1": 0.6, "G2": 0.4})],
... )
>>> result = risk_coverage(cohort)
>>> [(p.n_retained, round(p.accuracy, 3)) for p in result.points]
[(1, 1.0), (2, 1.0), (3, 0.667), (4, 0.75)]
```

| `confidence=` | Meaning |
| --- | --- |
| `None` | top predicted probability |
| a callable `f(distribution)` | e.g. the margin between the top two probabilities, or negative entropy |
| a mapping `subject_id -> score` | any externally computed score, such as one the inference engine emits |

A score of `None` means the predictor **abstained** on that subject.

## Abstention is part of coverage

Subjects the predictor abstained on remain in the denominator $N$ and are never
retained, so abstention lowers maximum coverage rather than silently shrinking
the evaluated set. In the example above, if one of four subjects abstains the
curve cannot exceed coverage $0.75$.

## Ties and order

Subjects with equal scores are always retained or dropped **together**; the
curve has one point per distinct score. It is therefore invariant to the order
in which subjects are supplied.

## Excluded subjects

Subjects whose truth probability is not identifiable are excluded and counted in
`n_undefined`; they do not enter the coverage denominator.

## Assumptions and failure modes

- **A threshold is chosen on the same data it is evaluated on.** The reported
  curve is descriptive. To pick an operating threshold, choose it on a training
  cohort and evaluate on an independent validation cohort.
- **Few retained subjects at low coverage.** The leftmost points rest on very
  few subjects; accuracy there is noisy. Check `n_retained`.
- **Coverage is relative to evaluable subjects**, not to every subject in the
  cohort.
- **Tied scores create plateaus**, so a coarse confidence score gives a coarse
  curve.

## References

- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural
  networks. *NeurIPS*.
- El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective
  classification. *JMLR*, 11, 1605–1641.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE
  Trans. Information Theory*, 16(1), 41–46.
