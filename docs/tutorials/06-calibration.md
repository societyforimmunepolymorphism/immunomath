# Tutorial 6 — Constructing a Calibration Diagram

This tutorial builds reliability diagrams for two imputation engines on the same
2,000 synthetic subjects: one honest, one over-confident. The punchline: **the
two engines are equally accurate, and only calibration (and the proper scores)
tells them apart.** All data is synthetic and seeded.

## Two engines with identical rankings

Each subject has a "true" distribution over four genotypes. The truth is drawn
from it, so the honest engine, which reports that distribution, is calibrated by
construction. The over-confident engine reports the same distribution raised to
the power 2.5 and renormalized. That sharpens it without changing which genotype
is ranked first.

```python
import numpy as np
from immunomath import Cohort, Distribution, top_label_calibration

rng = np.random.default_rng(606)
N = 2000
candidates = [f"G{i}" for i in range(4)]
ids = [f"s{i:04d}" for i in range(N)]

true_p, truths = [], []
for _ in range(N):
    p = rng.dirichlet([1.5] * 4)
    true_p.append(p)
    truths.append(candidates[int(rng.choice(4, p=p))])


def engine(power):
    dists = []
    for p in true_p:
        q = p**power
        q /= q.sum()
        dists.append(Distribution({c: float(x) for c, x in zip(candidates, q)}))
    return Cohort(ids, truths, dists)


honest, overconfident = engine(1.0), engine(2.5)
```

## Calibration tables

```python
honest_cal = top_label_calibration(honest)  # 10 fixed-width bins
over_cal = top_label_calibration(overconfident)
```

```text
                      ECE      MCE
honest              0.0179   0.1675
over-confident      0.1942   0.2956
```

The over-confident engine's ECE is about ten times larger. Its table shows the
pattern: **every bin sits below the diagonal**, and the shortfall is largest at
high confidence (about 0.2 to 0.28 for bins 5–9, against under 0.05 for bins 3–4).

```text
bin    n   mean predicted   observed   95% Wilson interval
  3    72       0.368         0.333      (0.235, 0.448)
  4   271       0.460         0.410      (0.353, 0.469)
  5   406       0.547         0.355      (0.310, 0.402)
  6   346       0.649         0.434      (0.382, 0.486)
  7   327       0.751         0.544      (0.490, 0.597)
  8   312       0.849         0.571      (0.515, 0.624)
  9   264       0.946         0.701      (0.643, 0.753)
```

When this engine says "95% confident" it is right about **70%** of the time, and
the interval (0.64–0.75) excludes the 0.95 it claimed.

## Do not over-read MCE

The honest engine's MCE of 0.1675 looks alarming next to its ECE of 0.018. Look
at the bin responsible:

```text
bin    n   mean predicted   observed   95% Wilson interval
  8     9       0.834         0.667      (0.354, 0.879)
```

It holds **9 subjects**, and its interval easily contains the diagonal. Indeed
every bin of the honest engine is consistent with perfect calibration. MCE is
the single worst bin, so it is dominated by sparse bins. Read it together with
bin counts, or use `binning="equal_frequency"` so that every bin is well
populated.

## Fixed-width versus equal-frequency bins

```python
eq = top_label_calibration(overconfident, binning="equal_frequency")
```

```text
fixed-width ECE = 0.1942     equal-frequency ECE = 0.1942
```

They agree here, and the reason is mathematical, not a coincidence. When every
bin is over-confident, the absolute values in ECE drop out and
$\mathrm{ECE} = \bar c - \bar y$ (mean confidence minus accuracy) for **any**
binning. See [Calibration](../mathematical-foundations/calibration.md). ECE only
becomes binning-sensitive when bins disagree in sign.

## Calibration is not quality: accuracy cannot see it

Both engines rank the same genotype first for every subject, so their top-1
accuracy is identical. The proper scores separate them:

```text
                 top-1 error   mean log score   mean Brier
honest              0.515          1.161           0.635
over-confident      0.515          1.377           0.696
```

The over-confident engine is worse on both proper scoring rules despite matching
the honest engine's accuracy exactly. This is the central argument of
ImmunoMath: probabilistic quality cannot be reduced to top-genotype concordance.

## Draw the diagram

Reliability diagrams and a complete report are produced from one analysis record
(requires `pip install 'immunomath[reporting]'`):

```python
from immunomath import analyze_cohort
from immunomath.reporting import write_json, render_report

result = analyze_cohort(overconfident, label="over-confident engine", seed=0)
write_json(result, "overconfident.json")  # the saved analysis record
render_report(result.to_dict(), "overconfident.html")
```

The report's reliability diagram shows the dashed perfect-calibration diagonal,
one marker per non-empty bin (area proportional to the bin's subject count), and
Wilson interval bars. Every bin of the over-confident engine lies below the
diagonal, and for bins 5–9 the Wilson intervals exclude it; the two lowest-
confidence bins (3–4) are still compatible with it.

## Risk–coverage for the honest engine

Because the honest engine's confidence is informative, abstaining on its least
confident calls helps:

```python
from immunomath import risk_coverage

curve = risk_coverage(honest)
```

```text
coverage   error rate   mean log loss
  0.25       0.349          0.990
  0.50       0.414          1.050
  0.75       0.471          1.110
  1.00       0.515          1.161
```

Answering only the most confident quarter of subjects cuts the error rate from 51.5% to
34.9% and the mean log loss from 1.161 to 0.990. Both fall together here, which
is what a useful confidence score should do. See
[Risk–Coverage](../mathematical-foundations/risk-coverage.md).

## What you learned

- A reliability diagram plots observed frequency against predicted probability;
  below the diagonal means over-confident.
- ECE is weighted by bin count; MCE is dominated by sparse bins, so check counts.
- Equal accuracy does **not** imply equal quality; calibration and the proper
  scores see what concordance cannot.
