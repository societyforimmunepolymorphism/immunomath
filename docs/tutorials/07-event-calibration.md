# Tutorial 7 — Evaluating Allele-Presence Probabilities

Often the clinical question is not "what is the exact genotype?" but "**does this
person carry allele X?**". This tutorial evaluates allele-presence probabilities
by aggregating genotype probabilities into an *event*, and shows why exact-
genotype concordance can hide a serious problem that event calibration exposes.
All data is synthetic and seeded.

## Genotypes as tuples, events as sets

Outcomes can be any hashable identifier, so a genotype is naturally a sorted
tuple of alleles. With three alleles there are six unordered genotypes:

```python
import itertools
import numpy as np
from immunomath import Cohort, Distribution, event_calibration, top_label_calibration

alleles = ["A*01", "A*02", "A*03"]
genotypes = [tuple(sorted(g)) for g in itertools.combinations_with_replacement(alleles, 2)]
# [('A*01','A*01'), ('A*01','A*02'), ('A*01','A*03'),
#  ('A*02','A*02'), ('A*02','A*03'), ('A*03','A*03')]

# The event "carries A*02" is the set of genotypes that contain it (3 of the 6).
carries_a02 = {g for g in genotypes if "A*02" in g}
```

## Two engines with the same genotype ranking

As in [Tutorial 6](06-calibration.md), each subject has a true genotype
distribution and the truth is drawn from it. The honest engine reports it; the
over-confident engine reports it raised to the power 3 and renormalized.

```python
rng = np.random.default_rng(707)
M = 1500
true_p, truths = [], []
for _ in range(M):
    p = rng.dirichlet([1.0] * len(genotypes))
    true_p.append(p)
    truths.append(genotypes[int(rng.choice(len(genotypes), p=p))])
ids = [f"p{i:04d}" for i in range(M)]


def engine(power):
    dists = []
    for p in true_p:
        q = p**power
        q /= q.sum()
        dists.append(Distribution({g: float(x) for g, x in zip(genotypes, q)}))
    return Cohort(ids, truths, dists)


honest, overconfident = engine(1.0), engine(3.0)
```

## Exact-genotype concordance cannot tell them apart

Raising probabilities to a power never changes which genotype is ranked first, so
both engines have exactly the same top-1 accuracy:

```text
exact-genotype top-1 accuracy:   honest 0.4027    over-confident 0.4027
```

A benchmark that reports only concordance would call these engines equivalent.

## Event calibration separates them

```python
event_cal = event_calibration(overconfident, carries_a02, label="carries A*02")
genotype_cal = top_label_calibration(overconfident)
```

The predicted probability for each subject is the **sum** of the probabilities of
the three genotypes in the event; the observed indicator is whether the true
genotype is in the event.

```text
                  genotype top-label ECE    event ECE    event Brier
honest                   0.0102               0.0184        0.2174
over-confident           0.2948               0.1782        0.2578
```

- The honest engine is well calibrated both for the exact genotype and for the
  allele-presence event.
- The over-confident engine is badly miscalibrated at the genotype level (ECE
  0.29) and still clearly miscalibrated for allele presence (ECE 0.18).
- The event's **Brier score**, a strictly proper score for the binary question,
  is worse for the over-confident engine (0.258 against 0.217), so it is
  genuinely a worse answer to the clinical question, not only a worse-calibrated
  one.

Note that aggregation partly averages out the over-confidence: the event ECE
(0.18) is smaller than the genotype ECE (0.29). Over-confidence about *which*
genotype does not fully carry over to *whether an allele is present*, which is a
coarser question. An engine can therefore be fit for allele-presence use while
unfit for exact-genotype use. Evaluate at the resolution of the decision.

## Residual mass makes events unidentifiable

If an engine truncates its output and reports leftover probability as
[residual mass](../mathematical-foundations/probability-distributions.md), the
summed probability of the event is only a **lower bound** on the true event
probability. `event_calibration` therefore excludes such subjects and counts them:

```python
from immunomath import Residual

truncated = Distribution({("A*01", "A*02"): 0.5, ("A*02", "A*02"): 0.3, Residual(0.2): None})
# Lists 0.8 of the mass. The event probability is somewhere in [0.8, 1.0].
```

They appear in `n_undefined` with the reason, rather than being scored as if
0.8 were exact. This is the library's standing rule: **never silently
renormalize or guess**.

## What you learned

- Events are just sets of outcomes; `event_calibration` aggregates their
  probability and compares it with how often the event occurred.
- Exact-genotype concordance is blind to over-confidence; event calibration and
  the event Brier score are not.
- A forecast can be unfit at one resolution and acceptable at a coarser one.
  Evaluate at the resolution the decision depends on.
- Event probabilities are identifiable only for complete distributions.
