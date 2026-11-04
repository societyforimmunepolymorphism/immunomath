# Probability Distributions

The [`Distribution`][immunomath.Distribution] is the foundational abstraction of
ImmunoMath. This page gives its formal definition, validation rules, and the
semantics of residual mass and projection.

## Formal definition

A finite discrete probability distribution over a countable outcome space
$\Omega = \{\omega_1, \dots, \omega_n\}$ assigns to each outcome a probability
$p_i = P(\omega_i)$ satisfying

$$
p_i \ge 0 \quad\text{for all } i,
\qquad
\sum_{i=1}^{n} p_i = 1.
$$

ImmunoMath additionally requires every $p_i$ to be **finite** (no `inf`, no
`nan`).

## Plain-English explanation

A distribution is a complete accounting of belief: it lists the outcomes you
think are possible and how much probability you assign to each, and those
probabilities must add up to exactly one (within a tiny numerical tolerance).
ImmunoMath checks this at construction and refuses anything that does not
balance — it will not quietly rescale your numbers to make them sum to one.

## Validation rules

| Rule | Behavior on violation |
| --- | --- |
| Probabilities finite | `ValueError` |
| Probabilities nonnegative | `ValueError` |
| Sum to one within `atol` (default `1e-9`) | `ValueError` |
| At most one `Residual` entry | `ValueError` |

The tolerance is configurable per distribution via the `atol` parameter, but it
is never zero by default and never silently widened.

### Worked example

```python
>>> from immunomath import Distribution
>>> d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> d["G1"]
0.8
>>> d.is_complete
True
```

A distribution that does not sum to one is rejected:

```python
>>> Distribution({"G1": 0.7, "G2": 0.2})
Traceback (most recent call last):
    ...
ValueError: Probabilities must sum to 1 ...
```

## The three-way mass distinction

!!! important
    ImmunoMath distinguishes three situations that naïve code conflates. The
    distinction determines how an unlisted truth is scored.

1. **Omitted outcome.** The outcome is not a key. In a *complete* distribution
   this means its probability is exactly $0$.
2. **Explicit zero.** The outcome is present with value `0.0`: declared nameable
   but impossible.
3. **Unknown residual mass.** A [`Residual(m)`][immunomath.Residual] entry
   declares that mass $m \in (0, 1]$ belongs to unspecified, unlisted outcomes.
   Such a distribution is **open-world**.

### Why this matters: identifiability

For a specific outcome $\omega$ the probability $P(\omega)$ is **identifiable**
from the reported distribution only in these cases:

- $\omega$ is explicitly listed (its value is reported, even if a residual also
  exists); or
- $\omega$ is absent and the distribution is *complete* (then $P(\omega) = 0$).

If $\omega$ is absent from an *open* distribution, it might be hidden inside the
residual mass, so $P(\omega)$ is **not identifiable**. ImmunoMath refuses to
guess:

```python
>>> from immunomath import Distribution, Residual
>>> complete = Distribution({"G1": 0.8, "G2": 0.2})
>>> complete.probability("G99")       # omitted, complete -> 0
0.0
>>> open_d = Distribution({"G1": 0.8, Residual(0.2): None})
>>> open_d.probability("G1")          # explicit, scored normally
0.8
>>> open_d.probability("G99")         # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
KeyError: ...
```

**Event probabilities remain identifiable** even in an open distribution,
because summing listed outcomes never needs the residual:

```python
>>> open_d.event_probability({"G1"})
0.8
```

## Projection and mass preservation

A **projection** is a surjection $f: \Omega \to \Omega'$ onto a coarser outcome
space. The projected distribution aggregates probability over each fibre:

$$
P'(\omega') = \sum_{\omega \,:\, f(\omega) = \omega'} P(\omega).
$$

For a **total** projection this preserves total mass exactly:
$\sum_{\omega'} P'(\omega') = \sum_{\omega} P(\omega) = 1$. ImmunoMath verifies
this invariant with property-based tests.

```python
>>> fine = Distribution({"A*01:01": 0.5, "A*01:02": 0.3, "A*02:01": 0.2})
>>> coarse = fine.project(lambda a: a.split(":")[0])   # collapse to one field
>>> coarse["A*01"], coarse["A*02"]
(0.8, 0.2)
```

A **partial** projection drops some outcomes (the mapping returns `None`). The
dropped mass is re-expressed as residual mass, so the result is still a valid
distribution and the total is preserved:

```python
>>> partial = fine.project(lambda a: "A*01" if a.startswith("A*01") else None)
>>> partial["A*01"], round(partial.residual_mass, 10)
(0.8, 0.2)
```

!!! warning
    Do not compare evaluations at different resolutions without projecting both
    onto a common outcome space first. An evaluation at two-field resolution
    must operate on the *induced* two-field distribution.

## Explicit renormalization

Renormalization is available but opt-in, via
[`renormalize`][immunomath.distributions.Distribution.renormalize]. It rescales explicit
probabilities to sum to one and discards residual mass, yielding a complete
distribution. Use it only when mathematically justified (e.g. after
conditioning), and be aware that it asserts a closed world the original did not.

## Assumptions and failure modes

- **Closed vs. open world.** Scoring an unlisted truth is `+inf` for a complete
  distribution but *undefined* for an open one. Choose the representation that
  matches your inference engine's actual semantics.
- **Hashability.** Outcomes must be hashable. Tuples of strings/ints are fine;
  mutable containers are not.
- **Numerical tolerance.** The sum-to-one check uses an absolute tolerance.
  Distributions assembled from many tiny terms may need a slightly larger
  `atol`; set it explicitly rather than relying on luck.

## References

- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd
  ed.). Wiley.
