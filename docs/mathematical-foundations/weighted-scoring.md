# Weighted Scoring (TRS-Aware)

Conventional cohort means weight every observation equally. When most
observations are near-deterministic and a minority are highly uncertain, the
uncertain minority can be diluted to statistical invisibility — even though they
are often where modelling choices matter most. This page defines
**covariate-weighted** proper scoring, the **paired** comparison of two methods,
and **TRS-stratified** validation with a group-balanced macro-average.

These tools are exposed by [`weighted_mean_loss`][immunomath.weighted_mean_loss],
[`compare_methods_weighted`][immunomath.compare_methods_weighted], and
[`stratified_comparison`][immunomath.stratified_comparison]. The kernel is
domain-independent; the TRS weighting function is a separate optional adapter,
[`trs_weights`][immunomath.trs_weights].

!!! warning "Not a utility metric"
    Neither the weighted score nor the macro-average establishes clinical
    utility. They measure probabilistic accuracy with a chosen emphasis.
    Direct donor-ranking evaluation is a separate downstream task.

## Weighted mean loss

For per-observation proper losses $L_i$ (log or Brier) and externally supplied
strictly positive weights $w_i$,

$$
L_w = \frac{\sum_i w_i L_i}{\sum_i w_i}.
$$

With uniform weights this is exactly the ordinary mean, so weighted scoring is a
strict generalization and the unweighted result is **always reported alongside**
it. Weights are validated (finite, strictly positive, correct length) but never
normalized or clipped.

### Effective sample size

Unequal weights concentrate influence in fewer observations. The Kish effective
sample size quantifies this:

$$
N_{\mathrm{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

It equals $n$ for uniform weights and falls toward 1 as weight concentrates on a
single observation. $N_{\mathrm{eff}}$ is scale-invariant: multiplying every
weight by a positive constant leaves both $L_w$ and $N_{\mathrm{eff}}$ unchanged.

## Conditional propriety

A scoring rule is **strictly proper** when honest reporting is the unique
minimizer of expected loss. Multiplying a strictly proper loss by a positive
constant $w_i$ does not move that minimizer. Therefore, if each $w_i$ is a
**fixed, strictly positive function of pre-registered covariates** — and in
particular does **not** depend on

- the observed true genotype,
- whether an imputation "succeeded", or
- either model's posterior probabilities,

then the weighted average remains a proper criterion **conditional on the
covariates**. A weight may depend on observed typing resolution; it may not peek
at the outcome or at a model's predictions. Weights that do so break the
guarantee and are a statistical error, not a modelling preference.

## The TRS weighting function (optional adapter)

For a normalized typing-resolution score $r_i \in [0, 1]$ (higher = better
resolved),

$$
w_i = 1 + \lambda\,(1 - r_i)^{\gamma}, \qquad \lambda \ge 0,\ \gamma > 0.
$$

- $\lambda = 0$ gives uniform weights (recovers ordinary scoring exactly).
- Larger $\lambda$ upweights poorly resolved observations more.
- $\gamma > 1$ concentrates the extra weight on the lowest-resolution tail.

The parameters are explicit and mandatory; the adapter imposes no scientific
default and does **not** silently normalize an unknown TRS scale (values outside
$[0,1]$ are rejected). TRS is computed outside ImmunoMath and supplied as a
plain number.

## Paired comparison of two methods

For an incumbent and a candidate scored on the **same** observations, truths,
and weights, the per-observation improvement is

$$
\Delta_i = L_{i,\mathrm{incumbent}} - L_{i,\mathrm{candidate}}.
$$

A **positive** $\Delta_i$ means the candidate assigns the truth a lower loss (an
improvement). [`compare_methods_weighted`][immunomath.compare_methods_weighted]
reports the conventional and weighted mean loss of each method, the conventional
and weighted mean improvement, and the counts of observations improved,
worsened, and unchanged, together with the full distribution of $\Delta_i$.

### Infinite and undefined losses

A difference is **defined** only when both methods yield an identifiable score.

- **Undefined in either method** (truth unidentifiable in an open distribution)
  — excluded and reported separately.
- **Indeterminate $\infty - \infty$** (both methods assign the truth zero
  probability) — excluded; never silently treated as a zero improvement.
- **One-sided infinity** ($\infty - \text{finite}$ or $\text{finite} - \infty$)
  — a genuine, defined improvement or deterioration, and retained.

## TRS-stratified validation and the balanced macro-average

Given a caller-supplied stratum label per observation (for example coarse TRS
bands `low` / `intermediate` / `high`),
[`stratified_comparison`][immunomath.stratified_comparison] reports, for each
nonempty stratum, the mean log and Brier scores of both methods and the paired
improvement. Across strata it reports the **group-balanced macro-average**

$$
L_{\mathrm{balanced}} = \frac{1}{K} \sum_{k=1}^{K} \bar L_k,
$$

where $K$ is the number of **nonempty** strata and $\bar L_k$ is the ordinary
within-stratum mean. This is reported **alongside** the conventional
donor-weighted cohort mean. The objective is to detect a candidate that improves
the donor-weighted mean while **degrading** a small, poorly resolved stratum:
the two summaries then disagree.

### Empty strata

A declared stratum with zero observations has no mean to contribute. It is
listed explicitly in `empty_strata` and **dropped from the macro-average
denominator** $K$, rather than being charged a zero or a `NaN` that would
silently bias the balanced score.

## Worked demonstration

`examples/trs_weighted_comparison.py` builds a synthetic cohort of 60 high-TRS
donors (near-deterministic; candidate slightly sharper) and 8 low-TRS donors
(uncertain; candidate worse). The four views disagree exactly as intended:

| View | Log improvement (incumbent − candidate) | Verdict |
|------|------------------------------------------|---------|
| Conventional (donor-weighted) | **+0.13** | candidate preferred |
| TRS-weighted | +0.03 | shrinks toward neutral |
| TRS-balanced macro | **−0.03** | incumbent preferred (reversed) |
| Per stratum: high (n=60) | +0.18 | candidate better |
| Per stratum: low (n=8) | −0.25 | candidate worse |

The donor-weighted mean favors the candidate because the large high-TRS majority
each improves; the TRS-balanced macro-average reverses the verdict, exposing the
low-TRS regression that the overall mean conceals.
