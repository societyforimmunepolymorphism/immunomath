# Calibration

A probabilistic forecast is **calibrated** if, among all cases where it reports
probability $p$ for something, that something happens a fraction $p$ of the
time. If an imputation engine says "80% confident" for a thousand genotypes, about
800 of them should be right.

Calibration is a *diagnostic*, not a proper scoring rule. It tells you whether
stated confidence can be taken at face value; it does not tell you whether the
forecasts were informative. A forecaster who always reports the population base
rate is perfectly calibrated and useless. Use calibration **alongside** the
[logarithmic](logarithmic-score.md) and [Brier](brier-score.md) scores, never
instead of them.

## Two diagnostics

### Top-label calibration

[`top_label_calibration`][immunomath.top_label_calibration] compares the
probability of the highest-probability outcome (the *confidence*) with how often
that outcome was correct.

For subject $i$ let $c_i = \max_k p_{ik}$ be the confidence and $y_i \in [0,1]$ the
correctness, equal to the fractional top-1 credit of the truth (see
[tie handling](proper-vs-descriptive.md)): $1$ if the truth is the unique mode,
$0$ if it is not, and $1/t$ if it is one of $t$ tied modes. Credit does not
depend on candidate order.

### Event calibration

[`event_calibration`][immunomath.event_calibration] calibrates **any event** you
define as a set of outcomes $E$, such as "the genotype contains allele X". The
predicted probability is the aggregate

$$
\hat p_i(E) = \sum_{\omega \in E} P_i(\omega),
$$

and the observed indicator is $y_i = \mathbf 1[g_i \in E]$. This matters wherever
allele presence matters more than exact genotype identity, as in
pharmacogenomics. It also reports the event's **Brier score**,
$\frac1N\sum_i(\hat p_i - y_i)^2$, which *is* a strictly proper score for the
binary event.

!!! important "Event probability is identifiable only for complete distributions"
    If a distribution carries [residual mass](probability-distributions.md), the
    listed mass of $E$ is only a *lower bound* on its true probability. Such
    subjects are **excluded and counted** in `n_undefined`, never scored as if the
    lower bound were exact.

## Binning

Both diagnostics group subjects into bins by predicted probability. The method
is always explicit and recorded in the result.

| Strategy | Behavior |
| --- | --- |
| `fixed_width` (default) | `n_bins` equal intervals of $[0,1]$. Empty bins are kept with count 0. A probability of exactly 1 falls in the last bin. |
| `equal_frequency` | Subjects are ranked by predicted probability and split into `n_bins` groups of near-equal size (sizes differ by at most one). Fewer subjects than bins gives one bin per subject. Ties are split by input order, so a boundary can fall inside a run of equal confidences. |

The default is **10 bins**. Fixed-width bins make the reliability diagram easy to
read but leave sparse bins at the extremes; equal-frequency bins keep every bin
equally well-estimated but their widths vary.

## Per-bin uncertainty: Wilson intervals

Each bin reports a $95\%$ **Wilson score interval** for its observed frequency.
For $s$ successes in $n$ trials, $\hat p = s/n$ and normal quantile $z$:

$$
\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}
     {1 + \frac{z^2}{n}} .
$$

Unlike the textbook Wald interval, it never leaves $[0,1]$ and behaves sensibly
for small $n$ and for $\hat p$ near 0 or 1. When tie credit makes $s$ fractional
the interval is an approximation.

```python
>>> from immunomath.calibration import wilson_interval
>>> low, high = wilson_interval(8, 10)
>>> round(low, 3), round(high, 3)
(0.49, 0.943)
```

A bin of 10 subjects with 8 correct is compatible with true accuracy anywhere
from 49% to 94%: **read bins with few subjects as nearly uninformative.** Reports
warn whenever a bin holds fewer than 10.

## ECE and MCE

With $N$ evaluated subjects, bin sizes $n_b$, mean confidence $\mathrm{conf}_b$
and observed frequency $\mathrm{acc}_b$:

$$
\mathrm{ECE} = \sum_b \frac{n_b}{N}\,\bigl|\mathrm{acc}_b - \mathrm{conf}_b\bigr|,
\qquad
\mathrm{MCE} = \max_b \bigl|\mathrm{acc}_b - \mathrm{conf}_b\bigr| .
$$

ECE is **weighted by bin count**, so a bin with one subject barely moves it; MCE
is the worst bin and is therefore noisy when bins are small. Empty bins
contribute nothing.

### Worked example

Four subjects with confidences $0.9, 0.9, 0.9, 0.8$ and correctness
$1, 1, 0, 1$. Bin $[0.9,1.0]$ holds three subjects with $\mathrm{acc} = 2/3$;
bin $[0.8,0.9)$ holds one with $\mathrm{acc}=1$:

$$
\mathrm{ECE} = \tfrac34\left|\tfrac23 - 0.9\right| + \tfrac14\left|1 - 0.8\right|
             = 0.175 + 0.05 = 0.225 .
$$

```python
>>> from immunomath import Cohort, Distribution, top_label_calibration
>>> cohort = Cohort(
...     ["a", "b", "c", "d"], ["G1", "G1", "G2", "G2"],
...     [Distribution({"G1": 0.9, "G2": 0.1})] * 3 + [Distribution({"G1": 0.2, "G2": 0.8})],
... )
>>> round(top_label_calibration(cohort).ece, 3)
0.225
```

### ECE depends on the binning only weakly, and sometimes not at all

If every populated bin has $\mathrm{conf}_b > \mathrm{acc}_b$ (a uniformly
over-confident forecaster), the absolute values drop out and

$$
\mathrm{ECE} = \bar c - \bar y ,
$$

the overall mean confidence minus overall accuracy, **for any binning**. In
[Tutorial 6](../tutorials/06-calibration.md) the fixed-width and equal-frequency
ECE agree to six digits for this reason. Binning only matters when bins disagree
in sign: sparse bins whose gaps flip sign cancel under the signed sum but not
under ECE, which is why ECE can drift slightly between binnings.

## Interpretation

| Reliability curve | Meaning |
| --- | --- |
| on the diagonal | calibrated |
| below the diagonal | **over-confident**: reported probability exceeds the observed frequency |
| above the diagonal | under-confident |

For genotype imputation, over-confidence is the dangerous direction: it is what
produces confidently wrong calls. See
[catastrophic-error analysis](catastrophic-error.md).

## Assumptions and failure modes

- **Binning dependence.** ECE and MCE change with `n_bins` and `binning`; always
  report them.
- **Small bins.** MCE and individual points are dominated by noise when bins hold
  few subjects. Prefer equal-frequency bins for small cohorts.
- **Not proper.** Minimizing ECE can be done by degrading informativeness. Do not
  optimize it; use it to diagnose.
- **Top-label only sees the mode.** A forecast can be top-label calibrated while
  its non-top probabilities are badly wrong; event calibration probes those.
- **Identifiability.** Subjects with unidentifiable truth, or (for events) with
  residual mass, are excluded and counted, never imputed.
- **Independence.** The Wilson intervals assume subjects are independent draws.
  Related individuals, or panels reused between training and evaluation, violate
  this.

## References

- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. *Monthly Weather Review*, 78(1), 1–3.
- DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and evaluation of
  forecasters. *The Statistician*, 32, 12–22.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of
  modern neural networks. *ICML*. (ECE and MCE.)
- Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *JASA*, 22(158), 209–212.
- Nixon, J., et al. (2019). Measuring calibration in deep learning. *CVPR
  Workshops*. (Binning sensitivity of ECE.)
