"""Paired comparison of two cohorts on the same subjects.

Milestone 3 compares two evaluations — algorithm A versus B, reference panel A
versus B, software version X versus Y — on the **same individuals**. The unit
of comparison is the per-subject difference in negative log loss,

.. math::

    d_i = L_A(i) - L_B(i),

so that a **negative** difference means A assigns the truth higher probability
(A is better for subject *i*).

Two headline summaries are reported as co-equals (per the scientific decision):

1.  the **mean paired log-loss difference** with a paired bootstrap confidence
    interval, and
2.  the **proportion of subjects improved versus worsened** by moving from B to
    A.

Design commitments
------------------
*   **Same subjects.** Subjects are aligned by identifier. A mismatch in the
    subject sets is an **error by default**; opt in to the common subjects with
    ``on="intersection"``.
*   **Strict common outcome space.** Comparing evaluations at different genotype
    resolutions is refused unless both cohorts share a common outcome space.
    Reconcile explicitly with a projection (see :func:`compare_cohorts`'s
    ``projection`` argument) rather than comparing incommensurable numbers.
*   **Explicit infinite / undefined handling.** A per-subject difference is only
    defined when *both* arms yield an identifiable score. The indeterminate
    ``inf - inf`` case (both arms assign the truth zero probability) and any
    subject undefined in either arm are excluded from the paired statistics and
    reported separately with counts.
*   **Raw paired CIs only.** No multiple-comparison adjustment is applied; the
    interval is the ordinary paired bootstrap interval.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

from .cohort import Cohort
from .diagnostics import top_k_accuracy
from .distributions import Distribution, Projection
from .scoring import negative_log_score
from .statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_RESAMPLES,
    BootstrapResult,
    paired_bootstrap_ci,
)

__all__ = [
    "PairedComparison",
    "compare_cohorts",
]


@dataclass(frozen=True)
class PairedComparison:
    """Result of a paired comparison of cohort A against cohort B.

    The sign convention is ``d = L_A - L_B``: a **negative** mean difference and
    a predominance of *improved* subjects both indicate that **A is better**
    than B (A assigns the truth lower loss).

    Attributes
    ----------
    label_a, label_b : str
        Human-readable names for the two arms.
    n_common : int
        Number of subjects present (and aligned) in both arms.
    n_paired : int
        Number of common subjects with a **defined** paired difference (both
        arms finite). The denominator of the paired statistics.
    n_both_undefined_or_infinite : int
        Common subjects excluded from the paired statistics because the
        difference is not defined (either arm undefined, or an ``inf - inf``).
    exclusion_reasons : tuple[tuple[Hashable, str], ...]
        ``(subject_id, reason)`` for each excluded common subject.
    mean_log_loss_diff : float
        Headline #1: mean of ``d_i`` over the ``n_paired`` subjects. Negative
        favors A. ``nan`` if ``n_paired == 0``.
    mean_log_loss_diff_ci : BootstrapResult or None
        Paired bootstrap CI for ``mean_log_loss_diff`` (raw, unadjusted).
    n_improved, n_worsened, n_tied : int
        Headline #2 (counts): subjects for which A improves on B
        (``d_i < -tolerance``), worsens (``d_i > tolerance``), or ties.
    proportion_improved, proportion_worsened, proportion_tied : float
        Headline #2 (proportions), over ``n_paired``.
    largest_improvement : tuple[Hashable, float] or None
        ``(subject_id, d_i)`` with the most negative difference (A best vs B).
    largest_worsening : tuple[Hashable, float] or None
        ``(subject_id, d_i)`` with the most positive difference (A worst vs B).
    subject_differences : tuple[tuple[Hashable, float], ...]
        The per-subject ``(subject_id, d_i)`` for all paired subjects, in order.
    concordance_diff_top1 : float
        Difference in top-1 concordance, ``acc_A - acc_B``, over paired subjects
        (positive favors A). Uses fractional-credit top-1.
    tolerance : float
        The absolute tolerance below which a difference counts as a tie.
    """

    label_a: str
    label_b: str
    n_common: int
    n_paired: int
    n_both_undefined_or_infinite: int
    exclusion_reasons: tuple[tuple[Hashable, str], ...]
    mean_log_loss_diff: float
    mean_log_loss_diff_ci: BootstrapResult | None
    n_improved: int
    n_worsened: int
    n_tied: int
    proportion_improved: float
    proportion_worsened: float
    proportion_tied: float
    largest_improvement: tuple[Hashable, float] | None
    largest_worsening: tuple[Hashable, float] | None
    subject_differences: tuple[tuple[Hashable, float], ...]
    concordance_diff_top1: float
    tolerance: float


def _outcome_universe(cohort: Cohort) -> frozenset[Hashable]:
    """Return the set of all explicitly listed outcomes across a cohort."""
    universe: set[Hashable] = set()
    for dist in cohort.distributions:
        universe.update(dist.outcomes)
    return frozenset(universe)


def _project_cohort(cohort: Cohort, projection: Projection[Hashable, Hashable]) -> Cohort:
    """Project every distribution in a cohort, projecting truths consistently."""
    new_truths: list[Hashable] = []
    new_dists: list[Distribution] = []
    for truth, dist in zip(cohort.truths, cohort.distributions, strict=True):
        projected_truth = projection(truth)
        if projected_truth is None:
            raise ValueError(
                f"The projection maps the truth {truth!r} to None (dropped). A "
                "truth must remain representable in the common outcome space; "
                "use a projection that retains all observed truths."
            )
        new_truths.append(projected_truth)
        new_dists.append(dist.project(projection))
    return Cohort(
        subject_ids=list(cohort.subject_ids),
        truths=new_truths,
        distributions=new_dists,
        metadata=cohort.metadata,
    )


def compare_cohorts(
    cohort_a: Cohort,
    cohort_b: Cohort,
    *,
    label_a: str = "A",
    label_b: str = "B",
    on: Literal["exact", "intersection"] = "exact",
    projection: Projection[Hashable, Hashable] | None = None,
    floor: float | None = None,
    tolerance: float = 1e-12,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    bootstrap_method: Literal["percentile", "bca"] = "percentile",
    seed: int | None = None,
) -> PairedComparison:
    r"""Compare cohort A against cohort B on the same subjects (paired).

    Parameters
    ----------
    cohort_a, cohort_b : Cohort
        The two cohorts. They must describe the **same individuals** (aligned by
        subject id) and, after any projection, the **same outcome space**.
    label_a, label_b : str, optional
        Names for the two arms, used in the result.
    on : {"exact", "intersection"}, optional
        Subject-set policy. ``"exact"`` (default) requires identical subject-id
        sets and raises on any mismatch. ``"intersection"`` restricts the
        comparison to subjects present in both cohorts.
    projection : Projection, optional
        If given, both cohorts (and their truths) are projected onto a common
        coarser outcome space before comparison. Required whenever the two
        cohorts' outcome spaces differ; see Notes on strict enforcement.
    floor : float, optional
        Optional probability floor passed to the scorer in both arms.
    tolerance : float, optional
        Absolute tolerance below which a paired difference counts as a tie.
        Defaults to ``1e-12``.
    confidence, n_resamples, bootstrap_method, seed
        Paired bootstrap settings (see
        :func:`~immunomath.statistics.paired_bootstrap_ci`). Intervals are
        **raw** — no multiple-comparison adjustment is applied.

    Returns
    -------
    PairedComparison
        Both headline summaries (mean difference with CI; improved/worsened
        proportions) plus per-subject diagnostics.

    Raises
    ------
    ValueError
        If subjects do not match under ``on="exact"``; if the same subject has
        different truths in the two arms; if the outcome spaces differ and no
        reconciling ``projection`` is supplied (strict enforcement); or if no
        subjects remain to compare.

    Notes
    -----
    **Strict common-outcome-space enforcement.** Two evaluations at different
    genotype resolutions produce probabilities over different outcome spaces;
    their log losses are not directly comparable. ``compare_cohorts`` therefore
    refuses to proceed when the cohorts' outcome universes differ, unless a
    ``projection`` is supplied that maps both onto a shared space. This prevents
    silently comparing, e.g., two-field against one-field predictions.

    **Sign convention.** ``d_i = L_A(i) - L_B(i)``; negative means A is better.

    Examples
    --------
    >>> from immunomath import Cohort, Distribution, compare_cohorts
    >>> a = Cohort(["s1", "s2"], ["G1", "G2"],
    ...            [Distribution({"G1": 0.8, "G2": 0.2}),
    ...             Distribution({"G1": 0.3, "G2": 0.7})])
    >>> b = Cohort(["s1", "s2"], ["G1", "G2"],
    ...            [Distribution({"G1": 0.6, "G2": 0.4}),
    ...             Distribution({"G1": 0.5, "G2": 0.5})])
    >>> cmp = compare_cohorts(a, b, label_a="A", label_b="B", seed=0, n_resamples=200)
    >>> cmp.mean_log_loss_diff < 0        # A assigns the truth higher probability
    True
    >>> cmp.n_improved
    2
    """
    # ---- align subjects ------------------------------------------------- #
    ids_a = list(cohort_a.subject_ids)
    ids_b = list(cohort_b.subject_ids)
    set_a, set_b = set(ids_a), set(ids_b)

    if on == "exact":
        if set_a != set_b:
            only_a = sorted(map(repr, set_a - set_b))[:5]
            only_b = sorted(map(repr, set_b - set_a))[:5]
            raise ValueError(
                "Cohorts do not contain the same subjects. With on='exact' the "
                "subject-id sets must be identical. "
                f"Only in {label_a}: {only_a}; only in {label_b}: {only_b}. "
                "Pass on='intersection' to compare the common subjects."
            )
        common_ids = ids_a  # identical set; preserve A's order
    elif on == "intersection":
        common = set_a & set_b
        if not common:
            raise ValueError(
                f"Cohorts {label_a} and {label_b} share no subjects; nothing to compare."
            )
        common_ids = [sid for sid in ids_a if sid in common]
    else:
        raise ValueError(f"on must be 'exact' or 'intersection', got {on!r}.")

    # ---- strict common outcome space ------------------------------------ #
    if projection is not None:
        cohort_a = _project_cohort(cohort_a, projection)
        cohort_b = _project_cohort(cohort_b, projection)

    universe_a = _outcome_universe(cohort_a)
    universe_b = _outcome_universe(cohort_b)
    if universe_a != universe_b:
        only_a = sorted(map(repr, universe_a - universe_b))[:5]
        only_b = sorted(map(repr, universe_b - universe_a))[:5]
        raise ValueError(
            "The two cohorts are defined over different outcome spaces, so their "
            "log losses are not directly comparable (for example, different "
            "genotype resolutions). Outcomes only in "
            f"{label_a}: {only_a}; only in {label_b}: {only_b}. "
            "Supply a `projection` that maps both cohorts onto a common outcome "
            "space before comparing."
        )

    # ---- per-subject paired differences --------------------------------- #
    idx_a = {sid: i for i, sid in enumerate(cohort_a.subject_ids)}
    idx_b = {sid: i for i, sid in enumerate(cohort_b.subject_ids)}

    diffs: list[tuple[Hashable, float]] = []
    losses_a: list[float] = []
    losses_b: list[float] = []
    excluded: list[tuple[Hashable, str]] = []
    hits_a = 0.0
    hits_b = 0.0

    for sid in common_ids:
        ia, ib = idx_a[sid], idx_b[sid]
        truth_a, truth_b = cohort_a.truths[ia], cohort_b.truths[ib]
        if truth_a != truth_b:
            raise ValueError(
                f"Subject {sid!r} has different truths in the two cohorts "
                f"({truth_a!r} vs {truth_b!r}); a paired comparison requires the "
                "same observed truth per subject."
            )
        dist_a, dist_b = cohort_a.distributions[ia], cohort_b.distributions[ib]

        # Classify identifiability per arm.
        try:
            la = negative_log_score(truth_a, dist_a, floor=floor)
            a_defined = True
        except KeyError:
            la = math.nan
            a_defined = False
        try:
            lb = negative_log_score(truth_b, dist_b, floor=floor)
            b_defined = True
        except KeyError:
            lb = math.nan
            b_defined = False

        if not a_defined or not b_defined:
            which = []
            if not a_defined:
                which.append(label_a)
            if not b_defined:
                which.append(label_b)
            excluded.append((sid, f"truth not identifiable in {' and '.join(which)}"))
            continue
        if math.isinf(la) and math.isinf(lb):
            excluded.append(
                (sid, "indeterminate inf - inf (both arms assign truth zero probability)")
            )
            continue

        d = la - lb
        diffs.append((sid, d))
        losses_a.append(la)
        losses_b.append(lb)
        hits_a += top_k_accuracy(truth_a, dist_a, k=1)
        hits_b += top_k_accuracy(truth_b, dist_b, k=1)

    n_common = len(common_ids)
    n_paired = len(diffs)
    n_excluded = len(excluded)

    if n_paired == 0:
        return PairedComparison(
            label_a=label_a,
            label_b=label_b,
            n_common=n_common,
            n_paired=0,
            n_both_undefined_or_infinite=n_excluded,
            exclusion_reasons=tuple(excluded),
            mean_log_loss_diff=math.nan,
            mean_log_loss_diff_ci=None,
            n_improved=0,
            n_worsened=0,
            n_tied=0,
            proportion_improved=math.nan,
            proportion_worsened=math.nan,
            proportion_tied=math.nan,
            largest_improvement=None,
            largest_worsening=None,
            subject_differences=(),
            concordance_diff_top1=math.nan,
            tolerance=tolerance,
        )

    d_values = [d for _, d in diffs]
    mean_diff = math.fsum(d_values) / n_paired

    n_improved = sum(1 for d in d_values if d < -tolerance)
    n_worsened = sum(1 for d in d_values if d > tolerance)
    n_tied = n_paired - n_improved - n_worsened

    largest_improvement = min(diffs, key=lambda kv: kv[1])
    largest_worsening = max(diffs, key=lambda kv: kv[1])
    # If no subject actually improved/worsened, report None for that extreme.
    if largest_improvement[1] >= -tolerance:
        largest_improvement = None  # type: ignore[assignment]
    if largest_worsening[1] <= tolerance:
        largest_worsening = None  # type: ignore[assignment]

    ci = paired_bootstrap_ci(
        losses_a,
        losses_b,
        confidence=confidence,
        n_resamples=n_resamples,
        method=bootstrap_method,
        seed=seed,
    )

    concordance_diff = (hits_a - hits_b) / n_paired

    return PairedComparison(
        label_a=label_a,
        label_b=label_b,
        n_common=n_common,
        n_paired=n_paired,
        n_both_undefined_or_infinite=n_excluded,
        exclusion_reasons=tuple(excluded),
        mean_log_loss_diff=mean_diff,
        mean_log_loss_diff_ci=ci,
        n_improved=n_improved,
        n_worsened=n_worsened,
        n_tied=n_tied,
        proportion_improved=n_improved / n_paired,
        proportion_worsened=n_worsened / n_paired,
        proportion_tied=n_tied / n_paired,
        largest_improvement=largest_improvement,
        largest_worsening=largest_worsening,
        subject_differences=tuple(diffs),
        concordance_diff_top1=concordance_diff,
        tolerance=tolerance,
    )
