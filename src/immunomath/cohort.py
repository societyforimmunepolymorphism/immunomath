"""Subject-identified cohorts and their statistical summaries.

Milestone 2 evaluates *collections* of predictions. A :class:`Cohort` pairs
stable subject identifiers with observed truths and predicted
:class:`~immunomath.distributions.Distribution` objects, plus optional
per-subject metadata.

The central discipline of this module is **explicit accounting**. Classification
of each subject into ``finite`` / ``infinite`` / ``undefined`` and the primary
proper-score mean are delegated to
:func:`~immunomath.scoring.score_cohort_accounting` (so the three-way logic and
denominators live in one place). :class:`Cohort` adds stable subject IDs,
metadata, and a richer :class:`CohortSummary` with dispersion statistics,
loss-distribution quantiles, the mean Brier score, and a bootstrap confidence
interval for the primary mean.

See Also
--------
immunomath.scoring.score_cohort_accounting
    The underlying finite/infinite/undefined accounting and primary mean.
immunomath.quantiles
    The Hyndman-Fan quantile machinery used for the loss-distribution quantiles.
immunomath.statistics.bootstrap_ci
    The seeded subject-level bootstrap used for the mean's confidence interval.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .distributions import Distribution
from .quantiles import QuantileMethod, loss_quantile
from .scoring import ScoreClass, SubjectScore, brier_score, score_cohort_accounting
from .statistics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_RESAMPLES,
    BootstrapResult,
    bootstrap_ci,
)

__all__ = [
    "Cohort",
    "CohortSummary",
]


class Cohort:
    """An ordered, subject-identified collection of predictions and truths.

    Parameters
    ----------
    subject_ids : sequence of Hashable
        Stable, unique identifiers, one per subject.
    truths : sequence of Hashable
        Observed true outcomes, aligned with ``subject_ids``.
    distributions : sequence of Distribution
        Predicted distributions, aligned with ``subject_ids``.
    metadata : mapping, optional
        Optional per-subject metadata as a mapping from subject id to an
        arbitrary attribute mapping (e.g. ``{"S1": {"population": "A"}}``). Used
        by population-stratified analysis in a later milestone; stored verbatim.

    Raises
    ------
    ValueError
        If the three sequences differ in length, are empty, or ``subject_ids``
        contains duplicates.
    TypeError
        If any element of ``distributions`` is not a :class:`Distribution`.

    Examples
    --------
    >>> from immunomath import Distribution, Cohort
    >>> cohort = Cohort(
    ...     subject_ids=["S1", "S2"],
    ...     truths=["G1", "G2"],
    ...     distributions=[
    ...         Distribution({"G1": 0.9, "G2": 0.1}),
    ...         Distribution({"G1": 0.3, "G2": 0.7}),
    ...     ],
    ... )
    >>> len(cohort)
    2
    """

    __slots__ = ("_distributions", "_ids", "_metadata", "_truths")

    def __init__(
        self,
        subject_ids: Sequence[Hashable],
        truths: Sequence[Hashable],
        distributions: Sequence[Distribution],
        *,
        metadata: Mapping[Hashable, Mapping[str, Any]] | None = None,
    ) -> None:
        n = len(subject_ids)
        if not (n == len(truths) == len(distributions)):
            raise ValueError(
                "subject_ids, truths, and distributions must have equal length; got "
                f"{len(subject_ids)}, {len(truths)}, {len(distributions)}."
            )
        if n == 0:
            raise ValueError("A cohort must contain at least one subject.")
        if len(set(subject_ids)) != n:
            raise ValueError("subject_ids must be unique.")
        for i, dist in enumerate(distributions):
            if not isinstance(dist, Distribution):
                raise TypeError(
                    f"distributions[{i}] is {type(dist).__name__}, expected Distribution."
                )
        self._ids: tuple[Hashable, ...] = tuple(subject_ids)
        self._truths: tuple[Hashable, ...] = tuple(truths)
        self._distributions: tuple[Distribution, ...] = tuple(distributions)
        self._metadata: dict[Hashable, Mapping[str, Any]] = (
            dict(metadata) if metadata is not None else {}
        )

    def __len__(self) -> int:
        """Return the number of subjects."""
        return len(self._ids)

    @property
    def subject_ids(self) -> tuple[Hashable, ...]:
        """The subject identifiers, in order."""
        return self._ids

    @property
    def truths(self) -> tuple[Hashable, ...]:
        """The observed true outcomes, in order."""
        return self._truths

    @property
    def distributions(self) -> tuple[Distribution, ...]:
        """The predicted distributions, in order."""
        return self._distributions

    @property
    def metadata(self) -> dict[Hashable, Mapping[str, Any]]:
        """A copy of the per-subject metadata mapping."""
        return dict(self._metadata)

    def subject_scores(self) -> tuple[SubjectScore, ...]:
        """Return per-subject scores and classifications (see :class:`ScoreClass`)."""
        return score_cohort_accounting(list(self._truths), list(self._distributions)).subjects

    def summarize(
        self,
        *,
        strict: bool = False,
        floor: float | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        bootstrap_method: str = "percentile",
        quantile_levels: Sequence[float] = (0.25, 0.5, 0.75, 0.95, 0.99),
        quantile_method: QuantileMethod = QuantileMethod.LINEAR,
        seed: int | None = None,
    ) -> CohortSummary:
        """Compute a full cohort summary with explicit denominators.

        Classification and the primary mean come from
        :func:`~immunomath.scoring.score_cohort_accounting`; this method layers
        dispersion statistics, loss quantiles, the mean Brier score, and a
        bootstrap CI on top.

        Parameters
        ----------
        strict : bool, optional
            If ``True``, raise :class:`ValueError` when any subject is undefined.
        floor : float, optional
            Optional probability floor for the log score.
        confidence : float, optional
            Bootstrap confidence level (default ``0.95``).
        n_resamples : int, optional
            Number of bootstrap resamples (default ``2000``).
        bootstrap_method : {"percentile", "bca"}, optional
            Bootstrap interval method.
        quantile_levels : sequence of float, optional
            Quantile levels of the individual-loss distribution to report.
        quantile_method : QuantileMethod, optional
            Quantile convention (default Hyndman-Fan type 7 / linear).
        seed : int or None, optional
            Seed for reproducible bootstrap.

        Returns
        -------
        CohortSummary
            The summary, including the proper-score mean over identifiable
            subjects and explicit counts.

        Raises
        ------
        ValueError
            In ``strict`` mode, if any subject is undefined.
        """
        accounting = score_cohort_accounting(
            list(self._truths), list(self._distributions), floor=floor, strict=strict
        )

        identifiable = [s for s in accounting.subjects if s.score_class is not ScoreClass.UNDEFINED]
        finite_losses = [
            s.score
            for s in identifiable
            if s.score_class is ScoreClass.FINITE and s.score is not None
        ]
        # Losses over identifiable subjects (finite values plus +inf for zeros).
        identifiable_losses = [(s.score if s.score is not None else math.inf) for s in identifiable]

        n_scored = accounting.n_scored
        n_infinite = accounting.n_infinite

        # Dispersion over finite subjects only.
        median_log_score = (
            loss_quantile(finite_losses, 0.5, method=quantile_method) if finite_losses else math.nan
        )
        if len(finite_losses) >= 2:
            fmean = math.fsum(finite_losses) / len(finite_losses)
            variance = math.fsum((x - fmean) ** 2 for x in finite_losses) / (len(finite_losses) - 1)
            std_log_score = math.sqrt(variance)
        else:
            std_log_score = math.nan

        loss_quantiles = (
            {
                q: loss_quantile(identifiable_losses, q, method=quantile_method)
                for q in quantile_levels
            }
            if identifiable_losses
            else {}
        )

        if n_scored == 0:
            max_log_score = math.nan
        elif n_infinite > 0:
            max_log_score = math.inf
        else:
            max_log_score = max(finite_losses)

        # Mean Brier over identifiable subjects (bounded, always finite).
        brier_values: list[float] = []
        for sid_idx, truth, dist in zip(
            range(len(self._ids)), self._truths, self._distributions, strict=True
        ):
            cls = accounting.subjects[sid_idx].score_class
            if cls is ScoreClass.UNDEFINED:
                continue
            brier_values.append(brier_score(truth, dist))
        mean_brier = math.fsum(brier_values) / len(brier_values) if brier_values else math.nan

        ci: BootstrapResult | None
        if n_scored >= 1:
            ci = bootstrap_ci(
                identifiable_losses,
                statistic="mean",
                confidence=confidence,
                n_resamples=n_resamples,
                method=bootstrap_method,  # type: ignore[arg-type]
                seed=seed,
            )
        else:
            ci = None

        undefined_reasons = tuple(
            (self._ids[idx], reason) for idx, reason in accounting.undefined_reasons
        )

        return CohortSummary(
            n_total=accounting.n_total,
            n_scored=n_scored,
            n_undefined=accounting.n_undefined,
            n_infinite=n_infinite,
            mean_log_score=accounting.primary_mean,
            median_log_score=median_log_score,
            std_log_score=std_log_score,
            max_log_score=max_log_score,
            loss_quantiles=loss_quantiles,
            mean_brier=mean_brier,
            mean_log_score_ci=ci,
            subject_scores=accounting.subjects,
            undefined_reasons=undefined_reasons,
            quantile_method=quantile_method.value,
        )


@dataclass(frozen=True)
class CohortSummary:
    """A cohort assessment with explicit denominators and classifications.

    The headline figure is :attr:`mean_log_score`, the ordinary (untrimmed) mean
    negative log score over **identifiable** subjects (``n_scored`` = finite +
    infinite). It is ``+inf`` whenever any identifiable subject has infinite
    loss, and ``nan`` only when every subject is undefined. Undefined subjects
    are excluded from the mean and reported via :attr:`n_undefined` and
    :attr:`undefined_reasons`.

    Attributes
    ----------
    n_total : int
        Total subjects in the cohort.
    n_scored : int
        Identifiable subjects (finite + infinite). The denominator of
        :attr:`mean_log_score`.
    n_undefined : int
        Subjects whose score is undefined (unidentifiable truth).
    n_infinite : int
        Identifiable subjects with infinite loss (zero probability on truth).
    mean_log_score : float
        Primary proper-score summary over identifiable subjects; ``+inf`` if any
        identifiable subject is infinite; ``nan`` if ``n_scored == 0``.
    median_log_score, std_log_score, max_log_score : float
        Descriptive statistics. Median and std are over the **finite** subjects
        (``std`` needs ≥2); ``max`` is ``+inf`` if any identifiable subject is
        infinite.
    loss_quantiles : dict[float, float]
        Quantiles of the identifiable individual-loss distribution (may be
        ``+inf`` in the upper tail when infinite losses are present).
    mean_brier : float
        Mean Brier score over identifiable subjects (always finite, bounded).
    mean_log_score_ci : BootstrapResult or None
        Bootstrap CI for :attr:`mean_log_score` over identifiable subjects.
    subject_scores : tuple[SubjectScore, ...]
        Per-subject scores and classifications (from the accounting).
    undefined_reasons : tuple[tuple[Hashable, str | None], ...]
        ``(subject_id, reason)`` for each undefined subject.
    quantile_method : str
        The quantile convention used.
    n_finite : int
        Derived: ``n_scored - n_infinite``.
    """

    n_total: int
    n_scored: int
    n_undefined: int
    n_infinite: int
    mean_log_score: float
    median_log_score: float
    std_log_score: float
    max_log_score: float
    loss_quantiles: dict[float, float]
    mean_brier: float
    mean_log_score_ci: BootstrapResult | None
    subject_scores: tuple[SubjectScore, ...]
    undefined_reasons: tuple[tuple[Hashable, str | None], ...]
    quantile_method: str
    n_finite: int = field(init=False)

    def __post_init__(self) -> None:
        """Derive ``n_finite`` from the identifiable counts."""
        object.__setattr__(self, "n_finite", self.n_scored - self.n_infinite)
