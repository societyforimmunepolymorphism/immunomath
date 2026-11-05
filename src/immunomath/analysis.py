"""A reproducible, serializable record of one cohort analysis.

:func:`analyze_cohort` runs the Milestone 2-4 assessments for a single cohort
and returns an :class:`AnalysisResult`. Its :meth:`~AnalysisResult.to_dict`
produces a plain, JSON-compatible structure (``schema_version`` 1) that holds
**everything needed to render a report**, so reports can be regenerated from the
saved results without the cohort, the predictions, or any re-run inference.

Non-finite numbers (``inf``, ``nan``) are legitimate values in this library; the
dict keeps them as Python floats and the writers in
:mod:`immunomath.reporting` encode them losslessly.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Literal

from ._version import __version__
from .calibration import (
    DEFAULT_N_BINS,
    CalibrationResult,
    event_calibration,
    top_label_calibration,
)
from .catastrophic import CatastrophicReport, catastrophic_error_report
from .cohort import Cohort, CohortSummary
from .risk_coverage import ConfidenceScore, RiskCoverageResult, risk_coverage
from .statistics import DEFAULT_CONFIDENCE, DEFAULT_N_RESAMPLES

__all__ = ["SCHEMA_VERSION", "AnalysisResult", "analyze_cohort", "to_plain"]

#: Version of the dict layout produced by :meth:`AnalysisResult.to_dict`.
SCHEMA_VERSION: int = 1


def to_plain(obj: Any) -> Any:
    """Convert dataclasses, enums, tuples and mappings to plain JSON-style data.

    Mapping keys become strings; tuples become lists; dataclasses become dicts;
    enums become their values; other non-primitive objects (for example subject
    identifiers that are tuples of strings) become ``str(obj)``. Floats are left
    untouched, including non-finite ones.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return to_plain(obj.value)
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


@dataclass(frozen=True)
class AnalysisResult:
    """The complete result of :func:`analyze_cohort`.

    Attributes
    ----------
    label : str
        Name of the analysis (for example the algorithm and panel).
    parameters : dict
        Every parameter that influenced the numbers (seed, bin count, ...).
    summary : CohortSummary
        The Milestone 2 cohort summary.
    catastrophic : CatastrophicReport or None
        Tail diagnostics, or ``None`` if no subject was identifiable.
    calibration : CalibrationResult
        Top-label calibration.
    event_calibrations : dict[str, CalibrationResult]
        Calibration of each user-defined event, keyed by event name.
    risk_coverage : RiskCoverageResult
        Risk-coverage curve.
    created : str or None
        Optional caller-supplied timestamp. The library never inserts the clock
        time itself, so identical inputs give identical outputs.
    subject_ids : tuple[Hashable, ...]
        Subject identifiers in cohort order (used to label per-subject tables).
    """

    label: str
    parameters: dict[str, Any]
    summary: CohortSummary
    catastrophic: CatastrophicReport | None
    calibration: CalibrationResult
    event_calibrations: dict[str, CalibrationResult]
    risk_coverage: RiskCoverageResult
    created: str | None = None
    subject_ids: tuple[Hashable, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record (``schema_version`` 1).

        Subject identifiers are converted to strings.
        """
        s = self.summary
        subjects: dict[str, list[Any]] = {
            "subject_id": [],
            "score_class": [],
            "log_score": [],
            "reason": [],
        }
        ids = self.subject_ids
        for sub in s.subject_scores:
            subjects["subject_id"].append(str(ids[sub.index]))
            subjects["score_class"].append(sub.score_class.value)
            subjects["log_score"].append(sub.score if sub.score is not None else math.nan)
            subjects["reason"].append(sub.reason or "")
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "immunomath_version": __version__,
                "label": self.label,
                "created": self.created,
                "parameters": to_plain(self.parameters),
            },
            "denominators": {
                "n_total": s.n_total,
                "n_scored": s.n_scored,
                "n_finite": s.n_finite,
                "n_infinite": s.n_infinite,
                "n_undefined": s.n_undefined,
            },
            "summary": {
                "mean_log_score": s.mean_log_score,
                "median_log_score": s.median_log_score,
                "std_log_score": s.std_log_score,
                "max_log_score": s.max_log_score,
                "mean_brier": s.mean_brier,
                "mean_log_score_ci": to_plain(s.mean_log_score_ci),
                "loss_quantiles": to_plain(s.loss_quantiles),
                "quantile_method": s.quantile_method,
            },
            "undefined": [[str(sid), reason] for sid, reason in s.undefined_reasons],
            "catastrophic": to_plain(self.catastrophic),
            "calibration": to_plain(self.calibration),
            "event_calibration": to_plain(self.event_calibrations),
            "risk_coverage": to_plain(self.risk_coverage),
            "subjects": subjects,
        }


def analyze_cohort(
    cohort: Cohort,
    *,
    label: str = "analysis",
    events: Mapping[str, Collection[Hashable]] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    binning: Literal["fixed_width", "equal_frequency"] = "fixed_width",
    confidence_score: ConfidenceScore | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    bootstrap_method: Literal["percentile", "bca"] = "percentile",
    seed: int | None = None,
    floor: float | None = None,
    created: str | None = None,
) -> AnalysisResult:
    """Run the cohort, tail, calibration and risk-coverage assessments together.

    Parameters
    ----------
    cohort : Cohort
        The cohort to analyze.
    label : str, optional
        Name recorded in the result.
    events : mapping, optional
        ``name -> outcomes`` for each event to calibrate.
    n_bins, binning
        Calibration binning (default 10 fixed-width bins).
    confidence_score : callable or mapping, optional
        User-defined confidence score for risk-coverage (default: top
        probability). A callable or mapping is recorded only by kind, not value;
        record its definition in your experiment manifest.
    confidence_level : float, optional
        Level for bootstrap and per-bin Wilson intervals.
    n_resamples, bootstrap_method, seed
        Bootstrap settings. Pass an integer ``seed`` for a reproducible result;
        the report warns when it is ``None``.
    floor : float, optional
        Optional probability floor (recorded; off by default).
    created : str, optional
        Timestamp to record. Never generated automatically.

    Returns
    -------
    AnalysisResult

    Examples
    --------
    >>> from immunomath import Cohort, Distribution, analyze_cohort
    >>> cohort = Cohort(["a", "b"], ["G1", "G2"],
    ...                 [Distribution({"G1": 0.8, "G2": 0.2}),
    ...                  Distribution({"G1": 0.3, "G2": 0.7})])
    >>> result = analyze_cohort(cohort, label="demo", seed=0, n_resamples=100)
    >>> result.to_dict()["denominators"]["n_scored"]
    2
    """
    summary = cohort.summarize(
        floor=floor,
        confidence=confidence_level,
        n_resamples=n_resamples,
        bootstrap_method=bootstrap_method,
        seed=seed,
    )
    catastrophic: CatastrophicReport | None
    try:
        catastrophic = catastrophic_error_report(cohort, floor=floor)
    except ValueError:
        catastrophic = None  # no identifiable subjects

    calibration = top_label_calibration(
        cohort, n_bins=n_bins, binning=binning, confidence=confidence_level
    )
    event_results = {
        name: event_calibration(
            cohort,
            outcomes,
            label=name,
            n_bins=n_bins,
            binning=binning,
            confidence=confidence_level,
        )
        for name, outcomes in (events or {}).items()
    }
    rc = risk_coverage(cohort, confidence=confidence_score, floor=floor)

    if confidence_score is None:
        score_description = "top_probability"
    elif callable(confidence_score):
        score_description = "user-defined callable"
    else:
        score_description = "user-defined per-subject mapping"

    parameters: dict[str, Any] = {
        "seed": seed,
        "n_resamples": n_resamples,
        "bootstrap_method": bootstrap_method,
        "confidence_level": confidence_level,
        "n_bins": n_bins,
        "binning": binning,
        "confidence_score": score_description,
        "events": {k: sorted(map(str, v)) for k, v in (events or {}).items()},
        "probability_floor": floor,
        "quantile_method": summary.quantile_method,
    }
    result = AnalysisResult(
        label=label,
        parameters=parameters,
        summary=summary,
        catastrophic=catastrophic,
        calibration=calibration,
        event_calibrations=event_results,
        risk_coverage=rc,
        created=created,
        subject_ids=cohort.subject_ids,
    )
    return result
