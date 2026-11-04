"""Machine-readable export: JSON, CSV and Parquet.

*   **JSON** holds the full analysis record. Standard JSON cannot represent
    ``inf`` or ``nan`` and Python's non-standard ``Infinity`` token is not
    portable, so non-finite floats are written as a tagged object
    ``{"__float__": "+inf" | "-inf" | "nan"}`` and restored exactly on read.
    Nothing is silently replaced by ``null``.
*   **CSV** and **Parquet** hold flat tables (per-subject scores, calibration
    bins, risk-coverage points). CSV needs only the standard library; Parquet
    needs the optional ``pyarrow`` dependency (``pip install immunomath[reporting]``).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "calibration_table",
    "decode_nonfinite",
    "encode_nonfinite",
    "read_json",
    "risk_coverage_table",
    "subject_table",
    "write_json",
    "write_table",
]

_TAG = "__float__"
_TOKENS = {"+inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def encode_nonfinite(obj: Any) -> Any:
    """Recursively replace non-finite floats with tagged JSON-safe objects."""
    if isinstance(obj, float) and not math.isfinite(obj):
        token = "nan" if math.isnan(obj) else ("+inf" if obj > 0 else "-inf")
        return {_TAG: token}
    if isinstance(obj, Mapping):
        return {k: encode_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_nonfinite(v) for v in obj]
    return obj


def decode_nonfinite(obj: Any) -> Any:
    """Invert :func:`encode_nonfinite`."""
    if isinstance(obj, Mapping):
        if set(obj) == {_TAG} and obj[_TAG] in _TOKENS:
            return _TOKENS[obj[_TAG]]
        return {k: decode_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_nonfinite(v) for v in obj]
    return obj


def _as_dict(analysis: Any) -> dict[str, Any]:
    return analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)


def write_json(analysis: Any, path: str | Path) -> Path:
    """Write an analysis record to JSON.

    Parameters
    ----------
    analysis : AnalysisResult or dict
        An :class:`~immunomath.analysis.AnalysisResult` or its ``to_dict()``.
    path : str or Path
        Destination file.

    Returns
    -------
    Path
        The written path.

    Examples
    --------
    >>> import tempfile, math
    >>> from pathlib import Path
    >>> record = {"x": math.inf, "y": [1.0, math.nan]}
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = write_json(record, Path(d) / "r.json")
    ...     back = read_json(out)
    >>> back["x"], math.isnan(back["y"][1])
    (inf, True)
    """
    out = Path(path)
    out.write_text(
        json.dumps(encode_nonfinite(_as_dict(analysis)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a record written by :func:`write_json`, restoring ``inf``/``nan``."""
    decoded = decode_nonfinite(json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return decoded


def subject_table(analysis: Any) -> dict[str, list[Any]]:
    """Per-subject scores as flat columns (``subject_id``, ``score_class``, ...)."""
    return {k: list(v) for k, v in _as_dict(analysis)["subjects"].items()}


def calibration_table(analysis: Any, which: str = "top_label") -> dict[str, list[Any]]:
    """Calibration bins as flat columns.

    Parameters
    ----------
    analysis : AnalysisResult or dict
    which : str, optional
        ``"top_label"`` or the name of an event.
    """
    record = _as_dict(analysis)
    cal = record["calibration"] if which == "top_label" else record["event_calibration"][which]
    columns = [
        "index",
        "lower",
        "upper",
        "count",
        "mean_predicted",
        "observed_rate",
        "ci_low",
        "ci_high",
    ]
    return {c: [b[c] for b in cal["bins"]] for c in columns}


def risk_coverage_table(analysis: Any) -> dict[str, list[Any]]:
    """Risk-coverage operating points as flat columns."""
    points = _as_dict(analysis)["risk_coverage"]["points"]
    columns = [
        "threshold",
        "n_retained",
        "coverage",
        "accuracy",
        "error_rate",
        "mean_log_loss",
        "n_infinite_retained",
    ]
    return {c: [p[c] for p in points] for c in columns}


def write_table(columns: Mapping[str, Sequence[Any]], path: str | Path) -> Path:
    """Write a flat table to ``.csv``, ``.parquet`` or ``.json`` by file suffix.

    Parameters
    ----------
    columns : mapping
        Column name to equal-length sequence of values.
    path : str or Path
        Destination; the suffix selects the format.

    Raises
    ------
    ValueError
        If columns differ in length or the suffix is not supported.
    ImportError
        If Parquet is requested and ``pyarrow`` is not installed.
    """
    out = Path(path)
    names = list(columns)
    lengths = {len(columns[n]) for n in names}
    if len(lengths) > 1:
        raise ValueError(f"All columns must have equal length; got {sorted(lengths)}.")
    n_rows = lengths.pop() if lengths else 0
    suffix = out.suffix.lower()

    if suffix == ".csv":
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(names)
            for i in range(n_rows):
                writer.writerow([columns[n][i] for n in names])
    elif suffix == ".json":
        out.write_text(
            json.dumps(encode_nonfinite({n: list(columns[n]) for n in names}), indent=2),
            encoding="utf-8",
        )
    elif suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
            raise ImportError(
                "Parquet output requires pyarrow: pip install 'immunomath[reporting]'."
            ) from exc
        pq.write_table(pa.table({n: list(columns[n]) for n in names}), out)
    else:
        raise ValueError(f"Unsupported table format {suffix!r}; use .csv, .parquet or .json.")
    return out
