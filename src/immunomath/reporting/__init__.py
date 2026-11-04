"""Reports and machine-readable output (optional ``reporting`` extra).

Importing this package needs only the standard library and NumPy. Plotting and
Parquet output import ``plotly`` / ``pyarrow`` lazily and raise a clear
``ImportError`` if they are missing:

    pip install 'immunomath[reporting]'
"""

from __future__ import annotations

from .export import (
    calibration_table,
    read_json,
    risk_coverage_table,
    subject_table,
    write_json,
    write_table,
)
from .report import collect_warnings, render_report

__all__ = [
    "calibration_table",
    "collect_warnings",
    "read_json",
    "render_report",
    "risk_coverage_table",
    "subject_table",
    "write_json",
    "write_table",
]
