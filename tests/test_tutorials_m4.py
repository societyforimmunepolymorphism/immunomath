"""Run the Milestone 4 tutorials' code verbatim and check the numbers they quote."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("plotly")

TUTORIALS = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


def _run(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.chdir(tmp_path)  # tutorials write analysis files to the cwd
    blocks = re.findall(r"```python\n(.*?)```", (TUTORIALS / name).read_text(), re.DOTALL)
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, name, "exec"), namespace)
    return namespace


def test_tutorial_6_calibration(monkeypatch, tmp_path) -> None:
    ns = _run("06-calibration.md", monkeypatch, tmp_path)
    honest, over = ns["honest_cal"], ns["over_cal"]
    assert (round(honest.ece, 4), round(honest.mce, 4)) == (0.0179, 0.1675)
    assert (round(over.ece, 4), round(over.mce, 4)) == (0.1942, 0.2956)
    assert round(ns["eq"].ece, 4) == 0.1942
    full = ns["curve"].full_coverage
    assert (round(full.error_rate, 3), round(full.mean_log_loss, 3)) == (0.515, 1.161)
    assert (tmp_path / "overconfident.json").exists()
    assert (tmp_path / "overconfident.html").stat().st_size > 1_000_000


def test_tutorial_7_event_calibration(monkeypatch, tmp_path) -> None:
    from immunomath import event_calibration, risk_coverage, top_label_calibration

    ns = _run("07-event-calibration.md", monkeypatch, tmp_path)
    assert len(ns["genotypes"]) == 6 and len(ns["carries_a02"]) == 3
    over_event, over_geno = ns["event_cal"], ns["genotype_cal"]
    assert round(over_geno.ece, 4) == 0.2948
    assert (round(over_event.ece, 4), round(over_event.brier, 4)) == (0.1782, 0.2578)

    honest = ns["honest"]
    honest_event = event_calibration(honest, ns["carries_a02"])
    assert round(top_label_calibration(honest).ece, 4) == 0.0102
    assert (round(honest_event.ece, 4), round(honest_event.brier, 4)) == (0.0184, 0.2174)

    # Identical exact-genotype top-1 accuracy for both engines.
    acc = [1 - risk_coverage(ns[k]).full_coverage.error_rate for k in ("honest", "overconfident")]
    assert round(acc[0], 4) == round(acc[1], 4) == 0.4027
