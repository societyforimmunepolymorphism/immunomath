"""Tests for the analysis record, machine-readable export and HTML report."""

from __future__ import annotations

import json
import math

import pytest

from immunomath import Cohort, Distribution, Residual, analyze_cohort
from immunomath.reporting import (
    calibration_table,
    collect_warnings,
    read_json,
    render_report,
    risk_coverage_table,
    subject_table,
    write_json,
    write_table,
)
from immunomath.reporting.export import decode_nonfinite, encode_nonfinite

D = Distribution


def make_cohort(extra_ids=("zero", "undef")) -> Cohort:
    ids = [f"s{i}" for i in range(12)]
    truths = ["G1", "G2", "G1", "G1", "G2", "G1", "G3", "G1", "G2", "G1", "G1", "G3"]
    rows = [
        {"G1": 0.6, "G2": 0.3, "G3": 0.1},
        {"G1": 0.3, "G2": 0.5, "G3": 0.2},
        {"G1": 0.7, "G2": 0.2, "G3": 0.1},
        {"G1": 0.5, "G2": 0.4, "G3": 0.1},
        {"G1": 0.2, "G2": 0.6, "G3": 0.2},
        {"G1": 0.8, "G2": 0.1, "G3": 0.1},
        {"G1": 0.3, "G2": 0.3, "G3": 0.4},
        {"G1": 0.9, "G2": 0.05, "G3": 0.05},
        {"G1": 0.4, "G2": 0.4, "G3": 0.2},
        {"G1": 0.55, "G2": 0.35, "G3": 0.1},
        {"G1": 0.65, "G2": 0.25, "G3": 0.1},
        {"G1": 0.1, "G2": 0.2, "G3": 0.7},
    ]
    dists = [D(r) for r in rows]
    if extra_ids:
        ids += list(extra_ids)
        truths += ["G2", "Gx"]
        dists += [
            D({"G1": 1.0, "G2": 0.0, "G3": 0.0}),
            D({"G1": 0.5, "G2": 0.3, Residual(0.2): None}),
        ]
    return Cohort(ids, truths, dists)


@pytest.fixture
def result():
    return analyze_cohort(
        make_cohort(), label="unit", events={"G1 present": {"G1"}}, seed=7, n_resamples=200
    )


class TestAnalysisRecord:
    def test_schema_and_denominators(self, result) -> None:
        d = result.to_dict()
        assert d["schema_version"] == 1
        assert d["denominators"] == {
            "n_total": 14,
            "n_scored": 13,
            "n_finite": 12,
            "n_infinite": 1,
            "n_undefined": 1,
        }
        assert d["metadata"]["parameters"]["seed"] == 7
        assert d["metadata"]["parameters"]["n_bins"] == 10
        assert d["metadata"]["created"] is None  # never auto-generated

    def test_contains_all_sections(self, result) -> None:
        d = result.to_dict()
        for key in (
            "summary",
            "catastrophic",
            "calibration",
            "event_calibration",
            "risk_coverage",
            "subjects",
            "undefined",
        ):
            assert key in d
        assert "G1 present" in d["event_calibration"]
        assert d["undefined"][0][0] == "undef"

    def test_deterministic_for_fixed_seed(self) -> None:
        a = analyze_cohort(make_cohort(), seed=3, n_resamples=200).to_dict()
        b = analyze_cohort(make_cohort(), seed=3, n_resamples=200).to_dict()
        assert json.dumps(encode_nonfinite(a)) == json.dumps(encode_nonfinite(b))

    def test_all_undefined_cohort_still_summarizes(self) -> None:
        cohort = Cohort(["a"], ["Gx"], [D({"G1": 0.5, Residual(0.5): None})])
        d = analyze_cohort(cohort, seed=0, n_resamples=50).to_dict()
        assert d["catastrophic"] is None
        assert d["denominators"]["n_scored"] == 0

    def test_parameters_record_confidence_score_kind(self) -> None:
        d = analyze_cohort(
            make_cohort(),
            confidence_score=lambda dist: max(dist.probabilities),
            seed=1,
            n_resamples=50,
        ).to_dict()
        assert d["metadata"]["parameters"]["confidence_score"] == "user-defined callable"


class TestJsonRoundTrip:
    def test_non_finite_values_survive(self, result, tmp_path) -> None:
        d = result.to_dict()
        assert d["summary"]["mean_log_score"] == math.inf
        path = write_json(result, tmp_path / "a.json")
        back = read_json(path)
        assert back["summary"]["mean_log_score"] == math.inf
        assert math.isnan(back["subjects"]["log_score"][-1])  # the undefined subject
        assert back["denominators"] == d["denominators"]

    def test_file_is_strict_json(self, result, tmp_path) -> None:
        text = write_json(result, tmp_path / "a.json").read_text()
        assert "Infinity" not in text and "NaN" not in text
        json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard {c}"))

    def test_string_that_looks_like_inf_is_not_mangled(self, tmp_path) -> None:
        record = {"subject": "+inf", "x": math.inf, "nested": [{"a": "nan"}, math.nan]}
        back = read_json(write_json(record, tmp_path / "r.json"))
        assert back["subject"] == "+inf"
        assert back["nested"][0]["a"] == "nan"
        assert back["x"] == math.inf
        assert math.isnan(back["nested"][1])

    def test_encode_decode_inverse(self) -> None:
        obj = {"a": [math.inf, -math.inf, 1.5], "b": {"c": math.nan}}
        back = decode_nonfinite(encode_nonfinite(obj))
        assert back["a"] == [math.inf, -math.inf, 1.5]
        assert math.isnan(back["b"]["c"])

    def test_read_rejects_non_object(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            read_json(path)


class TestTables:
    def test_csv(self, result, tmp_path) -> None:
        path = write_table(subject_table(result), tmp_path / "s.csv")
        lines = path.read_text().splitlines()
        assert lines[0] == "subject_id,score_class,log_score,reason"
        assert len(lines) == 15  # header + 14 subjects
        assert any(",infinite,inf," in line for line in lines)

    def test_parquet_round_trip_keeps_infinity(self, result, tmp_path) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        path = write_table(subject_table(result), tmp_path / "s.parquet")
        table = pq.read_table(path)
        assert table.num_rows == 14
        assert math.inf in table["log_score"].to_pylist()

    def test_json_table(self, result, tmp_path) -> None:
        path = write_table(calibration_table(result), tmp_path / "c.json")
        back = json.loads(path.read_text())
        assert len(back["count"]) == 10

    def test_event_and_risk_coverage_tables(self, result) -> None:
        assert set(calibration_table(result, "G1 present")) >= {"count", "ci_low"}
        assert "coverage" in risk_coverage_table(result)

    def test_unequal_columns_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="equal length"):
            write_table({"a": [1, 2], "b": [1]}, tmp_path / "x.csv")

    def test_unsupported_suffix_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            write_table({"a": [1]}, tmp_path / "x.xlsx")


class TestWarnings:
    def test_expected_warnings_present(self, result) -> None:
        text = " ".join(collect_warnings(result.to_dict()))
        assert "exactly zero probability" in text
        assert "not identifiable" in text
        assert "fewer than" in text  # sparse calibration bins

    def test_missing_seed_warned(self) -> None:
        d = analyze_cohort(make_cohort(extra_ids=()), seed=None, n_resamples=50).to_dict()
        assert any("not reproducible" in w for w in collect_warnings(d))

    def test_clean_analysis_has_no_data_warnings(self) -> None:
        cohort = Cohort(
            [f"s{i}" for i in range(40)],
            ["G1"] * 40,
            [D({"G1": 0.5 + 0.01 * (i % 40), "G2": 0.5 - 0.01 * (i % 40)}) for i in range(40)],
        )
        d = analyze_cohort(cohort, seed=1, n_resamples=50, n_bins=2).to_dict()
        assert collect_warnings(d) == []


class TestHtmlReport:
    def test_report_contains_required_sections(self, result, tmp_path) -> None:
        pytest.importorskip("plotly")
        out = render_report(result.to_dict(), tmp_path / "r.html", plotlyjs="cdn")
        text = out.read_text()
        for needle in (
            "Denominators",
            "Warnings",
            "Methods and assumptions",
            "Recorded parameters",
            "Extreme-loss diagnostics",
            "Calibration of the top prediction",
            "Event calibration: G1 present",
            "Risk–coverage",
            "Excluded (undefined) subjects",
        ):
            assert needle in text, needle
        assert "descriptive" not in text or "does not replace" in text

    def test_report_can_be_rebuilt_from_saved_json_only(self, result, tmp_path) -> None:
        pytest.importorskip("plotly")
        saved = write_json(result, tmp_path / "a.json")
        out = render_report(read_json(saved), tmp_path / "r.html", plotlyjs="cdn")
        text = out.read_text()
        assert "identifiable (finite + infinite)" in text
        assert "+∞" in text  # the infinite primary mean is shown, not hidden

    def test_html_is_escaped(self, tmp_path) -> None:
        pytest.importorskip("plotly")
        cohort = Cohort(
            ["<script>alert(1)</script>", "ok"],
            ["Gx", "G1"],
            [D({"G1": 0.5, Residual(0.5): None}), D({"G1": 0.7, "G2": 0.3})],
        )
        d = analyze_cohort(cohort, label="<b>bad</b>", seed=0, n_resamples=50).to_dict()
        text = render_report(d, tmp_path / "r.html", plotlyjs="cdn").read_text()
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
        assert "<b>bad</b>" not in text

    def test_inline_report_is_self_contained(self, result, tmp_path) -> None:
        pytest.importorskip("plotly")
        out = render_report(result.to_dict(), tmp_path / "r.html", plotlyjs="inline")
        text = out.read_text()
        assert out.stat().st_size > 1_000_000  # Plotly bundled in
        assert "cdn.plot.ly" not in text.split("<style>")[0] + text[:2000]

    def test_unknown_schema_rejected(self, result, tmp_path) -> None:
        pytest.importorskip("plotly")
        d = result.to_dict()
        d["schema_version"] = 99
        with pytest.raises(ValueError, match="schema_version"):
            render_report(d, tmp_path / "r.html")

    def test_trimmed_mean_labeled_descriptive(self, tmp_path) -> None:
        pytest.importorskip("plotly")
        cohort = make_cohort(extra_ids=())
        d = analyze_cohort(cohort, seed=0, n_resamples=50).to_dict()
        d["catastrophic"]["trimmed_mean_log_score"] = 0.5
        d["catastrophic"]["trimmed_fraction"] = 0.01
        text = render_report(d, tmp_path / "r.html", plotlyjs="cdn").read_text()
        assert "does not replace the primary mean" in text
