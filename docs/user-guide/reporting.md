# Reports and Machine-Readable Output

ImmunoMath turns one cohort analysis into a **saved analysis record** (JSON) from
which an interactive HTML report and flat tables (CSV, Parquet) are produced.
The record holds everything needed to rebuild the report, so **a report can be
regenerated at any time from the saved JSON without the cohort, the predictions,
or any re-run inference.**

Plotting and Parquet output are optional extras:

```bash
pip install 'immunomath[reporting]'      # plotly + pyarrow
```

Everything else, including JSON and CSV output, needs only NumPy.

## The workflow

```python
from immunomath import analyze_cohort
from immunomath.reporting import (
    write_json,
    read_json,
    render_report,
    write_table,
    subject_table,
    calibration_table,
    risk_coverage_table,
)

result = analyze_cohort(
    cohort,
    label="engine X, panel 2024-06",
    events={"carries A*02": {("A*01", "A*02"), ("A*02", "A*02"), ("A*02", "A*03")}},
    n_bins=10,  # calibration bins (default 10)
    binning="fixed_width",  # or "equal_frequency"
    confidence_score=None,  # risk-coverage score; default = top probability
    seed=20241104,  # fixes the bootstrap
)

write_json(result, "analysis.json")  # the analysis record
render_report(result.to_dict(), "report.html")  # interactive report

# Later, with only the JSON in hand:
render_report(read_json("analysis.json"), "report-again.html")

write_table(subject_table(result), "subjects.parquet")  # one row per subject
write_table(calibration_table(result), "calibration.csv")
write_table(risk_coverage_table(result), "risk_coverage.csv")
```

`write_table` chooses the format from the file suffix: `.csv`, `.parquet` or
`.json`.

## Why the library never stamps the time

`analyze_cohort` does not record the current time unless you pass
`created="..."`. Identical inputs and the same seed therefore give a
byte-identical record, which is what makes results comparable and auditable. If
you want a timestamp in the report, supply it explicitly.

## What the analysis record contains

| Key | Content |
| --- | --- |
| `schema_version` | layout version (currently `1`); `render_report` rejects unknown versions |
| `metadata` | library version, label, optional `created`, and **every parameter** that influenced a number (seed, bootstrap settings, bins, binning, confidence-score kind, events, probability floor, quantile method) |
| `denominators` | `n_total`, `n_scored`, `n_finite`, `n_infinite`, `n_undefined` |
| `summary` | mean log score with bootstrap interval, Brier, median, standard deviation, maximum, loss quantiles |
| `undefined` | each excluded subject with the reason |
| `catastrophic` | threshold fractions, the exactly-zero category, worst subjects, loss concentration |
| `calibration` | top-label table, ECE, MCE |
| `event_calibration` | the same for each named event, plus the event Brier score |
| `risk_coverage` | the curve, with accuracy and log loss at each point |
| `subjects` | per-subject class, log score and reason |

Subject identifiers are stored as strings.

## Infinite and undefined values are preserved exactly

Standard JSON cannot represent `inf` or `nan`, and silently writing `null` would
erase precisely the values this library exists to surface. Non-finite floats are
therefore written as a tagged object and restored exactly on read:

```json
{"mean_log_score": {"__float__": "+inf"}}
```

The files are strict JSON (no `Infinity` or `NaN` tokens). A string that merely
*looks* like a token, such as a subject named `"+inf"`, is left alone, because
only the tagged object is decoded. CSV writes `inf` and `nan` as text; Parquet
stores them as native floating-point values.

## What the report contains

Every report states, in this order: analysis metadata and **warnings**; the
**denominators**; the primary results; the loss-distribution plot; the
extreme-loss diagnostics; top-label calibration with the reliability diagram; one
section per event; the risk–coverage figure and table; the list of excluded
subjects with reasons; and the **methods and assumptions**, with all recorded
parameters.

### Warnings are generated, not hand-written

`collect_warnings` derives them from the record so they cannot be forgotten:

- no seed was set, so bootstrap intervals are not reproducible;
- subjects had exactly zero probability on the truth, making the primary mean
  $+\infty$;
- subjects were excluded as undefined, with the resulting denominator;
- bootstrap resamples contained infinite losses, so a bound is $+\infty$;
- fewer than 30 identifiable subjects;
- calibration bins hold fewer than 10 subjects;
- a probability floor was applied;
- an event or calibration table could not be evaluated.

### The primary mean is never replaced

The headline is the untrimmed mean negative log score. If a trimmed mean is
included, it appears in a clearly marked box as a *descriptive sensitivity
statistic* that "does not replace the primary mean".

### Plots

| Figure | Shows |
| --- | --- |
| Reliability diagram | observed vs predicted, the diagonal, Wilson bars, marker area $\propto$ bin count |
| Risk–coverage | error rate and mean log loss vs coverage |
| Loss histogram | finite per-subject losses; the number of infinite and undefined subjects is stated in the title, never hidden |

Points that cannot be drawn (infinite loss) are **counted in the figure title**,
not silently omitted.

### Self-contained files

By default the Plotly library is embedded, so the HTML works offline and is
genuinely self-contained, at a cost of about 4 MB per file. Pass
`plotlyjs="cdn"` for a small file that needs network access when opened.

All text from your data (subject ids, reasons, labels) is HTML-escaped.

## Recording what you ran

The analysis record captures the *evaluation* settings. It cannot know which
inference engine, reference panel or software version produced the predictions,
nor how a callable `confidence_score` was defined. Record those yourself next to
the record: engine and version, reference-panel identifier and version, the
nomenclature release, the cohort and split definitions, and the source of the
truth. See the [reproducibility guide](../development/reproducibility.md).
