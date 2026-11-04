# API Reference — Analysis Record and Reporting

See the [reporting guide](../user-guide/reporting.md).

## Analysis record

::: immunomath.analysis
    options:
      members:
        - analyze_cohort
        - AnalysisResult

## Export

::: immunomath.reporting.export
    options:
      members:
        - write_json
        - read_json
        - write_table
        - subject_table
        - calibration_table
        - risk_coverage_table

## HTML report

::: immunomath.reporting.report
    options:
      members:
        - render_report
        - collect_warnings

## Figures

::: immunomath.reporting.plots
    options:
      members:
        - reliability_figure
        - risk_coverage_figure
        - loss_histogram_figure
