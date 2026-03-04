from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
from jinja2 import Template

from fmeh.reporting.plots import plot_confusion_matrix, plot_metric_bars


def _table_text(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return empty_message
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def _load_results(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    db_path = run_dir / "results.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Run database not found: {db_path}")
    conn = duckdb.connect(str(db_path))
    df = conn.execute("SELECT * FROM sample_results").df()
    conn.close()
    return df


def _expand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics_records = []
    for payload in df.get("metrics_json", []):
        try:
            metrics_records.append(json.loads(payload or "{}"))
        except Exception:
            metrics_records.append({})
    metrics_df = pd.json_normalize(metrics_records)
    metrics_df.columns = [f"metric_{c}" for c in metrics_df.columns]
    return pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)


def _aggregate(full_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in full_df.columns if c.startswith("metric_")]
    numeric = metric_cols + ["latency_sec"]
    if not numeric:
        return pd.DataFrame()
    agg = (
        full_df.groupby(["task", "model_id", "prompt_version"], dropna=False)[numeric]
        .mean(numeric_only=True)
        .reset_index()
    )
    return agg


def create_report(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _load_results(run_dir)
    if raw_df.empty:
        raise RuntimeError("No rows found in sample_results")

    df = _expand_metrics(raw_df)
    agg = _aggregate(df)

    # Plots
    parse_plot = artifacts_dir / "parse_valid_rate.png"
    lat_plot = artifacts_dir / "latency.png"
    plot_metric_bars(agg, "metric_parse_valid", parse_plot)
    plot_metric_bars(agg, "metric_accuracy", artifacts_dir / "accuracy.png")
    plot_metric_bars(agg, "metric_f1", artifacts_dir / "extraction_f1.png")
    plot_metric_bars(agg, "metric_rougeL", artifacts_dir / "rougeL.png")
    plot_metric_bars(agg, "latency_sec", lat_plot)

    cls_df = df[df["task"] == "classification"].copy()
    target_labels = cls_df["target_text"].astype(str).str.lower().tolist()
    pred_labels = []
    for payload in cls_df.get("parsed_output", []):
        try:
            pred = json.loads(payload or "{}")
        except Exception:
            pred = {}
        pred_labels.append(str(pred.get("label", "maybe")).lower())
    cm_path = artifacts_dir / "classification_confusion_matrix.png"
    plot_confusion_matrix(target_labels, pred_labels, cm_path)

    parse_errors = (
        df[df["parse_valid"] == False]["parse_error"]  # noqa: E712
        .value_counts()
        .head(10)
        .reset_index(name="count")
        .rename(columns={"index": "parse_error"})
    )

    sort_cols = [
        c
        for c in ["metric_parse_valid", "metric_accuracy", "metric_f1", "metric_rougeL"]
        if c in df.columns
    ]
    failure_df = (
        df.sort_values(by=sort_cols, ascending=[True] * len(sort_cols), na_position="last")
        if sort_cols
        else df.copy()
    )
    failures = failure_df.head(10)[
        [
            "task",
            "model_id",
            "prompt_version",
            "example_id",
            "input",
            "raw_output",
            "parse_error",
        ]
    ].to_dict(orient="records")

    template = Template(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>FMEH Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #f5f5f5; }
    img { max-width: 100%; margin-bottom: 16px; }
    pre { white-space: pre-wrap; background: #fafafa; padding: 8px; border: 1px solid #eee; }
  </style>
</head>
<body>
  <h1>Foundation Model Eval Harness Report</h1>
  <h2>Aggregate Metrics (task × model × prompt_version)</h2>
  {{ agg_table }}

  <h2>Parse Error Analysis</h2>
  {{ parse_table }}

  <h2>Plots</h2>
  {% for img in images %}
    <h3>{{ img }}</h3>
    <img src="artifacts/{{ img }}" alt="{{ img }}" />
  {% endfor %}

  <h2>Top Failures</h2>
  {% for f in failures %}
    <h3>{{ f.task }} | {{ f.model_id }} | {{ f.prompt_version }} | {{ f.example_id }}</h3>
    <strong>Parse error:</strong> {{ f.parse_error }}
    <h4>Input</h4>
    <pre>{{ f.input }}</pre>
    <h4>Raw output</h4>
    <pre>{{ f.raw_output }}</pre>
  {% endfor %}
</body>
</html>
        """.strip()
    )

    agg_table = (
        agg.to_html(index=False, float_format=lambda x: f"{x:.4f}")
        if not agg.empty
        else "<p>No aggregate metrics.</p>"
    )
    parse_table = (
        parse_errors.to_html(index=False)
        if not parse_errors.empty
        else "<p>No parse errors recorded.</p>"
    )

    images = [
        p.name
        for p in [
            parse_plot,
            artifacts_dir / "accuracy.png",
            artifacts_dir / "extraction_f1.png",
            artifacts_dir / "rougeL.png",
            cm_path,
        ]
        if p.exists()
    ]

    html = template.render(
        agg_table=agg_table,
        parse_table=parse_table,
        failures=failures,
        images=images,
    )

    report_html = run_dir / "report.html"
    report_md = run_dir / "report.md"

    report_html.write_text(html, encoding="utf-8")

    lines = [
        "# FMEH Report",
        "",
        "## Aggregate Metrics",
        _table_text(agg, "No aggregate metrics."),
        "",
        "## Parse Errors",
        _table_text(parse_errors, "No parse errors."),
        "",
        "## Top Failures",
    ]
    for f in failures:
        lines.extend(
            [
                f"### {f['task']} | {f['model_id']} | {f['prompt_version']} | {f['example_id']}",
                f"Parse error: {f['parse_error']}",
                "```text",
                f["raw_output"][:1000],
                "```",
            ]
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    return {"html": report_html, "md": report_md}
