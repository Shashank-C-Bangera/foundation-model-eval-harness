from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fmeh.export.powerbi_export import export_run


def _create_fixture_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": "2026-03-04T10:00:00Z",
            "experiment": "fixture_export",
            "example_id": "cls_1",
            "task": "classification",
            "model_id": "model_a",
            "prompt_version": "v1",
            "input": "Question 1",
            "target_text": "yes",
            "raw_output": '{"label":"yes"}',
            "parsed_output": '{"label":"yes"}',
            "parse_valid": True,
            "repaired": False,
            "error": "",
            "y_true_norm": "yes",
            "y_pred_norm": "yes",
            "metrics_json": '{"accuracy": 1.0}',
            "latency_sec": 0.12,
        },
        {
            "timestamp": "2026-03-04T10:01:00Z",
            "experiment": "fixture_export",
            "example_id": "sum_1",
            "task": "summarization",
            "model_id": "model_a",
            "prompt_version": "v1",
            "input": "Long clinical paragraph",
            "target_text": "Short summary",
            "raw_output": "Summary text",
            "parsed_output": '{"summary":"Summary text"}',
            "parse_valid": True,
            "repaired": False,
            "error": "",
            "y_true_norm": "",
            "y_pred_norm": "",
            "metrics_json": '{"rougeL": 0.42, "bertscore_f1": 0.57}',
            "latency_sec": 0.3,
        },
        {
            "timestamp": "2026-03-04T10:02:00Z",
            "experiment": "fixture_export",
            "example_id": "ext_1",
            "task": "extraction",
            "model_id": "model_b",
            "prompt_version": "v2",
            "input": "Patient uses aspirin for flu",
            "target_text": "",
            "raw_output": '{"diseases":["flu"],"chemicals":["aspirin"]}',
            "parsed_output": '{"diseases":["flu"],"chemicals":["aspirin"]}',
            "parse_valid": True,
            "repaired": True,
            "error": "",
            "y_true_norm": "",
            "y_pred_norm": "",
            "metrics_json": '{"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}',
            "latency_sec": 0.4,
        },
        {
            "timestamp": "2026-03-04T10:03:00Z",
            "experiment": "fixture_export",
            "example_id": "ext_2",
            "task": "extraction",
            "model_id": "model_b",
            "prompt_version": "v2",
            "input": "No entities found",
            "target_text": "",
            "raw_output": "",
            "parsed_output": "{}",
            "parse_valid": False,
            "repaired": False,
            "error": "parse failure",
            "y_true_norm": "",
            "y_pred_norm": "",
            "metrics_json": '{"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}',
            "latency_sec": 0.5,
        },
    ]
    fixture_df = pd.DataFrame(rows)
    conn = duckdb.connect(str(run_dir / "results.duckdb"))
    conn.register("fixture_df", fixture_df)
    conn.execute("CREATE TABLE sample_results AS SELECT * FROM fixture_df")
    conn.close()


def test_powerbi_export_writes_expected_csvs(tmp_path: Path) -> None:
    run_dir = tmp_path / "fixture_run"
    _create_fixture_run(run_dir)
    out = export_run(run_dir=run_dir, out_dir=run_dir / "exports", max_failures=10)

    expected_files = {
        "run_summary": "run_summary.csv",
        "model_metrics": "model_metrics.csv",
        "model_task_metrics": "model_task_metrics.csv",
        "failure_examples": "failure_examples.csv",
    }
    for key, filename in expected_files.items():
        assert key in out
        assert out[key].name == filename
        assert out[key].exists()

    run_summary = pd.read_csv(out["run_summary"])
    model_metrics = pd.read_csv(out["model_metrics"])
    model_task_metrics = pd.read_csv(out["model_task_metrics"])
    failures = pd.read_csv(out["failure_examples"])

    assert len(run_summary) == 1
    assert len(model_metrics) > 0
    assert len(model_task_metrics) > 0
    assert len(failures) > 0

    assert set(
        [
            "run_name",
            "last_update",
            "total_samples",
            "n_models",
            "tasks",
            "best_model_overall",
            "overall_score",
            "parse_valid_rate",
            "error_rate",
            "repair_rate",
        ]
    ).issubset(run_summary.columns)

    assert set(
        [
            "run_name",
            "model_id",
            "score_overall",
            "classification_macro_f1",
            "summarization_bertscore_f1",
            "extraction_f1",
            "parse_valid_rate",
            "error_rate",
            "repair_rate",
            "n_total",
            "n_cls",
            "n_sum",
            "n_ext",
        ]
    ).issubset(model_metrics.columns)

    assert set(
        [
            "run_name",
            "model_id",
            "task",
            "prompt_version",
            "default_metric",
            "accuracy",
            "macro_f1",
            "rougeL",
            "bertscore_f1",
            "precision",
            "recall",
            "extraction_f1",
            "exact_match",
            "parse_valid_rate",
            "error_rate",
            "repair_rate",
            "n",
        ]
    ).issubset(model_task_metrics.columns)

    assert set(
        [
            "run_name",
            "model_id",
            "task",
            "prompt_version",
            "example_id",
            "parse_valid",
            "repaired",
            "error",
            "input_trunc",
            "output_trunc",
        ]
    ).issubset(failures.columns)
