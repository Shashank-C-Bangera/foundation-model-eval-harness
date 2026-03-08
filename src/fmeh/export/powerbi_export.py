from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sklearn.metrics import f1_score

DEFAULT_METRIC_BY_TASK: dict[str, tuple[str, str | None]] = {
    "classification": ("macro_f1", "accuracy"),
    "summarization": ("bertscore_f1", "rougeL"),
    "extraction": ("extraction_f1", "exact_match"),
}

TASK_ORDER = ["classification", "summarization", "extraction"]
TASK_TO_COUNT_COL = {
    "classification": "n_cls",
    "summarization": "n_sum",
    "extraction": "n_ext",
}

MODEL_METRICS_COLUMNS = [
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

MODEL_TASK_METRICS_COLUMNS = [
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

RUN_SUMMARY_COLUMNS = [
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

FAILURE_EXAMPLES_COLUMNS = [
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

NUMERIC_COLS = [
    "accuracy",
    "macro_f1",
    "rougeL",
    "bertscore_f1",
    "precision",
    "recall",
    "extraction_f1",
    "f1",
    "exact_match",
    "pred_length",
    "latency_ms",
    "latency_sec",
]


def _json_obj(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_label(raw: Any) -> str:
    if raw is None:
        return "maybe"
    if isinstance(raw, list | tuple | set):
        items = [str(x).strip() for x in raw if x is not None]
        if not items:
            return "maybe"
        raw_text = "".join(items) if all(len(x) <= 1 for x in items) else " ".join(items)
    else:
        raw_text = str(raw)

    lowered = raw_text.lower().strip()
    for c in ".,;:!?()[]{}'\"":
        lowered = lowered.replace(c, " ")
    lowered = " ".join(lowered.split())
    compact = lowered.replace(" ", "")

    if compact in {"yes", "y", "true", "entailment", "supports", "support"}:
        return "yes"
    if compact in {"no", "n", "false", "contradiction", "refutes", "refute"}:
        return "no"
    return "maybe"


def _truthy_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    normalized = (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )
    return float(normalized.mean())


def _error_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    has_error = series.fillna("").astype(str).str.strip().ne("")
    return float(has_error.mean())


def _repair_rate(df: pd.DataFrame) -> float:
    if "repaired" in df.columns:
        return _truthy_rate(df["repaired"])
    if "repair_attempted" in df.columns:
        return _truthy_rate(df["repair_attempted"])
    return float("nan")


def _score_from_task_metrics(task_scores: dict[str, float]) -> float:
    values: list[float] = []
    for task in TASK_ORDER:
        value = task_scores.get(task, float("nan"))
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _extract_metric(group_df: pd.DataFrame, primary: str, fallback: str | None) -> float:
    primary_value = float(group_df[primary].mean()) if primary in group_df.columns else float("nan")
    if pd.notna(primary_value):
        return primary_value
    if fallback and fallback in group_df.columns:
        return float(group_df[fallback].mean())
    return float("nan")


def _classification_scores(group_df: pd.DataFrame) -> tuple[float, float]:
    y_true = group_df.get("y_true_norm")
    y_pred = group_df.get("y_pred_norm")
    if y_true is None or y_pred is None:
        accuracy = (
            float(group_df["accuracy"].mean()) if "accuracy" in group_df.columns else float("nan")
        )
        macro_f1 = (
            float(group_df["macro_f1"].mean()) if "macro_f1" in group_df.columns else float("nan")
        )
        return accuracy, macro_f1

    cls_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if cls_df.empty:
        accuracy = (
            float(group_df["accuracy"].mean()) if "accuracy" in group_df.columns else float("nan")
        )
        macro_f1 = (
            float(group_df["macro_f1"].mean()) if "macro_f1" in group_df.columns else float("nan")
        )
        return accuracy, macro_f1

    truth = cls_df["y_true"].map(_normalize_label)
    pred = cls_df["y_pred"].map(_normalize_label)
    accuracy = float((truth == pred).mean())
    macro_f1 = float(
        f1_score(
            truth.tolist(),
            pred.tolist(),
            average="macro",
            labels=["yes", "no", "maybe"],
            zero_division=0,
        )
    )
    return accuracy, macro_f1


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _ensure_column(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


def _truncate(value: Any, max_chars: int = 700) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def load_sample_results(run_dir: Path) -> pd.DataFrame:
    db_path = run_dir / "results.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DuckDB file at {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = conn.execute("SELECT * FROM sample_results").df()
    finally:
        conn.close()

    if df.empty:
        return df

    _ensure_column(df, "metrics_json", "{}")
    metrics_df = pd.json_normalize(df["metrics_json"].map(_json_obj))
    for col in metrics_df.columns:
        if col not in df.columns:
            df[col] = metrics_df[col]

    if "macro_f1" not in df.columns and "macro_f1_single" in df.columns:
        df["macro_f1"] = df["macro_f1_single"]
    if "extraction_f1" not in df.columns:
        if "f1" in df.columns:
            df["extraction_f1"] = df["f1"]
        else:
            df["extraction_f1"] = pd.Series([float("nan")] * len(df))
    if "latency_ms" not in df.columns and "latency_sec" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_sec"], errors="coerce") * 1000.0

    _to_numeric(df, NUMERIC_COLS)

    _ensure_column(df, "task", "")
    _ensure_column(df, "model_id", "unknown")
    _ensure_column(df, "prompt_version", "v1")
    _ensure_column(df, "error", "")
    _ensure_column(df, "parse_valid", False)
    _ensure_column(df, "run_name", run_dir.name)
    _ensure_column(df, "example_id", pd.Series(range(len(df))).astype(str))

    if "y_true_norm" not in df.columns:
        df["y_true_norm"] = pd.Series([""] * len(df))
    if "y_pred_norm" not in df.columns:
        df["y_pred_norm"] = pd.Series([""] * len(df))

    cls_mask = df["task"].astype(str).str.lower().eq("classification")
    if cls_mask.any():
        targets = df.loc[cls_mask, "target_text"] if "target_text" in df.columns else ""
        target_norm = pd.Series(targets).map(_normalize_label)
        df.loc[cls_mask, "y_true_norm"] = (
            df.loc[cls_mask, "y_true_norm"]
            .replace("", pd.NA)
            .fillna(target_norm)
            .map(_normalize_label)
        )

        parsed_series = df.loc[cls_mask, "parsed_output"] if "parsed_output" in df.columns else ""
        parsed_labels = (
            pd.Series(parsed_series).map(_json_obj).map(lambda payload: payload.get("label", ""))
        )
        raw_pred = df.loc[cls_mask, "raw_output"] if "raw_output" in df.columns else ""
        fallback_pred = pd.Series(raw_pred).map(_normalize_label)
        pred_series = (
            df.loc[cls_mask, "y_pred_norm"]
            .replace("", pd.NA)
            .fillna(parsed_labels)
            .map(_normalize_label)
            .replace("", pd.NA)
            .fillna(fallback_pred)
            .map(_normalize_label)
        )
        df.loc[cls_mask, "y_pred_norm"] = pred_series

    df["run_name"] = run_dir.name
    return df


def build_model_task_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MODEL_TASK_METRICS_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_cols = ["model_id", "task", "prompt_version"]
    grouped = df.groupby(group_cols, dropna=False, sort=True)

    for (model_id, task, prompt_version), group_df in grouped:
        task_name = str(task)
        parse_valid_rate = (
            _truthy_rate(group_df["parse_valid"])
            if "parse_valid" in group_df.columns
            else float("nan")
        )
        error_rate = _error_rate(group_df["error"]) if "error" in group_df.columns else float("nan")
        repair_rate = _repair_rate(group_df)

        accuracy = (
            float(group_df["accuracy"].mean()) if "accuracy" in group_df.columns else float("nan")
        )
        macro_f1 = (
            float(group_df["macro_f1"].mean()) if "macro_f1" in group_df.columns else float("nan")
        )
        if task_name == "classification":
            accuracy, macro_f1 = _classification_scores(group_df)

        rouge_l = float(group_df["rougeL"].mean()) if "rougeL" in group_df.columns else float("nan")
        bertscore_f1 = (
            float(group_df["bertscore_f1"].mean())
            if "bertscore_f1" in group_df.columns
            else float("nan")
        )
        precision = (
            float(group_df["precision"].mean()) if "precision" in group_df.columns else float("nan")
        )
        recall = float(group_df["recall"].mean()) if "recall" in group_df.columns else float("nan")
        extraction_f1 = (
            float(group_df["extraction_f1"].mean())
            if "extraction_f1" in group_df.columns
            else float("nan")
        )
        exact_match = (
            float(group_df["exact_match"].mean())
            if "exact_match" in group_df.columns
            else float("nan")
        )

        default_metric = float("nan")
        if task_name == "classification":
            default_metric = macro_f1 if pd.notna(macro_f1) else accuracy
        elif task_name == "summarization":
            default_metric = bertscore_f1 if pd.notna(bertscore_f1) else rouge_l
        elif task_name == "extraction":
            default_metric = extraction_f1 if pd.notna(extraction_f1) else exact_match

        rows.append(
            {
                "run_name": str(group_df["run_name"].iloc[0]),
                "model_id": str(model_id),
                "task": task_name,
                "prompt_version": str(prompt_version),
                "default_metric": default_metric,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "rougeL": rouge_l,
                "bertscore_f1": bertscore_f1,
                "precision": precision,
                "recall": recall,
                "extraction_f1": extraction_f1,
                "exact_match": exact_match,
                "parse_valid_rate": parse_valid_rate,
                "error_rate": error_rate,
                "repair_rate": repair_rate,
                "n": int(len(group_df)),
            }
        )

    out = pd.DataFrame(rows, columns=MODEL_TASK_METRICS_COLUMNS)
    out = out.sort_values(["task", "model_id", "prompt_version"], kind="mergesort").reset_index(
        drop=True
    )
    return out


def build_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MODEL_METRICS_COLUMNS)

    rows: list[dict[str, Any]] = []
    grouped = df.groupby("model_id", dropna=False, sort=True)
    for model_id, group_df in grouped:
        task_scores: dict[str, float] = {}

        cls_df = group_df[group_df["task"] == "classification"]
        cls_macro = float("nan")
        cls_acc = float("nan")
        if not cls_df.empty:
            cls_acc, cls_macro = _classification_scores(cls_df)
        task_scores["classification"] = cls_macro if pd.notna(cls_macro) else cls_acc

        sum_df = group_df[group_df["task"] == "summarization"]
        sum_bertscore = (
            _extract_metric(sum_df, "bertscore_f1", "rougeL") if not sum_df.empty else float("nan")
        )
        task_scores["summarization"] = sum_bertscore

        ext_df = group_df[group_df["task"] == "extraction"]
        ext_f1 = (
            _extract_metric(ext_df, "extraction_f1", "exact_match")
            if not ext_df.empty
            else float("nan")
        )
        task_scores["extraction"] = ext_f1

        rows.append(
            {
                "run_name": str(group_df["run_name"].iloc[0]),
                "model_id": str(model_id),
                "score_overall": _score_from_task_metrics(task_scores),
                "classification_macro_f1": cls_macro,
                "summarization_bertscore_f1": sum_bertscore,
                "extraction_f1": ext_f1,
                "parse_valid_rate": (
                    _truthy_rate(group_df["parse_valid"])
                    if "parse_valid" in group_df.columns
                    else float("nan")
                ),
                "error_rate": (
                    _error_rate(group_df["error"]) if "error" in group_df.columns else float("nan")
                ),
                "repair_rate": _repair_rate(group_df),
                "n_total": int(len(group_df)),
                "n_cls": int((group_df["task"] == "classification").sum()),
                "n_sum": int((group_df["task"] == "summarization").sum()),
                "n_ext": int((group_df["task"] == "extraction").sum()),
            }
        )

    out = pd.DataFrame(rows, columns=MODEL_METRICS_COLUMNS)
    out = out.sort_values(["score_overall", "model_id"], ascending=[False, True], kind="mergesort")
    return out.reset_index(drop=True)


def build_run_summary(df: pd.DataFrame, run_name: str, last_update: str) -> pd.DataFrame:
    total_samples = int(len(df))
    n_models = int(df["model_id"].nunique()) if "model_id" in df.columns else 0
    tasks = (
        ",".join(sorted(df["task"].dropna().astype(str).unique())) if "task" in df.columns else ""
    )

    model_metrics = build_model_metrics(df)
    best_model = ""
    best_score = float("nan")
    if not model_metrics.empty:
        top = model_metrics.iloc[0]
        best_model = str(top["model_id"])
        best_score = float(top["score_overall"])

    summary = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "last_update": last_update,
                "total_samples": total_samples,
                "n_models": n_models,
                "tasks": tasks,
                "best_model_overall": best_model,
                "overall_score": best_score,
                "parse_valid_rate": (
                    _truthy_rate(df["parse_valid"]) if "parse_valid" in df.columns else float("nan")
                ),
                "error_rate": _error_rate(df["error"]) if "error" in df.columns else float("nan"),
                "repair_rate": _repair_rate(df),
            }
        ],
        columns=RUN_SUMMARY_COLUMNS,
    )
    return summary


def build_failure_examples(df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FAILURE_EXAMPLES_COLUMNS)

    parse_valid = df["parse_valid"] if "parse_valid" in df.columns else pd.Series([False] * len(df))
    parse_valid_bool = (
        parse_valid.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )
    error_text = (
        df["error"].fillna("").astype(str) if "error" in df.columns else pd.Series([""] * len(df))
    )
    has_error = error_text.str.strip().ne("")

    failures = df[has_error | (~parse_valid_bool)].copy()
    if failures.empty:
        return pd.DataFrame(columns=FAILURE_EXAMPLES_COLUMNS)

    failures["__has_error"] = has_error.loc[failures.index]
    failures["__parse_valid"] = parse_valid_bool.loc[failures.index]
    failures["example_id"] = (
        failures["example_id"].astype(str)
        if "example_id" in failures.columns
        else failures.index.astype(str)
    )

    repaired_col = "repaired" if "repaired" in failures.columns else "repair_attempted"
    if repaired_col in failures.columns:
        repaired_series = failures[repaired_col]
    else:
        repaired_series = pd.Series([pd.NA] * len(failures), index=failures.index)

    out = pd.DataFrame(
        {
            "run_name": failures["run_name"] if "run_name" in failures.columns else "",
            "model_id": failures["model_id"] if "model_id" in failures.columns else "unknown",
            "task": failures["task"] if "task" in failures.columns else "",
            "prompt_version": (
                failures["prompt_version"] if "prompt_version" in failures.columns else "v1"
            ),
            "example_id": failures["example_id"],
            "parse_valid": failures["__parse_valid"],
            "repaired": repaired_series,
            "error": error_text.loc[failures.index],
            "input_trunc": failures["input"].map(_truncate) if "input" in failures.columns else "",
            "output_trunc": (
                failures["raw_output"].map(_truncate) if "raw_output" in failures.columns else ""
            ),
        },
        columns=FAILURE_EXAMPLES_COLUMNS,
    )

    out = out.assign(
        __has_error=failures["__has_error"].values, __parse_valid=failures["__parse_valid"].values
    )
    out = out.sort_values(
        by=["__has_error", "__parse_valid", "model_id", "task", "example_id"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["__has_error", "__parse_valid"])
    return out.head(max_rows).reset_index(drop=True)


def _round_numeric(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        out[col] = out[col].round(decimals)
    return out


def export_run(run_dir: Path, out_dir: Path, max_failures: int = 200) -> dict[str, Path]:
    run_path = Path(run_dir)
    export_dir = Path(out_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    df = load_sample_results(run_path)
    if df.empty:
        raise ValueError(f"No rows found in {run_path / 'results.duckdb'} sample_results table")

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        last_update = str(pd.to_datetime(df["timestamp"], errors="coerce").max())
    else:
        last_update = datetime.now(UTC).isoformat()

    run_name = run_path.name
    run_summary = build_run_summary(df, run_name=run_name, last_update=last_update)
    model_metrics = build_model_metrics(df)
    model_task_metrics = build_model_task_metrics(df)
    failure_examples = build_failure_examples(df, max_rows=max_failures)

    paths = {
        "run_summary": export_dir / "run_summary.csv",
        "model_metrics": export_dir / "model_metrics.csv",
        "model_task_metrics": export_dir / "model_task_metrics.csv",
        "failure_examples": export_dir / "failure_examples.csv",
    }

    _round_numeric(run_summary).to_csv(paths["run_summary"], index=False)
    _round_numeric(model_metrics).to_csv(paths["model_metrics"], index=False)
    _round_numeric(model_task_metrics).to_csv(paths["model_task_metrics"], index=False)
    _round_numeric(failure_examples).to_csv(paths["failure_examples"], index=False)

    return paths
