from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

DEFAULT_METRIC_BY_TASK = {
    "classification": "macro_f1",
    "summarization": "bertscore_f1",
    "extraction": "extraction_f1",
}
TASK_SCORE_COLS = {
    "classification": "classification_macro_f1",
    "summarization": "summarization_bertscore_f1",
    "extraction": "extraction_f1",
}
TASK_N_COLS = {
    "classification": "N_cls",
    "summarization": "N_sum",
    "extraction": "N_ext",
}

RUN_PRIORITY = ["baseline_models", "rag_baseline", "smoke_ci"]
MIN_TASK_N = 50

CANONICAL_METRICS = [
    "accuracy",
    "macro_f1",
    "rougeL",
    "bertscore_f1",
    "extraction_f1",
    "precision",
    "recall",
    "exact_match",
    "pred_length",
    "compression_ratio",
    "latency_ms",
    "latency_sec",
    "retrieval_recall_proxy",
    "unsupported_claim_proxy",
]


def _normalize_label(value: Any) -> str:
    if value is None:
        return "maybe"

    if isinstance(value, list | tuple | set):
        raw = " ".join(str(v) for v in value)
    else:
        raw = str(value)

    cleaned = re.sub(r"[^a-z]+", "", raw.lower())
    if cleaned in {
        "yes",
        "y",
        "true",
        "entailment",
        "supports",
        "support",
        "supported",
    }:
        return "yes"
    if cleaned in {
        "no",
        "n",
        "false",
        "contradiction",
        "refutes",
        "refute",
    }:
        return "no"
    if cleaned in {
        "maybe",
        "unknown",
        "uncertain",
        "unsure",
        "cannotdetermine",
        "notsure",
        "",
    }:
        return "maybe"
    return "maybe"


def _classification_slice_scores(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}

    yt = [_normalize_label(x) for x in y_true]
    yp = [_normalize_label(x) for x in y_pred]
    n = min(len(yt), len(yp))
    if n == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}

    yt = yt[:n]
    yp = yp[:n]
    accuracy = float(sum(t == p for t, p in zip(yt, yp, strict=False)) / n)

    labels = ["yes", "no", "maybe"]
    per_label_f1: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(yt, yp, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(yt, yp, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(yt, yp, strict=False) if t == label and p != label)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label_f1.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(per_label_f1) / len(per_label_f1)),
    }


def _candidate_run_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("FMEH_RUNS_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.extend([Path("space_assets/runs"), Path("runs")])

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


@st.cache_data(show_spinner=False, ttl=10)
def discover_run_dirs() -> dict[str, str]:
    runs_by_name: dict[str, str] = {}
    for root in _candidate_run_roots():
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*")):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "results.duckdb").exists():
                continue
            runs_by_name.setdefault(run_dir.name, str(run_dir))
    return runs_by_name


def default_run_name(runs_by_name: dict[str, str]) -> str:
    for preferred in RUN_PRIORITY:
        if preferred in runs_by_name:
            return preferred
    if not runs_by_name:
        return ""
    return max(runs_by_name, key=lambda k: Path(runs_by_name[k]).stat().st_mtime)


def _metric_payloads(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    payloads = df["metrics_json"] if "metrics_json" in df.columns else pd.Series([], dtype=str)
    for payload in payloads.tolist():
        if isinstance(payload, str):
            try:
                records.append(json.loads(payload or "{}"))
                continue
            except Exception:
                pass
        records.append({})
    return pd.json_normalize(records).add_prefix("metric_")


def _safe_json_load(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload or "{}")
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _add_metric_aliases(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c.startswith("metric_")]:
        bare = col.replace("metric_", "", 1)
        if bare not in df.columns:
            df[bare] = df[col]

    if "extraction_f1" not in df.columns and "f1" in df.columns:
        df["extraction_f1"] = df["f1"]
    if "latency_ms" not in df.columns and "latency_sec" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_sec"], errors="coerce") * 1000.0

    for col in CANONICAL_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _derive_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "y_true_norm" not in df.columns:
        df["y_true_norm"] = ""
    if "y_pred_norm" not in df.columns:
        df["y_pred_norm"] = ""
    if "correct" not in df.columns:
        df["correct"] = False

    cls_mask = df["task"] == "classification"
    if not cls_mask.any():
        return df

    df.loc[cls_mask, "y_true_norm"] = df.loc[cls_mask, "y_true_norm"].replace("", pd.NA)
    df.loc[cls_mask, "y_true_norm"] = (
        df.loc[cls_mask, "y_true_norm"]
        .fillna(df.loc[cls_mask, "target_text"].apply(_normalize_label))
        .apply(_normalize_label)
    )

    parsed_labels = (
        df.loc[cls_mask, "parsed_output"].apply(_safe_json_load).apply(lambda x: x.get("label", ""))
    )
    df.loc[cls_mask, "y_pred_norm"] = (
        df.loc[cls_mask, "y_pred_norm"]
        .replace("", pd.NA)
        .fillna(parsed_labels)
        .apply(_normalize_label)
    )

    df.loc[cls_mask, "correct"] = df.loc[cls_mask, "correct"].astype(bool) | (
        df.loc[cls_mask, "y_true_norm"] == df.loc[cls_mask, "y_pred_norm"]
    )
    return df


@st.cache_data(show_spinner=False)
def load_sample_results(run_dir: str | Path) -> pd.DataFrame:
    run_path = Path(run_dir)
    db_path = run_path / "results.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DuckDB file: {db_path}")

    conn = duckdb.connect(str(db_path))
    raw_df = conn.execute("SELECT * FROM sample_results").df()
    conn.close()
    if raw_df.empty:
        return raw_df

    metric_df = _metric_payloads(raw_df)
    df = pd.concat([raw_df.reset_index(drop=True), metric_df.reset_index(drop=True)], axis=1)
    df = _add_metric_aliases(df)

    if "prompt_version" not in df.columns:
        df["prompt_version"] = "(default)"
    df["prompt_version"] = df["prompt_version"].fillna("(default)").astype(str)

    if "task" not in df.columns:
        df["task"] = "unknown"
    df["task"] = df["task"].fillna("unknown").astype(str)

    if "model_id" not in df.columns:
        df["model_id"] = "unknown"
    df["model_id"] = df["model_id"].fillna("unknown").astype(str)

    if "example_id" in df.columns:
        df["id"] = df["example_id"]
    elif "id" not in df.columns:
        df["id"] = df.index.astype(str)

    if "parse_valid" not in df.columns:
        df["parse_valid"] = False
    df["parse_valid"] = df["parse_valid"].fillna(False).astype(bool)
    df["invalid_output"] = ~df["parse_valid"]

    error_series = df["error"] if "error" in df.columns else pd.Series("", index=df.index)
    df["has_error"] = error_series.fillna("").astype(str).str.strip().ne("")

    if "exception_occurred" not in df.columns:
        df["exception_occurred"] = df["has_error"]
    df["exception_occurred"] = df["exception_occurred"].fillna(False).astype(bool)

    if "repaired" not in df.columns:
        if "repair_attempted" in df.columns:
            df["repaired"] = df["repair_attempted"]
        else:
            df["repaired"] = False
    df["repaired"] = df["repaired"].fillna(False).astype(bool)

    if "empty_output" not in df.columns:
        raw_series = (
            df["raw_output"] if "raw_output" in df.columns else pd.Series("", index=df.index)
        )
        df["empty_output"] = raw_series.fillna("").astype(str).str.strip().eq("")
    df["empty_output"] = df["empty_output"].fillna(False).astype(bool)

    df = _derive_classification_columns(df)
    return df


def _mean_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in CANONICAL_METRICS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _aggregate_base(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    mean_cols = _mean_columns(df)
    agg_spec: dict[str, tuple[str, str]] = {
        "n": ("task", "size"),
        "parse_valid_rate": ("parse_valid", "mean"),
        "invalid_output_rate": ("invalid_output", "mean"),
        "repair_rate": ("repaired", "mean"),
        "exception_rate": ("exception_occurred", "mean"),
        "empty_output_rate": ("empty_output", "mean"),
    }
    for col in mean_cols:
        agg_spec[col] = (col, "mean")
    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def _aggregate_classification_slice(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cls_df = df[df["task"] == "classification"].copy()
    if cls_df.empty:
        return pd.DataFrame(columns=group_cols + ["accuracy", "macro_f1"])

    rows: list[dict[str, Any]] = []
    for keys, part in cls_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys, strict=False))
        y_true = part["y_true_norm"].astype(str).tolist()
        y_pred = part["y_pred_norm"].astype(str).tolist()
        metrics = _classification_slice_scores(y_true, y_pred)
        rows.append(
            {
                **key_map,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def _apply_extraction_minimum(df: pd.DataFrame, min_task_n: int) -> pd.DataFrame:
    out = df.copy()
    out["insufficient_data"] = False
    if "task" not in out.columns:
        return out
    ext_mask = out["task"] == "extraction"
    low_n_mask = ext_mask & (out["n"] < min_task_n)
    if low_n_mask.any():
        out.loc[low_n_mask, "insufficient_data"] = True
        for col in ["extraction_f1", "precision", "recall", "exact_match"]:
            if col in out.columns:
                out.loc[low_n_mask, col] = pd.NA
    return out


def _aggregate(df: pd.DataFrame, group_cols: list[str], min_task_n: int) -> pd.DataFrame:
    base = _aggregate_base(df, group_cols)
    cls = _aggregate_classification_slice(df, group_cols)
    if not cls.empty:
        base = base.merge(cls, on=group_cols, how="left", suffixes=("", "_cls"))
        if "accuracy_cls" in base.columns:
            base["accuracy"] = base["accuracy_cls"].combine_first(base.get("accuracy"))
            base = base.drop(columns=["accuracy_cls"])
        if "macro_f1_cls" in base.columns:
            base["macro_f1"] = base["macro_f1_cls"].combine_first(base.get("macro_f1"))
            base = base.drop(columns=["macro_f1_cls"])
    return _apply_extraction_minimum(base, min_task_n=min_task_n)


@st.cache_data(show_spinner=False)
def agg_metrics(df: pd.DataFrame, min_task_n: int = MIN_TASK_N) -> dict[str, Any]:
    if df.empty:
        empty = pd.DataFrame()
        return {
            "by_prompt": empty,
            "by_task_model": empty,
            "by_model": empty,
            "metric_columns": [],
            "min_task_n": min_task_n,
        }

    by_prompt = _aggregate(df, ["task", "model_id", "prompt_version"], min_task_n=min_task_n)
    by_task_model = _aggregate(df, ["task", "model_id"], min_task_n=min_task_n)
    by_model = _aggregate(df, ["model_id"], min_task_n=min_task_n).rename(columns={"n": "n_total"})

    metric_columns = [c for c in CANONICAL_METRICS if c in by_prompt.columns]
    return {
        "by_prompt": by_prompt,
        "by_task_model": by_task_model,
        "by_model": by_model,
        "metric_columns": metric_columns,
        "min_task_n": min_task_n,
    }


def build_model_leaderboard(agg: dict[str, Any]) -> pd.DataFrame:
    by_model = agg["by_model"].copy()
    by_task_model = agg["by_task_model"].copy()
    min_task_n = int(agg.get("min_task_n", MIN_TASK_N))
    if by_model.empty:
        return by_model

    default_score_cols: list[str] = []
    for task, default_metric in DEFAULT_METRIC_BY_TASK.items():
        score_col = TASK_SCORE_COLS[task]
        n_col = TASK_N_COLS[task]

        task_df = by_task_model[by_task_model["task"] == task].copy()
        if task_df.empty:
            by_model[n_col] = 0
            by_model[score_col] = pd.NA
            continue

        task_n = task_df.set_index("model_id")["n"]
        by_model[n_col] = by_model["model_id"].map(task_n).fillna(0).astype(int)

        if default_metric in task_df.columns:
            metric_series = task_df.set_index("model_id")[default_metric]
            by_model[score_col] = by_model["model_id"].map(metric_series)
            by_model.loc[by_model[n_col] < min_task_n, score_col] = pd.NA
            default_score_cols.append(score_col)
        else:
            by_model[score_col] = pd.NA

    if default_score_cols:
        by_model["overall_score"] = by_model[default_score_cols].mean(axis=1, skipna=True)
    else:
        by_model["overall_score"] = pd.NA

    keep_cols = [
        "model_id",
        "classification_macro_f1",
        "summarization_bertscore_f1",
        "extraction_f1",
        "N_cls",
        "N_sum",
        "N_ext",
        "parse_valid_rate",
        "invalid_output_rate",
        "repair_rate",
        "exception_rate",
        "n_total",
        "overall_score",
    ]
    for col in keep_cols:
        if col not in by_model.columns:
            by_model[col] = pd.NA

    return by_model[keep_cols].sort_values(
        by=["overall_score", "parse_valid_rate"], ascending=[False, False], na_position="last"
    )
