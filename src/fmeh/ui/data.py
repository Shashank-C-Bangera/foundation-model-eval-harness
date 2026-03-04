from __future__ import annotations

import json
import os
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

RUN_PRIORITY = ["baseline_models", "rag_baseline", "smoke_ci"]


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


def _add_metric_aliases(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c.startswith("metric_")]:
        bare = col.replace("metric_", "", 1)
        if bare not in df.columns:
            df[bare] = df[col]

    if "macro_f1" not in df.columns and "macro_f1_single" in df.columns:
        df["macro_f1"] = df["macro_f1_single"]
    if "extraction_f1" not in df.columns and "f1" in df.columns:
        df["extraction_f1"] = df["f1"]
    if "latency_ms" not in df.columns and "latency_sec" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_sec"], errors="coerce") * 1000.0

    for col in [
        "macro_f1",
        "accuracy",
        "rougeL",
        "bertscore_f1",
        "extraction_f1",
        "precision",
        "recall",
        "exact_match",
        "pred_length",
        "compression_ratio",
        "latency_sec",
        "latency_ms",
        "retrieval_recall_proxy",
        "unsupported_claim_proxy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

    error_series = df["error"] if "error" in df.columns else pd.Series("", index=df.index)
    df["has_error"] = error_series.fillna("").astype(str).str.strip().ne("")

    parse_series = (
        df["parse_valid"] if "parse_valid" in df.columns else pd.Series(False, index=df.index)
    )
    df["parse_valid"] = parse_series.fillna(False).astype(bool)

    return df


def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "parse_valid",
        "has_error",
        "n",
        "n_total",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _aggregate(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, str]] = {
        "n": ("task", "size"),
        "parse_valid_rate": ("parse_valid", "mean"),
        "error_rate": ("has_error", "mean"),
    }
    for col in metric_cols:
        agg_spec[col] = (col, "mean")

    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


@st.cache_data(show_spinner=False)
def agg_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        empty = pd.DataFrame()
        return {
            "by_prompt": empty,
            "by_task_model": empty,
            "by_model": empty,
            "metric_columns": [],
        }

    metric_cols = _numeric_metric_columns(df)
    by_prompt = _aggregate(df, ["task", "model_id", "prompt_version"], metric_cols)
    by_task_model = _aggregate(df, ["task", "model_id"], metric_cols)
    by_model = _aggregate(df, ["model_id"], metric_cols).rename(columns={"n": "n_total"})

    return {
        "by_prompt": by_prompt,
        "by_task_model": by_task_model,
        "by_model": by_model,
        "metric_columns": metric_cols,
    }
