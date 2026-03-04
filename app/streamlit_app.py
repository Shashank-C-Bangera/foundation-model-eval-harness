from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _show_dataframe(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, use_container_width=True)
    except TypeError:
        st.dataframe(df)


def _candidate_run_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("FMEH_RUNS_ROOT")
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


def _discover_runs() -> dict[str, Path]:
    runs_by_name: dict[str, Path] = {}
    for root in _candidate_run_roots():
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*")):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "results.duckdb").exists():
                continue
            # Prefer first-seen path (roots are ordered by preference).
            runs_by_name.setdefault(run_dir.name, run_dir)
    return runs_by_name


def _default_run_name(runs_by_name: dict[str, Path]) -> str:
    for preferred in ["baseline_models", "rag_baseline", "smoke_ci"]:
        if preferred in runs_by_name:
            return preferred
    return max(runs_by_name, key=lambda k: runs_by_name[k].stat().st_mtime)


def _load_run_df(run_dir: Path) -> pd.DataFrame:
    db_path = run_dir / "results.duckdb"
    conn = duckdb.connect(str(db_path))
    raw_df = conn.execute("SELECT * FROM sample_results").df()
    conn.close()

    metric_rows = []
    for payload in raw_df["metrics_json"].tolist():
        try:
            metric_rows.append(json.loads(payload or "{}"))
        except Exception:
            metric_rows.append({})
    metrics_df = pd.json_normalize(metric_rows).add_prefix("metric_")
    return pd.concat([raw_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)


def main() -> None:
    st.set_page_config(page_title="FMEH Results Browser", layout="wide")
    st.title("Foundation Model Eval Harness")
    st.caption("Browse experiment runs, metrics, and qualitative outputs.")

    runs_by_name = _discover_runs()
    if not runs_by_name:
        searched = ", ".join(str(x) for x in _candidate_run_roots())
        st.warning(f"No run directories with results.duckdb found. Searched: {searched}")
        st.stop()

    run_names = sorted(runs_by_name.keys())
    default_name = _default_run_name(runs_by_name)
    selected = st.selectbox("Run", run_names, index=run_names.index(default_name))
    run_dir = runs_by_name[selected]
    st.caption(f"Using run directory: `{run_dir}`")

    raw_df = _load_run_df(run_dir)
    if raw_df.empty:
        st.warning("No rows in sample_results.")
        st.stop()

    st.subheader("Run Summary")
    model_ids = sorted(raw_df["model_id"].dropna().astype(str).unique().tolist())
    task_ids = sorted(raw_df["task"].dropna().astype(str).unique().tolist())
    st.markdown(f"**Models in run:** `{', '.join(model_ids)}`")
    st.markdown(f"**Tasks in run:** `{', '.join(task_ids)}`")

    report_html = run_dir / "report.html"
    if report_html.exists():
        with st.expander("Report Preview (report.html)", expanded=False):
            html = report_html.read_text(encoding="utf-8", errors="ignore")
            components.html(html, height=700, scrolling=True)

    preds_path = run_dir / "preds.jsonl"
    preds_sample_path = run_dir / "preds.sample.jsonl"
    if preds_path.exists():
        st.caption(f"Qualitative source: `{preds_path.name}`")
    elif preds_sample_path.exists():
        st.caption(f"Qualitative source: `{preds_sample_path.name}`")
    else:
        st.info(
            "No preds JSONL found in this run directory. Qualitative browsing uses DuckDB rows only."
        )

    leaderboard = (
        raw_df.groupby(["task", "model_id", "prompt_version"], dropna=False)
        .agg(
            **{
                "parse_valid_rate": ("parse_valid", "mean"),
                "avg_latency": ("latency_sec", "mean"),
                **(
                    {"avg_accuracy": ("metric_accuracy", "mean")}
                    if "metric_accuracy" in raw_df.columns
                    else {}
                ),
                **(
                    {"avg_macro_f1": ("metric_macro_f1_single", "mean")}
                    if "metric_macro_f1_single" in raw_df.columns
                    else {}
                ),
                **(
                    {"avg_extraction_f1": ("metric_f1", "mean")}
                    if "metric_f1" in raw_df.columns
                    else {}
                ),
                **(
                    {"avg_rougeL": ("metric_rougeL", "mean")}
                    if "metric_rougeL" in raw_df.columns
                    else {}
                ),
                **(
                    {"avg_retrieval_recall_proxy": ("metric_retrieval_recall_proxy", "mean")}
                    if "metric_retrieval_recall_proxy" in raw_df.columns
                    else {}
                ),
                **(
                    {"avg_unsupported_claim_proxy": ("metric_unsupported_claim_proxy", "mean")}
                    if "metric_unsupported_claim_proxy" in raw_df.columns
                    else {}
                ),
            }
        )
        .reset_index()
    )

    st.subheader("Leaderboard")
    _show_dataframe(leaderboard)

    col1, col2, col3 = st.columns(3)
    task_filter = col1.multiselect(
        "Task", sorted(raw_df["task"].dropna().unique().tolist()), default=[]
    )
    model_filter = col2.multiselect(
        "Model", sorted(raw_df["model_id"].dropna().unique().tolist()), default=[]
    )
    prompt_filter = col3.multiselect(
        "Prompt version",
        sorted(raw_df["prompt_version"].dropna().unique().tolist()),
        default=[],
    )

    filtered = raw_df.copy()
    if task_filter:
        filtered = filtered[filtered["task"].isin(task_filter)]
    if model_filter:
        filtered = filtered[filtered["model_id"].isin(model_filter)]
    if prompt_filter:
        filtered = filtered[filtered["prompt_version"].isin(prompt_filter)]

    st.subheader("Sample-level Results")
    show_cols = [
        "task",
        "model_id",
        "prompt_version",
        "example_id",
        "parse_valid",
        "latency_sec",
        "metric_accuracy",
        "metric_f1",
        "metric_rougeL",
    ]
    available_cols = [c for c in show_cols if c in filtered.columns]
    _show_dataframe(filtered[available_cols])

    st.subheader("Qualitative Example")
    row_idx = st.number_input(
        "Row index",
        min_value=0,
        max_value=max(len(filtered) - 1, 0),
        value=0,
    )
    if len(filtered) > 0:
        row = filtered.iloc[int(row_idx)]
        st.markdown(
            f"**Task**: `{row['task']}` | **Model**: `{row['model_id']}` | **Prompt**: `{row['prompt_version']}`"
        )
        st.markdown("**Input**")
        st.code(str(row.get("input", "")), language="text")
        st.markdown("**Raw output**")
        st.code(str(row.get("raw_output", "")), language="json")
        st.markdown("**Parsed output**")
        st.code(str(row.get("parsed_output", "{}")), language="json")
        st.markdown("**Metrics JSON**")
        st.code(str(row.get("metrics_json", "{}")), language="json")


if __name__ == "__main__":
    main()
