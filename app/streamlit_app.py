from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FMEH Results Browser", layout="wide")

runs_root = Path(os.getenv("FMEH_RUNS_ROOT", "runs"))
run_dirs = sorted([p for p in runs_root.glob("*") if p.is_dir()])

st.title("Foundation Model Eval Harness")
st.caption("Browse experiment runs, metrics, and qualitative outputs.")

if not run_dirs:
    st.warning(f"No run directories found under {runs_root}")
    st.stop()

selected = st.selectbox("Run", [p.name for p in run_dirs], index=len(run_dirs) - 1)
run_dir = runs_root / selected

db_path = run_dir / "results.duckdb"
if not db_path.exists():
    st.error(f"Missing results DB: {db_path}")
    st.stop()

conn = duckdb.connect(str(db_path))
raw_df = conn.execute("SELECT * FROM sample_results").df()
conn.close()

if raw_df.empty:
    st.warning("No rows in sample_results.")
    st.stop()

metric_rows = []
for payload in raw_df["metrics_json"].tolist():
    try:
        metric_rows.append(json.loads(payload or "{}"))
    except Exception:
        metric_rows.append({})
metrics_df = pd.json_normalize(metric_rows).add_prefix("metric_")
df = pd.concat([raw_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

leaderboard = (
    df.groupby(["task", "model_id", "prompt_version"], dropna=False)
    .agg(
        **{
            "parse_valid_rate": ("parse_valid", "mean"),
            "avg_latency": ("latency_sec", "mean"),
            **(
                {"avg_accuracy": ("metric_accuracy", "mean")}
                if "metric_accuracy" in df.columns
                else {}
            ),
            **(
                {"avg_macro_f1": ("metric_macro_f1_single", "mean")}
                if "metric_macro_f1_single" in df.columns
                else {}
            ),
            **({"avg_extraction_f1": ("metric_f1", "mean")} if "metric_f1" in df.columns else {}),
            **({"avg_rougeL": ("metric_rougeL", "mean")} if "metric_rougeL" in df.columns else {}),
        }
    )
    .reset_index()
)

st.subheader("Leaderboard")
st.dataframe(leaderboard, width="stretch")

col1, col2, col3 = st.columns(3)
task_filter = col1.multiselect("Task", sorted(df["task"].dropna().unique().tolist()), default=[])
model_filter = col2.multiselect(
    "Model", sorted(df["model_id"].dropna().unique().tolist()), default=[]
)
prompt_filter = col3.multiselect(
    "Prompt version",
    sorted(df["prompt_version"].dropna().unique().tolist()),
    default=[],
)

filtered = df.copy()
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
st.dataframe(filtered[available_cols], width="stretch")

st.subheader("Qualitative Example")
row_idx = st.number_input("Row index", min_value=0, max_value=max(len(filtered) - 1, 0), value=0)
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
