from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from fmeh.ui.data import (
    DEFAULT_METRIC_BY_TASK,
    agg_metrics,
    build_model_leaderboard,
    default_run_name,
    discover_run_dirs,
    load_sample_results,
)

PAGE_OPTIONS = ["Overview", "Compare", "Inspect"]
NON_INSTRUCT_MODELS = {"t5_small", "tiny_gpt2"}

COMPARE_METRIC_ORDER = [
    "macro_f1",
    "accuracy",
    "bertscore_f1",
    "rougeL",
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

METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "accuracy": "Accuracy",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "extraction_f1": "Extraction F1",
    "precision": "Precision",
    "recall": "Recall",
    "exact_match": "Exact match",
    "pred_length": "Output length",
    "compression_ratio": "Compression ratio",
    "latency_ms": "Latency (ms)",
    "latency_sec": "Latency (s)",
    "retrieval_recall_proxy": "Retrieval recall proxy",
    "unsupported_claim_proxy": "Unsupported-claim proxy",
}


def _show_dataframe(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True)


def _fmt_rate(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_metric(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.3f}"


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _model_label(model_id: str) -> str:
    if model_id in NON_INSTRUCT_MODELS:
        return f"{model_id} (non-instruct)"
    return model_id


def _metric_options_for_task(task: str, frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    task_frame = frame[frame["task"] == task]
    if task_frame.empty:
        return []

    options: list[str] = []
    for metric in COMPARE_METRIC_ORDER:
        if metric not in task_frame.columns:
            continue
        if task_frame[metric].notna().any():
            options.append(metric)
    return options


def _render_overview(run_name: str, df: pd.DataFrame, agg: dict[str, pd.DataFrame]) -> None:
    st.header("Overview")

    timestamps = pd.to_datetime(df.get("timestamp"), errors="coerce")
    latest_ts = "-"
    if not timestamps.empty and timestamps.notna().any():
        latest_ts = timestamps.max().strftime("%Y-%m-%d %H:%M")

    tasks = sorted(df["task"].dropna().astype(str).unique().tolist())
    models = sorted(df["model_id"].dropna().astype(str).unique().tolist())
    st.caption(
        f"Run: `{run_name}` | Last update: `{latest_ts}` | Samples: `{len(df)}` | "
        f"Models: `{len(models)}` | Tasks: `{', '.join(tasks)}`"
    )

    leaderboard = build_model_leaderboard(agg)
    best_model = "-"
    best_score = pd.NA
    if not leaderboard.empty:
        top = leaderboard.iloc[0]
        best_model = _model_label(str(top["model_id"]))
        best_score = top.get("overall_score", pd.NA)

    valid_rate = float(df["parse_valid"].mean()) if not df.empty else float("nan")
    invalid_rate = float(df["invalid_output"].mean()) if not df.empty else float("nan")
    repair_rate = float(df["repaired"].mean()) if not df.empty else float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Best model",
        best_model,
        delta=f"Score {_fmt_metric(best_score)}" if not pd.isna(best_score) else None,
    )
    col2.metric("Valid output %", _fmt_rate(valid_rate))
    col3.metric("Invalid output %", _fmt_rate(invalid_rate))
    col4.metric("Repair rate %", _fmt_rate(repair_rate))

    chart_source = leaderboard[["model_id", "overall_score"]].dropna(subset=["overall_score"])
    if not chart_source.empty:
        chart_source = chart_source.copy()
        chart_source["model_id"] = chart_source["model_id"].map(_model_label)
        st.bar_chart(chart_source.set_index("model_id")["overall_score"])

    display = leaderboard[
        [
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
        ]
    ].rename(
        columns={
            "model_id": "Model",
            "classification_macro_f1": "Classification Macro F1",
            "summarization_bertscore_f1": "Summarization BERTScore F1",
            "extraction_f1": "Extraction F1",
            "parse_valid_rate": "Valid output %",
            "invalid_output_rate": "Invalid output %",
            "repair_rate": "Repair rate %",
            "exception_rate": "Exception rate %",
            "n_total": "N_total",
        }
    )
    display["Model"] = display["Model"].astype(str).map(_model_label)
    _show_dataframe(display)


def _render_compare(agg: dict[str, pd.DataFrame]) -> None:
    st.header("Compare")

    by_prompt = agg["by_prompt"]
    by_task_model = agg["by_task_model"]
    if by_prompt.empty:
        st.info("No aggregated data available for comparison.")
        return

    task_options = sorted(by_prompt["task"].dropna().astype(str).unique().tolist())
    default_task = "classification" if "classification" in task_options else task_options[0]
    task = st.selectbox("Task", task_options, index=task_options.index(default_task))

    metric_options = _metric_options_for_task(task, by_prompt)
    if not metric_options:
        st.info("No numeric metrics found for this task.")
        return

    default_metric = DEFAULT_METRIC_BY_TASK.get(task, metric_options[0])
    metric_idx = metric_options.index(default_metric) if default_metric in metric_options else 0
    metric = st.selectbox("Metric", metric_options, index=metric_idx, format_func=_metric_label)
    show_prompt_versions = st.checkbox("Show prompt versions", value=False)

    source = by_prompt if show_prompt_versions else by_task_model
    view = source[source["task"] == task].copy()
    if view.empty:
        st.info("No rows for selected task.")
        return

    if task == "extraction" and view["n"].max() < int(agg.get("min_task_n", 50)):
        st.warning(
            "Extraction has insufficient data (< min task N). Metric is excluded from scoring."
        )

    if show_prompt_versions:
        view["label"] = view["model_id"].map(_model_label) + " | " + view["prompt_version"]
    else:
        view["label"] = view["model_id"].map(_model_label)

    view = view.sort_values(by=metric, ascending=False, na_position="last")
    st.bar_chart(view.set_index("label")[metric])

    table_cols = ["model_id"]
    if show_prompt_versions:
        table_cols.append("prompt_version")
    table_cols.extend(
        [
            "n",
            metric,
            "parse_valid_rate",
            "invalid_output_rate",
            "repair_rate",
            "exception_rate",
        ]
    )
    table = view[table_cols].rename(
        columns={
            "model_id": "Model",
            "prompt_version": "Prompt version",
            "n": "N",
            metric: _metric_label(metric),
            "parse_valid_rate": "Valid output %",
            "invalid_output_rate": "Invalid output %",
            "repair_rate": "Repair rate %",
            "exception_rate": "Exception rate %",
        }
    )
    table["Model"] = table["Model"].astype(str).map(_model_label)
    _show_dataframe(table)


def _target_value(row: pd.Series) -> str:
    target_text = str(row.get("target_text", "") or "")
    target_json = str(row.get("target_json", "") or "")
    if target_text.strip():
        return target_text
    if target_json.strip():
        return target_json
    return "-"


def _render_inspect(df: pd.DataFrame) -> None:
    st.header("Inspect")

    task_options = sorted(df["task"].dropna().astype(str).unique().tolist())
    default_task = "classification" if "classification" in task_options else task_options[0]
    task = st.selectbox("Task", task_options, index=task_options.index(default_task))

    task_df = df[df["task"] == task].copy()
    model_options = sorted(task_df["model_id"].dropna().astype(str).unique().tolist())
    model_choice = st.selectbox("Model", ["All"] + model_options, index=0)
    only_failures = st.checkbox("Only failures", value=True)
    parse_valid_only = st.checkbox("Parse valid only", value=False)

    if model_choice != "All":
        task_df = task_df[task_df["model_id"] == model_choice]
    if only_failures:
        task_df = task_df[
            (~task_df["parse_valid"])
            | task_df["exception_occurred"]
            | (task_df["y_pred_norm"] == "")
        ]
    if parse_valid_only:
        task_df = task_df[task_df["parse_valid"]]

    task_df = task_df.reset_index(drop=True)
    if task_df.empty:
        st.info("No rows match current filters.")
        return

    page_size = 10
    max_page = max(1, math.ceil(len(task_df) / page_size))
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)

    start = (int(page) - 1) * page_size
    end = start + page_size
    page_df = task_df.iloc[start:end].copy()

    table_cols = [
        "id",
        "task",
        "model_id",
        "prompt_version",
        "parse_valid",
        "repaired",
        "exception_occurred",
    ]
    if task == "classification":
        table_cols.extend(["y_true_norm", "y_pred_norm", "correct"])
    table = page_df[table_cols].rename(
        columns={
            "id": "ID",
            "task": "Task",
            "model_id": "Model",
            "prompt_version": "Prompt version",
            "parse_valid": "Valid output",
            "repaired": "Repaired",
            "exception_occurred": "Exception",
            "y_true_norm": "y_true_norm",
            "y_pred_norm": "y_pred_norm",
            "correct": "correct",
        }
    )
    table["Model"] = table["Model"].astype(str).map(_model_label)
    _show_dataframe(table)

    for _, row in page_df.iterrows():
        row_label = f"{row.get('id', '-')} | valid={bool(row.get('parse_valid', False))}"
        with st.expander(row_label, expanded=False):
            st.markdown("**Input**")
            st.code(str(row.get("input", "")), language="text")
            st.markdown("**Target**")
            st.code(_target_value(row), language="text")
            st.markdown("**Raw output**")
            st.code(str(row.get("raw_output", "")), language="text")
            st.markdown("**Parsed output**")
            st.code(str(row.get("parsed_output", "")), language="text")
            st.markdown("**Error**")
            st.code(str(row.get("error", "")), language="text")


def main() -> None:
    st.set_page_config(page_title="FMEH Dashboard", layout="wide")

    runs_by_name = discover_run_dirs()
    if not runs_by_name:
        st.error("No runs found. Expected results.duckdb under ./runs/* or ./space_assets/runs/*.")
        st.stop()

    st.sidebar.title("Navigation")
    run_names = sorted(runs_by_name.keys())
    selected_default = default_run_name(runs_by_name)
    run_index = run_names.index(selected_default) if selected_default in run_names else 0
    selected_run = st.sidebar.selectbox("Run", run_names, index=run_index)
    page = st.sidebar.radio("Page", PAGE_OPTIONS, index=0)

    run_dir = runs_by_name[selected_run]
    try:
        df = load_sample_results(run_dir)
    except FileNotFoundError:
        st.error(f"Missing DuckDB file for `{selected_run}`: `{run_dir}/results.duckdb`")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load run `{selected_run}`: {exc}")
        st.stop()

    if df.empty:
        st.warning("sample_results is empty for the selected run.")
        st.stop()

    agg = agg_metrics(df)

    if page == "Overview":
        _render_overview(selected_run, df, agg)
    elif page == "Compare":
        _render_compare(agg)
    else:
        _render_inspect(df)


if __name__ == "__main__":
    main()
