from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from fmeh.config import HarnessConfig, load_experiment_config, save_resolved_config
from fmeh.data.build_datasets import build_datasets, sample_for_run
from fmeh.graph.build_graph import build_eval_graph
from fmeh.graph.nodes import NodeContext
from fmeh.logging.duckdb_logger import DuckDBLogger
from fmeh.logging.mlflow_logger import MLflowRunLogger
from fmeh.models.registry import build_runner
from fmeh.rag.index import build_index
from fmeh.rag.retriever import FaissRetriever
from fmeh.reporting.make_report import create_report

app = typer.Typer(help="Foundation Model Eval Harness CLI")
data_app = typer.Typer(help="Dataset commands")
app.add_typer(data_app, name="data")

console = Console()


def _load_or_build_data(cfg: HarnessConfig, force_rebuild: bool = False) -> pd.DataFrame:
    parquet_path = Path(cfg.paths.data_parquet)
    if force_rebuild or not parquet_path.exists():
        return build_datasets(cfg)
    return pd.read_parquet(parquet_path)


@data_app.command("build")
def data_build(
    experiment: str = typer.Option(
        "baseline_models", help="Experiment config name under configs/experiments"
    ),
) -> None:
    cfg = load_experiment_config(experiment)
    df = build_datasets(cfg)
    console.print(f"Saved dataset to {cfg.paths.data_parquet} and {cfg.paths.data_jsonl}")
    console.print(df.groupby("task").size())


@app.command("run")
def run_experiment(
    experiment: str = typer.Option("baseline_models", help="Experiment config name"),
    force_rebuild_data: bool = typer.Option(False, help="Rebuild data before running"),
) -> None:
    cfg = load_experiment_config(experiment)
    run_dir = Path(cfg.paths.runs_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep each named experiment run deterministic and clean.
    for path in [run_dir / "preds.jsonl", run_dir / "results.duckdb"]:
        if path.exists():
            path.unlink()
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)

    data_df = _load_or_build_data(cfg, force_rebuild=force_rebuild_data)
    sampled_df = sample_for_run(data_df, cfg)

    run_id = str(uuid.uuid4())
    save_resolved_config(cfg, run_dir / "config_resolved.yaml")

    db_logger = DuckDBLogger(run_dir / "results.duckdb")
    mlflow_logger = MLflowRunLogger(cfg, run_dir)

    retriever = None
    if cfg.rag.enabled:
        rag_source = sampled_df[["id", "input"]].rename(columns={"id": "example_id"})
        if not Path(cfg.rag.index_path).exists() or not Path(cfg.rag.passages_path).exists():
            console.print("RAG enabled. Building index...")
            build_index(cfg, rag_source)
        retriever = FaissRetriever(
            index_path=cfg.rag.index_path,
            passages_path=cfg.rag.passages_path,
            embedding_model=cfg.rag.embedding_model,
        )

    rows = sampled_df.to_dict(orient="records")
    total_jobs = len(rows) * len(cfg.models) * len(cfg.prompt_versions)
    console.print(
        f"Running {total_jobs} jobs: {len(rows)} examples × {len(cfg.models)} models × {len(cfg.prompt_versions)} prompt versions"
    )

    completed = 0
    for model_id in cfg.models:
        runner = build_runner(
            model_id=model_id,
            device=cfg.device,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            max_new_tokens=cfg.generation.max_new_tokens,
        )
        try:
            for prompt_version in cfg.prompt_versions:
                ctx = NodeContext(
                    runner=runner,
                    prompt_version=prompt_version,
                    duckdb_logger=db_logger,
                    jsonl_path=run_dir / "preds.jsonl",
                    retriever=retriever,
                    rag_top_k=cfg.rag.top_k,
                )
                graph = build_eval_graph(ctx)

                for row in track(rows, description=f"Model={model_id} Prompt={prompt_version}"):
                    initial_state = {
                        "run_id": run_id,
                        "experiment": cfg.name,
                        "example_id": row["id"],
                        "split": row.get("split", ""),
                        "task": row["task"],
                        "model_id": model_id,
                        "prompt_version": prompt_version,
                        "input": row.get("input", ""),
                        "target_text": row.get("target_text", ""),
                        "target_json": row.get("target_json", ""),
                        "meta_json": row.get("meta_json", "{}"),
                        "repair_attempted": False,
                        "error": "",
                    }
                    try:
                        graph.invoke(initial_state)
                    except Exception as exc:
                        error_row = {
                            "timestamp": pd.Timestamp.utcnow().isoformat(),
                            "run_id": run_id,
                            "experiment": cfg.name,
                            "example_id": row["id"],
                            "split": row.get("split", ""),
                            "task": row["task"],
                            "model_id": model_id,
                            "prompt_version": prompt_version,
                            "input": row.get("input", ""),
                            "target_text": row.get("target_text", ""),
                            "target_json": row.get("target_json", ""),
                            "meta_json": row.get("meta_json", "{}"),
                            "retrieved_context": "",
                            "raw_output": "",
                            "parsed_output": "{}",
                            "parse_valid": False,
                            "parse_error": "",
                            "metrics_json": "{}",
                            "latency_sec": 0.0,
                            "prompt_tokens": 0,
                            "output_tokens": 0,
                            "error": str(exc),
                        }
                        db_logger.log_sample(error_row)
                        with (run_dir / "preds.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                    completed += 1
                    if completed % 50 == 0 or completed == total_jobs:
                        console.print(f"Completed {completed}/{total_jobs}")
        finally:
            if hasattr(runner, "close"):
                runner.close()
            del runner
            gc.collect()

    results_df = db_logger.read_all()
    db_logger.close()

    mlflow_logger.log_aggregates(results_df)
    mlflow_logger.finish()

    console.print(f"Run complete. Artifacts in: {run_dir}")


@app.command("report")
def report(
    run_dir: str = typer.Option(..., help="Run directory, e.g. runs/baseline_models"),
) -> None:
    outputs = create_report(run_dir)
    console.print(f"Wrote report HTML: {outputs['html']}")
    console.print(f"Wrote report MD: {outputs['md']}")


@app.command("serve")
def serve(
    run_dir: str = typer.Option("runs", help="Root run directory to browse"),
    port: int = typer.Option(8501, help="Streamlit port"),
) -> None:
    app_path = Path("app") / "streamlit_app.py"
    env = dict(os.environ)
    env["FMEH_RUNS_ROOT"] = run_dir
    cmd = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    app()
