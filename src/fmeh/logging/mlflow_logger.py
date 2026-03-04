from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from fmeh.config import HarnessConfig


class MLflowRunLogger:
    def __init__(self, config: HarnessConfig, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        tracking_uri = config.mlflow.tracking_uri.strip() or str((run_dir / "mlruns").resolve())
        if not tracking_uri.startswith("file:"):
            tracking_uri = f"file:{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

        self._run = mlflow.start_run(run_name=f"{config.name}")
        self._config = config
        self._run_dir = run_dir
        mlflow.log_params(
            {
                "experiment": config.name,
                "seed": config.seed,
                "models": ",".join(config.models),
                "prompt_versions": ",".join(config.prompt_versions),
                "tasks": ",".join(config.tasks),
                "rag_enabled": str(config.rag.enabled),
            }
        )

    def log_aggregates(self, results_df: pd.DataFrame) -> None:
        if results_df.empty:
            return

        # task/model/prompt aggregates
        agg = (
            results_df.groupby(["task", "model_id", "prompt_version"])  # type: ignore[arg-type]
            .agg(parse_valid_rate=("parse_valid", "mean"), avg_latency=("latency_sec", "mean"))
            .reset_index()
        )

        for row in agg.to_dict(orient="records"):
            prefix = f"{row['task']}.{row['model_id']}.{row['prompt_version']}"
            mlflow.log_metric(f"{prefix}.parse_valid_rate", float(row["parse_valid_rate"]))
            mlflow.log_metric(f"{prefix}.avg_latency", float(row["avg_latency"]))

        artifacts = [
            self._run_dir / "preds.jsonl",
            self._run_dir / "report.html",
            self._run_dir / "report.md",
            self._run_dir / "config_resolved.yaml",
            self._run_dir / "results.duckdb",
        ]
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

    def finish(self) -> None:
        if self._run is not None:
            mlflow.end_run()
            self._run = None
