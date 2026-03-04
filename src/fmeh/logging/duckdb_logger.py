from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class DuckDBLogger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_results (
                timestamp TEXT,
                run_id TEXT,
                experiment TEXT,
                example_id TEXT,
                split TEXT,
                task TEXT,
                model_id TEXT,
                prompt_version TEXT,
                input TEXT,
                target_text TEXT,
                target_json TEXT,
                meta_json TEXT,
                retrieved_context TEXT,
                raw_output TEXT,
                parsed_output TEXT,
                parse_valid BOOLEAN,
                parse_error TEXT,
                metrics_json TEXT,
                latency_sec DOUBLE,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                error TEXT
            )
            """
        )

    def log_sample(self, row: dict) -> None:
        self.conn.execute(
            """
            INSERT INTO sample_results VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                row.get("timestamp", ""),
                row.get("run_id", ""),
                row.get("experiment", ""),
                row.get("example_id", ""),
                row.get("split", ""),
                row.get("task", ""),
                row.get("model_id", ""),
                row.get("prompt_version", ""),
                row.get("input", ""),
                row.get("target_text", ""),
                row.get("target_json", ""),
                row.get("meta_json", "{}"),
                row.get("retrieved_context", ""),
                row.get("raw_output", ""),
                row.get("parsed_output", "{}"),
                bool(row.get("parse_valid", False)),
                row.get("parse_error", ""),
                row.get("metrics_json", "{}"),
                float(row.get("latency_sec", 0.0)),
                int(row.get("prompt_tokens", 0)),
                int(row.get("output_tokens", 0)),
                row.get("error", ""),
            ],
        )

    def read_all(self) -> pd.DataFrame:
        return self.conn.execute("SELECT * FROM sample_results").df()

    def close(self) -> None:
        self.conn.close()
