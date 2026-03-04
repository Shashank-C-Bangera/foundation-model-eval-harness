from pathlib import Path

from fmeh.graph.build_graph import build_eval_graph
from fmeh.graph.nodes import NodeContext
from fmeh.logging.duckdb_logger import DuckDBLogger
from fmeh.models.hf_local import MockRunner


def test_graph_end_to_end(tmp_path: Path) -> None:
    db_logger = DuckDBLogger(tmp_path / "results.duckdb")
    ctx = NodeContext(
        runner=MockRunner(),
        prompt_version="v1",
        duckdb_logger=db_logger,
        jsonl_path=tmp_path / "preds.jsonl",
        retriever=None,
    )
    graph = build_eval_graph(ctx)

    final_state = graph.invoke(
        {
            "run_id": "r1",
            "experiment": "smoke",
            "example_id": "ex1",
            "split": "test",
            "task": "classification",
            "model_id": "mock_json",
            "prompt_version": "v1",
            "input": "Question: Is this true? Abstract: Some evidence.",
            "target_text": "maybe",
            "target_json": "",
            "meta_json": "{}",
            "repair_attempted": False,
            "error": "",
        }
    )

    assert final_state["parse_valid"] is True
    df = db_logger.read_all()
    assert len(df) == 1
    db_logger.close()
