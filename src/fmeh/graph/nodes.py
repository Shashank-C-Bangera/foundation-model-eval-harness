from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fmeh.data.schemas import ClassificationOutput, ExtractionOutput, SummarizationOutput
from fmeh.eval.metrics import (
    classification_scores,
    extraction_scores,
    summarize_scores,
    unsupported_claim_proxy,
)
from fmeh.eval.validators import json_validity, non_empty_output, simple_json_repair
from fmeh.graph.state import EvalState
from fmeh.prompts.templates import render_prompt


@dataclass
class NodeContext:
    runner: Any
    prompt_version: str
    duckdb_logger: Any
    jsonl_path: Path
    retriever: Any | None = None
    rag_top_k: int = 3


def _parse_output(task: str, raw_output: str) -> tuple[dict[str, Any] | None, str]:
    if not non_empty_output(raw_output):
        return None, "empty output"

    payload: dict[str, Any] | None = None
    parse_error = ""
    candidates = [raw_output, simple_json_repair(raw_output)]

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            parse_error = str(exc)
            continue

        try:
            if task == "classification":
                payload = ClassificationOutput.model_validate(parsed).model_dump()
            elif task == "summarization":
                payload = SummarizationOutput.model_validate(parsed).model_dump()
            elif task == "extraction":
                payload = ExtractionOutput.model_validate(parsed).model_dump()
            else:
                return None, f"unsupported task: {task}"
            return payload, ""
        except ValidationError as exc:
            parse_error = str(exc)

    return None, parse_error or "unable to parse"


def node_retrieve_context(ctx: NodeContext):
    def _node(state: EvalState) -> dict[str, Any]:
        if ctx.retriever is None:
            return {"retrieved_context": "", "retrieval_meta": {}}

        query = state.get("input", "")
        docs = ctx.retriever.retrieve(query, top_k=ctx.rag_top_k)
        joined = "\n\n".join(d["text"] for d in docs)
        return {"retrieved_context": joined, "retrieval_meta": {"docs": docs}}

    return _node


def node_build_prompt(ctx: NodeContext):
    def _node(state: EvalState) -> dict[str, Any]:
        prompt = render_prompt(
            task=state["task"],
            input_text=state["input"],
            version=state["prompt_version"],
            retrieved_context=state.get("retrieved_context", ""),
        )
        return {"prompt": prompt}

    return _node


def node_run_model(ctx: NodeContext):
    def _node(state: EvalState) -> dict[str, Any]:
        response = ctx.runner.generate(state["prompt"], state["task"])
        return {
            "raw_output": response.text,
            "latency_sec": response.latency_sec,
            "prompt_tokens": response.prompt_tokens,
            "output_tokens": response.output_tokens,
        }

    return _node


def node_parse_output(state: EvalState) -> dict[str, Any]:
    parsed, parse_error = _parse_output(state["task"], state.get("raw_output", ""))
    return {
        "parsed_output": parsed,
        "parse_valid": json_validity(parsed),
        "parse_error": parse_error,
    }


def node_repair_output(state: EvalState) -> dict[str, Any]:
    repaired = simple_json_repair(state.get("raw_output", ""))
    return {
        "raw_output": repaired,
        "repair_attempted": True,
    }


def node_evaluate(state: EvalState) -> dict[str, Any]:
    metrics: dict[str, float] = {"parse_valid": float(bool(state.get("parse_valid")))}
    parsed = state.get("parsed_output") or {}
    docs = state.get("retrieval_meta", {}).get("docs", [])
    if docs:
        query_terms = {t.lower() for t in state.get("input", "").split() if len(t) > 4}
        doc_terms = {t.lower() for d in docs for t in str(d.get("text", "")).split()}
        if query_terms:
            metrics["retrieval_recall_proxy"] = float(
                len(query_terms & doc_terms) / len(query_terms)
            )

    if state["task"] == "classification":
        pred_label = str(parsed.get("label", "maybe"))
        metrics.update(classification_scores(state.get("target_text", "maybe"), pred_label))
    elif state["task"] == "summarization":
        pred_summary = str(parsed.get("summary", ""))
        target = state.get("target_text", "")
        source = state.get("input", "")
        metrics.update(summarize_scores(target=target, pred=pred_summary, source=source))
        context = state.get("retrieved_context", source)
        metrics["unsupported_claim_proxy"] = unsupported_claim_proxy(pred_summary, context)
    elif state["task"] == "extraction":
        target_obj = {}
        if state.get("target_json"):
            try:
                target_obj = json.loads(state["target_json"])
            except Exception:
                target_obj = {}
        metrics.update(extraction_scores(target_obj, parsed))

    return {"metrics": metrics}


def node_log(ctx: NodeContext):
    def _node(state: EvalState) -> dict[str, Any]:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": state.get("run_id", ""),
            "experiment": state.get("experiment", ""),
            "example_id": state.get("example_id", ""),
            "split": state.get("split", ""),
            "task": state.get("task", ""),
            "model_id": state.get("model_id", ""),
            "prompt_version": state.get("prompt_version", ""),
            "input": state.get("input", ""),
            "target_text": state.get("target_text", ""),
            "target_json": state.get("target_json", ""),
            "meta_json": state.get("meta_json", "{}"),
            "retrieved_context": state.get("retrieved_context", ""),
            "raw_output": state.get("raw_output", ""),
            "parsed_output": json.dumps(state.get("parsed_output") or {}, ensure_ascii=False),
            "parse_valid": bool(state.get("parse_valid", False)),
            "parse_error": state.get("parse_error", ""),
            "metrics_json": json.dumps(state.get("metrics") or {}, ensure_ascii=False),
            "latency_sec": float(state.get("latency_sec", 0.0)),
            "prompt_tokens": int(state.get("prompt_tokens", 0)),
            "output_tokens": int(state.get("output_tokens", 0)),
            "error": state.get("error", ""),
        }

        ctx.duckdb_logger.log_sample(row)
        ctx.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with ctx.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {}

    return _node


def route_after_parse(state: EvalState) -> str:
    if state.get("parse_valid", False):
        return "evaluate"
    if state.get("repair_attempted", False):
        return "evaluate"
    return "repair_output"
