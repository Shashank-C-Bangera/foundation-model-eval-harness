from __future__ import annotations

import json
import re
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
from fmeh.eval.validators import (
    json_validity,
    non_empty_output,
    normalize_label,
    simple_json_repair,
)
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


def _classification_fallback(raw_output: str) -> dict[str, Any] | None:
    match = re.search(
        r"\b(yes|no|maybe|true|false|entailment|contradiction|supports|refutes)\b",
        raw_output.lower(),
    )
    if match is None:
        return None
    payload = {
        "label": normalize_label(match.group(1)),
        "rationale": raw_output.strip()[:500],
    }
    try:
        return ClassificationOutput.model_validate(payload).model_dump()
    except ValidationError:
        return None


def _summarization_fallback(raw_output: str) -> dict[str, Any] | None:
    summary = raw_output.strip().strip("`").strip()
    if not summary:
        return None
    payload = {"summary": summary[:4000]}
    try:
        return SummarizationOutput.model_validate(payload).model_dump()
    except ValidationError:
        return None


def _split_mentions(text: str) -> list[str]:
    cleaned = text.strip().strip("[]")
    if not cleaned:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,\n;|]", cleaned):
        term = part.strip().strip('"').strip("'").strip("`")
        term = re.sub(r"\s+", " ", term).strip(" .:-")
        lowered = term.lower()
        if not term or lowered in {"none", "null", "n/a", "na"}:
            continue
        if any(
            noise in lowered
            for noise in (
                "return exactly one json object",
                "required json shape",
                "do not return markdown",
                "do not echo the prompt",
                '"string"',
                "output rules",
            )
        ):
            continue
        words = re.findall(r"[a-z0-9-]+", lowered)
        if not words:
            continue
        # Drop obvious degeneration like "factors factors factors ...".
        if len(words) >= 4 and len(set(words)) / len(words) < 0.35:
            continue
        if len(words) > 8:
            continue
        canonical = " ".join(words)
        if canonical not in seen:
            seen.add(canonical)
            out.append(" ".join(term.split()))
    return out


def _extraction_fallback(raw_output: str) -> dict[str, Any] | None:
    disease_match = re.search(
        r"diseases?\s*[:=-]\s*(.+?)(?:\n|$|chemicals?\s*[:=-])",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    chemical_match = re.search(
        r"chemicals?\s*[:=-]\s*(.+?)(?:\n|$|diseases?\s*[:=-])",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    diseases = _split_mentions(disease_match.group(1)) if disease_match else []
    chemicals = _split_mentions(chemical_match.group(1)) if chemical_match else []
    if not diseases and not chemicals:
        # Handle compact outputs like "epilepticus" or "aspirin, fever".
        fallback_mentions = _split_mentions(raw_output)
        if fallback_mentions:
            diseases = fallback_mentions
    if not diseases and not chemicals:
        return None
    payload = {"diseases": diseases, "chemicals": chemicals}
    try:
        return ExtractionOutput.model_validate(payload).model_dump()
    except ValidationError:
        return None


def _parse_output(task: str, raw_output: str) -> tuple[dict[str, Any] | None, str, bool]:
    if not non_empty_output(raw_output):
        return None, "empty output", True

    payload: dict[str, Any] | None = None
    parse_error = ""
    empty_output = False
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
                return None, f"unsupported task: {task}", empty_output
            return payload, "", empty_output
        except ValidationError as exc:
            parse_error = str(exc)

    fallback: dict[str, Any] | None = None
    if task == "classification":
        fallback = _classification_fallback(raw_output)
    elif task == "summarization":
        fallback = _summarization_fallback(raw_output)
    elif task == "extraction":
        fallback = _extraction_fallback(raw_output)
    if fallback is not None:
        return fallback, "", empty_output

    return None, parse_error or "unable to parse", empty_output


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
    parsed, parse_error, empty_output = _parse_output(state["task"], state.get("raw_output", ""))
    return {
        "parsed_output": parsed,
        "parse_valid": json_validity(parsed),
        "parse_error": parse_error,
        "empty_output": empty_output,
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
    y_true_norm = ""
    y_pred_norm = ""
    correct = False
    if docs:
        query_terms = {t.lower() for t in state.get("input", "").split() if len(t) > 4}
        doc_terms = {t.lower() for d in docs for t in str(d.get("text", "")).split()}
        if query_terms:
            metrics["retrieval_recall_proxy"] = float(
                len(query_terms & doc_terms) / len(query_terms)
            )

    if state["task"] == "classification":
        y_true_norm = normalize_label(state.get("target_text", "maybe"))
        y_pred_norm = normalize_label(parsed.get("label", "maybe"))
        correct = y_true_norm == y_pred_norm
        metrics.update(classification_scores(y_true_norm, y_pred_norm))
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

    return {
        "metrics": metrics,
        "y_true_norm": y_true_norm,
        "y_pred_norm": y_pred_norm,
        "correct": correct,
    }


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
            "empty_output": bool(state.get("empty_output", False)),
            "repaired": bool(state.get("repair_attempted", False)),
            "exception_occurred": bool(str(state.get("error", "")).strip()),
            "y_true_norm": state.get("y_true_norm", ""),
            "y_pred_norm": state.get("y_pred_norm", ""),
            "correct": bool(state.get("correct", False)),
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
