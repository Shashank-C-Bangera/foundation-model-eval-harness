from __future__ import annotations

from typing import Any, TypedDict


class EvalState(TypedDict, total=False):
    run_id: str
    experiment: str
    example_id: str
    split: str
    task: str
    model_id: str
    prompt_version: str
    input: str
    target_text: str
    target_json: str
    meta_json: str

    retrieved_context: str
    retrieval_meta: dict[str, Any]

    prompt: str
    raw_output: str
    parsed_output: dict[str, Any] | None
    parse_valid: bool
    parse_error: str
    repair_attempted: bool

    metrics: dict[str, float]
    latency_sec: float
    prompt_tokens: int
    output_tokens: int
    error: str
