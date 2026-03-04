from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import PromptTemplate

from fmeh.data.schemas import ClassificationOutput, ExtractionOutput, SummarizationOutput


def _versions_dir() -> Path:
    return Path(__file__).resolve().parent / "versions"


def load_prompt_version(version: str) -> dict[str, Any]:
    path = _versions_dir() / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version not found: {version}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def schema_for_task(task: str) -> dict[str, Any]:
    if task == "classification":
        return ClassificationOutput.model_json_schema()
    if task == "summarization":
        return SummarizationOutput.model_json_schema()
    if task == "extraction":
        return ExtractionOutput.model_json_schema()
    raise ValueError(f"Unsupported task: {task}")


def render_prompt(task: str, input_text: str, version: str, retrieved_context: str = "") -> str:
    cfg = load_prompt_version(version)
    system = cfg.get("system", "You are a careful biomedical NLP assistant.")
    task_instruction = cfg.get(task, "")

    base = PromptTemplate.from_template(
        "{system}\n\n"
        "Task: {task}\n"
        "Instruction: {instruction}\n"
        "Context: {context}\n"
        "Input:\n{input_text}\n\n"
        "Return ONLY valid JSON matching this schema:\n{schema_json}\n"
    )
    schema_json = json.dumps(schema_for_task(task), indent=2)
    return base.format(
        system=system,
        task=task,
        instruction=task_instruction,
        context=retrieved_context or "none",
        input_text=input_text,
        schema_json=schema_json,
    )
