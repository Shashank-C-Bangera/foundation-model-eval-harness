from __future__ import annotations

import re
from typing import Any

VALID_LABELS = {"yes", "no", "maybe"}


def normalize_label(raw: str | None) -> str:
    if raw is None:
        return "maybe"
    lowered = raw.strip().lower()
    if lowered in VALID_LABELS:
        return lowered
    if lowered in {"true", "entailment", "supports"}:
        return "yes"
    if lowered in {"false", "contradiction", "refutes"}:
        return "no"
    return "maybe"


def simple_json_repair(text: str) -> str:
    candidate = text.strip()
    first = candidate.find("{")
    last = candidate.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidate = candidate[first : last + 1]

    candidate = candidate.replace("\n", " ").replace("\t", " ")
    candidate = re.sub(r"'", '"', candidate)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return candidate


def json_validity(parsed_output: Any) -> bool:
    return parsed_output is not None


def non_empty_output(text: str) -> bool:
    return bool(text and text.strip())
