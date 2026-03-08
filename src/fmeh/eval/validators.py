from __future__ import annotations

import re
from typing import Any

VALID_LABELS = {"yes", "no", "maybe"}


def _flatten_label_input(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list | tuple | set):
        parts = [str(x).strip() for x in raw if x is not None]
        if not parts:
            return ""
        # Handle cases like ['y', 'e', 's'] cleanly.
        if all(len(p) <= 1 for p in parts):
            return "".join(parts)
        return " ".join(parts)
    return str(raw)


def normalize_label(raw: Any) -> str:
    if raw is None:
        return "maybe"

    flattened = _flatten_label_input(raw)
    lowered = flattened.strip().lower()
    if lowered in VALID_LABELS:
        return lowered

    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    compact = lowered.replace(" ", "")

    if compact in {"y", "yes"}:
        return "yes"
    if compact in {"n", "no"}:
        return "no"
    if compact in {"m", "maybe"}:
        return "maybe"

    if lowered in {"true", "entailment", "supports"}:
        return "yes"
    if lowered in {"false", "contradiction", "refutes"}:
        return "no"
    if lowered in {"unknown", "uncertain", "undetermined"}:
        return "maybe"

    if compact in {"true", "entailment", "supports"}:
        return "yes"
    if compact in {"false", "contradiction", "refutes"}:
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
