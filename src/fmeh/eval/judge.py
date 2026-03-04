from __future__ import annotations


def llm_judge_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Optional LLM-as-judge hook, disabled by default."""
    _ = (args, kwargs)
    return {"enabled": False, "score": None}
