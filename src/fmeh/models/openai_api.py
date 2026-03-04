"""Optional provider stub for future API-based model runs."""

from __future__ import annotations


class OpenAIAPIRunner:
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("OpenAI API runner is optional and not enabled by default.")
