from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractionOutput(BaseModel):
    diseases: list[str] = Field(default_factory=list)
    chemicals: list[str] = Field(default_factory=list)


class ClassificationOutput(BaseModel):
    label: Literal["yes", "no", "maybe"]
    rationale: str = ""


class SummarizationOutput(BaseModel):
    summary: str
