from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import get_dataset_split_names, load_dataset
from rich.console import Console

from fmeh.config import HarnessConfig
from fmeh.eval.validators import normalize_label

console = Console()


@dataclass
class UnifiedExample:
    id: str
    task: str
    input: str
    target_text: str
    target_json: str
    source: str
    split: str
    meta_json: str


def _stable_split(example_id: str, seed: int) -> str:
    digest = hashlib.md5(f"{example_id}:{seed}".encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest[:2], 16) / 255
    if bucket < 0.8:
        return "train"
    if bucket < 0.9:
        return "val"
    return "test"


def _normalize_split(split_name: str, example_id: str, seed: int) -> str:
    lowered = split_name.lower()
    if lowered == "validation":
        return "val"
    if lowered in {"train", "val", "test"}:
        return lowered
    return _stable_split(example_id, seed)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for v in value:
            if isinstance(v, list):
                chunks.extend(str(x) for x in v)
            else:
                chunks.append(str(v))
        return " ".join(chunks)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return str(value)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _pubmedqa_examples(cfg: HarnessConfig) -> list[UnifiedExample]:
    source_cfg = cfg.datasets.pubmedqa
    splits = get_dataset_split_names(source_cfg.name, source_cfg.subset)
    split_names = [s for s in splits if s in {"train", "validation", "test"}] or splits

    output: list[UnifiedExample] = []
    for split_name in split_names:
        ds = load_dataset(source_cfg.name, source_cfg.subset, split=split_name)
        for row in ds:
            example_id = str(_coalesce(row.get("pubid"), row.get("id"), row.get("qid"), ""))
            question = _flatten_text(_coalesce(row.get("question"), row.get("query")))
            context = _flatten_text(
                _coalesce(
                    row.get("contexts"), row.get("context"), row.get("abstract"), row.get("text")
                )
            )
            decision = normalize_label(
                _coalesce(row.get("final_decision"), row.get("label"), "maybe")
            )
            long_answer = _flatten_text(
                _coalesce(row.get("long_answer"), row.get("answer"), row.get("final_answer"))
            )

            if not context:
                continue
            base_id = f"pubmedqa:{example_id or hashlib.md5(context.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}"

            cls_input = f"Question: {question}\n\nAbstract: {context}".strip()
            cls_meta = {"source_split": split_name, "question": question}
            output.append(
                UnifiedExample(
                    id=f"{base_id}:classification",
                    task="classification",
                    input=cls_input,
                    target_text=decision,
                    target_json="",
                    source="PubMedQA",
                    split=_normalize_split(split_name, base_id, cfg.seed),
                    meta_json=json.dumps(cls_meta),
                )
            )

            summary_target = long_answer or f"Question: {question}. Decision: {decision}."
            sum_meta = {"source_split": split_name, "question": question}
            output.append(
                UnifiedExample(
                    id=f"{base_id}:summarization",
                    task="summarization",
                    input=context,
                    target_text=summary_target,
                    target_json="",
                    source="PubMedQA",
                    split=_normalize_split(split_name, base_id + ":sum", cfg.seed),
                    meta_json=json.dumps(sum_meta),
                )
            )
    return output


def _bc5cdr_examples(cfg: HarnessConfig) -> list[UnifiedExample]:
    source_cfg = cfg.datasets.bc5cdr
    splits = get_dataset_split_names(source_cfg.name, source_cfg.subset)

    output: list[UnifiedExample] = []
    for split_name in splits:
        ds = load_dataset(source_cfg.name, source_cfg.subset, split=split_name)
        for row in ds:
            doc_id = str(_coalesce(row.get("document_id"), row.get("id"), ""))

            passages = row.get("passages") or []
            text_chunks: list[str] = []
            for p in passages:
                p_text = p.get("text") if isinstance(p, dict) else None
                text_chunks.append(_flatten_text(p_text))
            doc_text = " ".join(t for t in text_chunks if t).strip()
            if not doc_text:
                doc_text = _flatten_text(
                    _coalesce(row.get("document"), row.get("text"), row.get("sentence"))
                )
            if not doc_text:
                continue

            diseases: set[str] = set()
            chemicals: set[str] = set()
            entities = row.get("entities") or []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                etype = str(entity.get("type", "")).lower()
                mention = _flatten_text(entity.get("text", ""))
                if not mention:
                    continue
                mention = mention.strip()
                if "disease" in etype:
                    diseases.add(mention)
                elif "chemical" in etype:
                    chemicals.add(mention)

            target_obj = {
                "diseases": sorted(diseases),
                "chemicals": sorted(chemicals),
            }
            base_id = f"bc5cdr:{doc_id or hashlib.md5(doc_text.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}"

            output.append(
                UnifiedExample(
                    id=f"{base_id}:extraction",
                    task="extraction",
                    input=doc_text,
                    target_text="",
                    target_json=json.dumps(target_obj),
                    source="BC5CDR",
                    split=_normalize_split(split_name, base_id, cfg.seed),
                    meta_json=json.dumps({"source_split": split_name}),
                )
            )
    return output


def _synthetic_examples(seed: int) -> list[UnifiedExample]:
    _ = seed
    rows = [
        UnifiedExample(
            id="synthetic:1:classification",
            task="classification",
            input="Question: Does aspirin reduce fever?\n\nAbstract: Aspirin reduced fever in this cohort.",
            target_text="yes",
            target_json="",
            source="synthetic",
            split="test",
            meta_json=json.dumps({"note": "fallback synthetic sample"}),
        ),
        UnifiedExample(
            id="synthetic:1:summarization",
            task="summarization",
            input="Aspirin reduced fever and mild pain without severe side effects.",
            target_text="Aspirin reduced fever and mild pain in the study.",
            target_json="",
            source="synthetic",
            split="test",
            meta_json=json.dumps({"note": "fallback synthetic sample"}),
        ),
        UnifiedExample(
            id="synthetic:1:extraction",
            task="extraction",
            input="The patient had diabetes and was treated with metformin.",
            target_text="",
            target_json=json.dumps({"diseases": ["diabetes"], "chemicals": ["metformin"]}),
            source="synthetic",
            split="test",
            meta_json=json.dumps({"note": "fallback synthetic sample"}),
        ),
    ]
    return rows


def build_datasets(cfg: HarnessConfig) -> pd.DataFrame:
    all_examples: list[UnifiedExample] = []
    errors: list[str] = []

    try:
        all_examples.extend(_pubmedqa_examples(cfg))
    except Exception as exc:
        errors.append(f"PubMedQA load failed: {exc}")

    try:
        all_examples.extend(_bc5cdr_examples(cfg))
    except Exception as exc:
        errors.append(f"BC5CDR load failed: {exc}")

    required_tasks = {"classification", "summarization", "extraction"}
    existing_tasks = {e.task for e in all_examples}
    missing_tasks = required_tasks - existing_tasks
    if missing_tasks:
        console.print(
            f"Adding synthetic fallback samples for missing tasks: {sorted(missing_tasks)}"
        )
        for sample in _synthetic_examples(cfg.seed):
            if sample.task in missing_tasks:
                all_examples.append(sample)

    if not all_examples:
        all_examples = _synthetic_examples(cfg.seed)
        console.print("Falling back to synthetic samples because public dataset load failed.")
        for err in errors:
            console.print(f"- {err}")

    df = pd.DataFrame(asdict(x) for x in all_examples)
    if df.empty:
        raise RuntimeError("Dataset build produced zero examples")

    out_parquet = Path(cfg.paths.data_parquet)
    out_jsonl = Path(cfg.paths.data_jsonl)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_parquet, index=False)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = df.groupby("task").size().to_dict()
    console.print(f"Built dataset with {len(df)} rows. By task: {summary}")
    return df


def sample_for_run(df: pd.DataFrame, cfg: HarnessConfig) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for task in cfg.tasks:
        task_df = df[df["task"] == task]
        n = min(cfg.sampling.n_samples_per_task, len(task_df))
        if n == 0:
            continue
        sampled = task_df.sample(n=n, random_state=cfg.seed)
        parts.append(sampled)
    if not parts:
        raise RuntimeError("No task data available after sampling")
    return pd.concat(parts, ignore_index=True)
