from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

TaskName = Literal["classification", "summarization", "extraction"]


class DatasetSource(BaseModel):
    name: str
    subset: str | None = None


class DatasetsConfig(BaseModel):
    pubmedqa: DatasetSource
    bc5cdr: DatasetSource


class SamplingConfig(BaseModel):
    n_samples_per_task: int = 20


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128


class PathsConfig(BaseModel):
    data_parquet: str = "data/processed/examples.parquet"
    data_jsonl: str = "data/processed/examples.jsonl"
    runs_root: str = "runs"


class MlflowConfig(BaseModel):
    experiment_name: str = "fmeh"
    tracking_uri: str = ""


class RagConfig(BaseModel):
    enabled: bool = False
    top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "data/processed/rag.index"
    passages_path: str = "data/processed/rag_passages.parquet"


class HarnessConfig(BaseModel):
    name: str = "default"
    seed: int = 42
    device: str = "cpu"
    tasks: list[TaskName] = Field(
        default_factory=lambda: ["classification", "summarization", "extraction"]
    )
    prompt_versions: list[str] = Field(default_factory=lambda: ["v1"])
    models: list[str] = Field(default_factory=lambda: ["flan_t5_small"])
    datasets: DatasetsConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    rag: RagConfig = Field(default_factory=RagConfig)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_experiment_config(experiment: str, config_dir: str | Path = "configs") -> HarnessConfig:
    root = _repo_root()
    cfg_dir = (root / config_dir).resolve()
    default_cfg_path = cfg_dir / "default.yaml"
    exp_cfg_path = cfg_dir / "experiments" / f"{experiment}.yaml"

    default_cfg = OmegaConf.load(default_cfg_path)
    exp_cfg = (
        OmegaConf.load(exp_cfg_path)
        if exp_cfg_path.exists()
        else OmegaConf.create({"name": experiment})
    )

    merged = OmegaConf.merge(default_cfg, exp_cfg)
    payload = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Resolved config is not a mapping")
    payload.setdefault("name", experiment)
    return HarnessConfig.model_validate(payload)


def save_resolved_config(config: HarnessConfig, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
