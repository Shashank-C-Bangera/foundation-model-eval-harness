from __future__ import annotations

from dataclasses import dataclass

from fmeh.models.hf_local import HFLocalRunner, MockRunner


@dataclass(frozen=True)
class ModelSpec:
    runner: str
    model_name: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "flan_t5_small": ModelSpec(runner="hf_local", model_name="google/flan-t5-small"),
    "t5_small": ModelSpec(runner="hf_local", model_name="t5-small"),
    "tiny_gpt2": ModelSpec(runner="hf_local", model_name="sshleifer/tiny-gpt2"),
    "distilgpt2": ModelSpec(runner="hf_local", model_name="distilgpt2"),
    "tinyllama_chat": ModelSpec(runner="hf_local", model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    "mock_json": ModelSpec(runner="mock", model_name="mock_json"),
}


def build_runner(
    model_id: str,
    device: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model id '{model_id}'. Add it in models/registry.py")

    spec = MODEL_REGISTRY[model_id]
    if spec.runner == "hf_local":
        return HFLocalRunner(
            model_name=spec.model_name,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    if spec.runner == "mock":
        return MockRunner()
    raise ValueError(f"Unsupported runner type: {spec.runner}")
