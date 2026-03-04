from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from fmeh.eval.validators import normalize_label


@dataclass
class ModelResponse:
    text: str
    latency_sec: float
    prompt_tokens: int
    output_tokens: int


class HFLocalRunner:
    def __init__(
        self,
        model_name: str,
        device: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name)
        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._is_encoder_decoder = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self._is_encoder_decoder = False

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self._device_obj = torch.device("cuda")
        else:
            self._device_obj = torch.device("cpu")

        self.model.eval()

    def generate(self, prompt: str, task: str) -> ModelResponse:
        _ = task
        started = time.perf_counter()

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if (
            not isinstance(tokenizer_limit, int)
            or tokenizer_limit <= 0
            or tokenizer_limit > 1_000_000
        ):
            tokenizer_limit = 1024
        max_input_len = min(1024, tokenizer_limit)
        if not self._is_encoder_decoder:
            model_limit = getattr(self.model.config, "max_position_embeddings", None)
            if isinstance(model_limit, int) and model_limit > 0:
                # Reserve room for generated tokens to avoid causal-model position overflows.
                max_input_len = min(max_input_len, model_limit - self.max_new_tokens - 1)
        max_input_len = max(32, max_input_len)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        )
        inputs = {k: v.to(self._device_obj) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "temperature": self.temperature if do_sample else None,
            "top_p": self.top_p if do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        if self._is_encoder_decoder:
            generated_ids = output_ids[0]
        else:
            prompt_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0][prompt_len:]

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        ended = time.perf_counter()

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        output_tokens = int(generated_ids.shape[-1]) if generated_ids.ndim > 0 else 0
        return ModelResponse(
            text=text,
            latency_sec=ended - started,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    def close(self) -> None:
        del self.model
        del self.tokenizer


class MockRunner:
    def __init__(self) -> None:
        pass

    def generate(self, prompt: str, task: str) -> ModelResponse:
        _ = prompt
        started = time.perf_counter()
        if task == "classification":
            payload = {
                "label": normalize_label("maybe"),
                "rationale": "insufficient explicit evidence",
            }
        elif task == "summarization":
            payload = {"summary": "Biomedical abstract discussing clinical evidence and outcomes."}
        else:
            payload = {"diseases": ["cancer"], "chemicals": ["aspirin"]}
        ended = time.perf_counter()
        text = json.dumps(payload)
        return ModelResponse(
            text=text,
            latency_sec=ended - started,
            prompt_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    def close(self) -> None:
        return None
