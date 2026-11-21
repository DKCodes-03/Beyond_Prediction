"""Lazy-loading wrapper for optional 4-bit Qwen2.5 inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class QwenWrapper:
    """Load Qwen once and provide chat-style or plain-text generation.

    This wrapper intentionally remains outside the core answer pipeline. It
    supports 4-bit inference via bitsandbytes, but never performs full-model
    fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        revision: str | None = None,
        device_map: str | dict[str, Any] = "auto",
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        use_double_quant: bool = True,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be a non-empty string or None")
        if not isinstance(load_in_4bit, bool):
            raise ValueError("load_in_4bit must be a boolean")
        if bnb_4bit_compute_dtype not in {"float16", "bfloat16"}:
            raise ValueError("bnb_4bit_compute_dtype must be 'float16' or 'bfloat16'")
        if bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("bnb_4bit_quant_type must be 'nf4' or 'fp4'")
        if not isinstance(use_double_quant, bool):
            raise ValueError("use_double_quant must be a boolean")

        self.model_name = model_name.strip()
        self.revision = revision
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_double_quant = use_double_quant
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> "QwenWrapper":
        if self.model is not None and self.tokenizer is not None:
            return self
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Qwen inference requires transformers, accelerate, torch, and "
                "bitsandbytes. Install the optional 4-bit inference stack."
            ) from exc

        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "revision": self.revision,
        }
        if self.load_in_4bit:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.use_double_quant,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, self.bnb_4bit_compute_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs
        )
        return self

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")

        self.load()
        assert self.tokenizer is not None
        assert self.model is not None
        messages = [{"role": "user", "content": prompt.strip()}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt.strip()

        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {
                name: tensor.to(device) for name, tensor in inputs.items()
            }
        inputs = dict(inputs)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)
        outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
