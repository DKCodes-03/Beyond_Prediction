"""Optional QLoRA training entry point for the separate Qwen integration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.llm.prompt_templates import build_answer_prompt


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def run_qlora_training(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Path:
    """Fine-tune only LoRA adapters on a 4-bit Qwen base model.

    This is an opt-in comparison path. It does not alter or replace the baseline
    or SELR checkpoints and requires the separate PEFT/bitsandbytes stack.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a non-empty sequence")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("records must contain mappings")
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError("every record must contain a non-empty question")
        if not isinstance(record.get("accepted_answer_ids"), (list, tuple)):
            raise ValueError("every record must contain accepted_answer_ids")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    for name, value in (
        ("num_epochs", num_epochs),
        ("batch_size", batch_size),
        ("lora_r", lora_r),
        ("lora_alpha", lora_alpha),
        ("max_seq_length", max_seq_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(lora_dropout, bool) or not isinstance(lora_dropout, (int, float)):
        raise ValueError("lora_dropout must be a number between 0 and 1")
    if not 0 <= float(lora_dropout) <= 1:
        raise ValueError("lora_dropout must be a number between 0 and 1")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError("learning_rate must be a positive number")
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate must be a positive number")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA training requires transformers, accelerate, datasets, peft, "
            "torch, bitsandbytes, and a CUDA-capable environment."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires CUDA; CPU full/adapter training is unsupported")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    rows = []
    for record in records:
        evidence = record.get("evidence", [])
        if evidence is None:
            evidence = []
        prompt = build_answer_prompt(
            record["question"],
            evidence,
            answer_ids=record.get("candidate_answer_ids"),
        )
        response = _json_answer(record["accepted_answer_ids"])
        messages = [
            {"role": "system", "content": "You are a precise temporal knowledge-graph reasoning assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if hasattr(tokenizer, "apply_chat_template")
            else f"{prompt}\n{response}"
        )
        rows.append({"text": text})

    dataset = Dataset.from_list(rows)

    def tokenize(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        token_ids = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        token_ids["labels"] = [
            list(input_ids) for input_ids in token_ids["input_ids"]
        ]
        return token_ids

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
    args = TrainingArguments(
        output_dir=str(destination),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=float(learning_rate),
        fp16=True,
        report_to=[],
        logging_steps=1,
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return destination


def _json_answer(answer_ids: Sequence[str]) -> str:
    return json.dumps(
        {"answer_ids": list(answer_ids)},
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional Qwen 4-bit QLoRA training")
    parser.add_argument("--records", required=True, help="JSON array or JSONL/JSON records path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    path = Path(args.records)
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError("records JSON must contain an array")
            records = value
    else:
        value = json.loads(args.records)
        if not isinstance(value, list):
            raise ValueError("records JSON must contain an array")
        records = value

    run_qlora_training(
        records,
        output_dir=args.output_dir,
        model_name=args.model,
        revision=args.revision,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
