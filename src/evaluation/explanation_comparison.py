"""Compare SELR answer performance before and after explanation training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.evaluation.metrics import compute_answer_metrics, normalize_answer_ids
from src.graph.graph_builder import build_temporal_graph
from src.graph.graph_utils import canonical_id
from src.ingestion.explanation_targets import generate_training_explanation_targets
from src.training.train_selr import EncodedQADataset, collate_encoded_examples
from src.training.trainer import SELRAnswerModel, train_selr_explanations
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed


def compare_answer_predictions(
    qa_records: Sequence[Mapping[str, Any]],
    before_predictions: Sequence[Mapping[str, Any]],
    after_predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare paired predictions and retain all post-training answer failures."""
    before_metrics = compute_answer_metrics(qa_records, before_predictions)
    after_metrics = compute_answer_metrics(qa_records, after_predictions)
    labels = {
        canonical_id(record["qa_id"], "qa_id"): normalize_answer_ids(
            record["accepted_answer_ids"], field="accepted_answer_ids"
        )
        for record in qa_records
    }
    before = _prediction_map(before_predictions)
    after = _prediction_map(after_predictions)
    before_correct = {
        qa_id: set(before[qa_id]) == set(gold) for qa_id, gold in labels.items()
    }
    after_correct = {
        qa_id: set(after[qa_id]) == set(gold) for qa_id, gold in labels.items()
    }

    after_failures = []
    for record in qa_records:
        qa_id = canonical_id(record["qa_id"], "qa_id")
        if after_correct[qa_id]:
            continue
        after_failures.append(
            {
                "qa_id": qa_id,
                "accepted_answer_ids": list(labels[qa_id]),
                "before_predicted_answer_ids": list(before[qa_id]),
                "after_predicted_answer_ids": list(after[qa_id]),
                "failure_type": (
                    "regression" if before_correct[qa_id] else "persistent_failure"
                ),
                "query": {
                    field: record[field]
                    for field in (
                        "seed_entity_ids",
                        "relation_sequence",
                        "temporal_constraint",
                    )
                    if field in record
                },
            }
        )

    improved_count = sum(
        not before_correct[qa_id] and after_correct[qa_id] for qa_id in labels
    )
    regression_count = sum(
        before_correct[qa_id] and not after_correct[qa_id] for qa_id in labels
    )
    summary = {
        "example_count": len(labels),
        "before_explanation_training": before_metrics,
        "after_explanation_training": after_metrics,
        "exact_match_delta": (
            after_metrics["exact_match_accuracy"]
            - before_metrics["exact_match_accuracy"]
        ),
        "improved_count": improved_count,
        "regression_count": regression_count,
        "after_failure_count": len(after_failures),
    }
    failures = {
        "failure_definition": (
            "Every test QA incorrectly answered after explanation training; "
            "classified as regression if the answer-only checkpoint was correct, "
            "otherwise persistent_failure."
        ),
        "failures": after_failures,
    }
    return summary, failures


def run_explanation_comparison(
    facts: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    input_hashes: Mapping[str, str],
    answer_checkpoint: str | Path,
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    output_dir: str | Path,
    seed: int = 17,
    explanation_loss_weight: float = 0.5,
) -> dict[str, Any]:
    """Train explanation stage two and compare both checkpoints on test once."""
    _validate_inputs(facts, splits, input_hashes)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(explanation_loss_weight, bool)
        or not isinstance(explanation_loss_weight, (int, float))
        or explanation_loss_weight <= 0
    ):
        raise ValueError("explanation_loss_weight must be positive")

    destination = Path(output_dir)
    result_path = destination / "comparison.json"
    if result_path.exists():
        raise FileExistsError(
            f"{result_path} already exists; choose a fresh output directory"
        )
    destination.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        answer_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("answer checkpoint root must be a mapping")
    answer_config_value = checkpoint.get("config")
    if not isinstance(answer_config_value, Mapping):
        raise ValueError("answer checkpoint config must be a mapping")
    answer_config = dict(answer_config_value)
    if (
        answer_config.get("model_variant") != "selr"
        or answer_config.get("objective") != "answer_only"
        or answer_config.get("seed") != seed
    ):
        raise ValueError(
            "answer checkpoint must be an answer_only SELR checkpoint for "
            f"seed {seed}"
        )

    training = dict(answer_config)
    stage_two_config = dict(answer_config)
    stage_two_config.update(
        objective="answer_plus_explanation",
        checkpoint_dir=str(destination / "checkpoints"),
    )

    set_seed(
        seed,
        deterministic=bool(training["deterministic"]),
        num_threads=training["num_threads"],
    )
    graph = build_temporal_graph(facts)
    vocabulary = build_encoder_vocabulary(graph, splits["train"])
    encoder_config = model_config.get("encoder")
    answer_head_config = model_config.get("answer_head")
    selr_config = model_config.get("selr")
    if not all(
        isinstance(value, Mapping)
        for value in (encoder_config, answer_head_config, selr_config)
    ):
        raise ValueError("model_config must contain encoder, answer_head, and selr")
    encoder_dim = _positive_int(encoder_config.get("embedding_dim"), "embedding_dim")
    encoder = GraphQuestionEncoder(
        vocabulary,
        embedding_dim=encoder_dim,
        dropout=float(encoder_config["dropout"]),
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    retrieval_settings = {
        name: retrieval_config[name]
        for name in ("strategy", "max_hops", "max_nodes", "max_edges")
        if name in retrieval_config
    }
    graph_digest = hashlib.sha256(
        json.dumps(facts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    datasets = {
        split: EncodedQADataset(
            graph,
            encoder,
            records,
            split=split,
            graph_id=graph_digest,
            retrieval_settings=retrieval_settings,
            max_candidates=_positive_int(
                answer_head_config.get("max_candidates"), "max_candidates"
            ),
        )
        for split, records in splits.items()
    }

    targets = generate_training_explanation_targets(graph, splits["train"])
    target_sequences, target_vocabulary = _encode_targets(targets)
    stage_two_config["explanation"] = {
        "vocab_size": len(target_vocabulary),
        "max_decode_steps": max(map(len, target_sequences.values())),
        "pad_token_id": 0,
        "start_token_id": 1,
        "end_token_id": 2,
        "loss_weight": float(explanation_loss_weight),
    }
    train_dataset = _ExplanationDataset(datasets["train"], target_sequences)
    loader_settings = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
    }
    train_batches = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=_collate_explanation_examples,
        **loader_settings,
    )
    validation_batches = DataLoader(
        datasets["validation"],
        shuffle=False,
        collate_fn=collate_encoded_examples,
        **loader_settings,
    )

    model = SELRAnswerModel(
        encoder_dim=encoder_dim,
        latent_dim=_positive_int(selr_config.get("latent_dim"), "latent_dim"),
        max_steps=_positive_int(selr_config.get("max_steps"), "max_steps"),
    )
    started_at = time.monotonic()
    training_result = train_selr_explanations(
        model,
        train_batches,
        validation_batches,
        answer_checkpoint=answer_checkpoint,
        answer_config=answer_config,
        config=stage_two_config,
        seed=seed,
        runtime_started_at=started_at,
    )
    if training_result.stop_reason == "max_runtime":
        raise RuntimeError(
            "explanation training hit max_runtime; comparison was not evaluated"
        )

    test_batches = DataLoader(
        datasets["test"],
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        collate_fn=collate_encoded_examples,
    )
    answer_model = SELRAnswerModel(
        encoder_dim=encoder_dim,
        latent_dim=_positive_int(selr_config.get("latent_dim"), "latent_dim"),
        max_steps=_positive_int(selr_config.get("max_steps"), "max_steps"),
    )
    _load_selected_checkpoint(answer_model, answer_checkpoint, answer_config, seed)
    before_predictions = _predict_answers(
        answer_model, test_batches, training["device"]
    )

    stage_two_model = SELRAnswerModel(
        encoder_dim=encoder_dim,
        latent_dim=_positive_int(selr_config.get("latent_dim"), "latent_dim"),
        max_steps=_positive_int(selr_config.get("max_steps"), "max_steps"),
    )
    stage_two_model.enable_explanation_decoder(
        vocab_size=stage_two_config["explanation"]["vocab_size"],
        max_decode_steps=stage_two_config["explanation"]["max_decode_steps"],
        pad_token_id=stage_two_config["explanation"]["pad_token_id"],
    )
    _load_selected_checkpoint(
        stage_two_model,
        training_result.best_checkpoint,
        stage_two_config,
        seed,
    )
    after_predictions = _predict_answers(
        stage_two_model, test_batches, training["device"]
    )

    summary, failures = compare_answer_predictions(
        splits["test"], before_predictions, after_predictions
    )
    _write_json(destination / "failures.json", failures)
    _write_json(destination / "before_predictions.json", before_predictions)
    _write_json(destination / "after_predictions.json", after_predictions)
    result = {
        "benchmark": "selr_answer_before_after_explanation_training",
        "status": "complete",
        "seed": seed,
        "dataset": "ICEWS14 pilot",
        "input_sha256": dict(input_hashes),
        "answer_checkpoint": str(answer_checkpoint),
        "explanation_checkpoint": training_result.best_checkpoint,
        "training": {
            "stop_reason": training_result.stop_reason,
            "global_step": training_result.global_step,
            "epochs_completed": training_result.epochs_completed,
            "best_validation_exact_match": (
                training_result.best_validation_exact_match
            ),
            "elapsed_seconds": time.monotonic() - started_at,
            "history": list(training_result.history),
            "config": stage_two_config,
            "explanation_target_count": len(target_sequences),
            "explanation_vocabulary_size": len(target_vocabulary),
            "explanation_target_policy": (
                "one train-only path per QA; for multiple accepted answers, "
                "use the lexicographically first canonical answer ID"
            ),
        },
        "test_comparison": summary,
        "artifacts": {
            "before_predictions": "before_predictions.json",
            "after_predictions": "after_predictions.json",
            "failures": "failures.json",
        },
    }
    _write_json(result_path, result)
    return result


class _ExplanationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: EncodedQADataset,
        target_sequences: Mapping[str, Sequence[int]],
    ) -> None:
        self.dataset = dataset
        self.target_sequences = target_sequences

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = dict(self.dataset[index])
        target_ids = self.target_sequences.get(example["qa_id"])
        if target_ids is None:
            raise ValueError(
                f"training QA {example['qa_id']!r} has no explanation target"
            )
        example["explanation_target_ids"] = list(target_ids)
        return example


def _collate_explanation_examples(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples must be a non-empty list")
    batch = collate_encoded_examples(examples)
    lengths = [len(example["explanation_target_ids"]) for example in examples]
    max_length = max(lengths)
    target_ids = torch.zeros((len(examples), max_length), dtype=torch.long)
    target_mask = torch.zeros((len(examples), max_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = lengths[row]
        target_ids[row, :length] = torch.tensor(
            example["explanation_target_ids"], dtype=torch.long
        )
        target_mask[row, :length] = True
    batch["explanation_target_ids"] = target_ids
    batch["explanation_target_mask"] = target_mask
    return batch


def _encode_targets(
    targets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    target_text_by_qa: dict[str, str] = {}
    for target in targets:
        qa_id = str(target["qa_id"])
        target_text_by_qa.setdefault(qa_id, str(target["target_text"]))
    vocabulary = {
        token: index
        for index, token in enumerate(
            ("<pad>", "<bos>", "<eos>")
            + tuple(
                sorted(
                    {
                        token
                        for text in target_text_by_qa.values()
                        for token in text.split()
                    }
                    - {"<pad>", "<bos>", "<eos>"}
                )
            )
        )
    }
    encoded = {
        qa_id: [vocabulary[token] for token in text.split()]
        + [vocabulary["<eos>"]]
        for qa_id, text in target_text_by_qa.items()
    }
    return encoded, vocabulary


def _load_selected_checkpoint(
    model: SELRAnswerModel,
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_seed=seed,
        map_location=_resolve_device(config["device"]),
    )


def _predict_answers(
    model: SELRAnswerModel,
    batches: DataLoader,
    configured_device: str,
) -> list[dict[str, Any]]:
    device = _resolve_device(configured_device)
    model.to(device)
    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in batches:
            device_batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            probabilities = torch.sigmoid(model(device_batch))
            for row, (qa_id, candidate_ids) in enumerate(
                zip(batch["qa_ids"], batch["candidate_ids"], strict=True)
            ):
                scores = probabilities[row, : len(candidate_ids)].tolist()
                predicted = [
                    candidate_id
                    for candidate_id, score in zip(
                        candidate_ids, scores, strict=True
                    )
                    if score >= 0.5
                ]
                if not predicted:
                    best_index = max(range(len(scores)), key=scores.__getitem__)
                    predicted = [candidate_ids[best_index]]
                predictions.append(
                    {
                        "qa_id": qa_id,
                        "model_variant": "selr",
                        "status": "ok",
                        "predicted_answer_ids": predicted,
                        "answer_scores": [
                            {"answer_id": candidate_id, "score": float(score)}
                            for candidate_id, score in zip(
                                candidate_ids, scores, strict=True
                            )
                        ],
                    }
                )
    return predictions


def _prediction_map(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    return {
        canonical_id(prediction["qa_id"], "qa_id"): normalize_answer_ids(
            prediction["predicted_answer_ids"],
            field="predicted_answer_ids",
            allow_empty=True,
        )
        for prediction in predictions
    }


def _validate_inputs(
    facts: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    input_hashes: Mapping[str, str],
) -> None:
    if not isinstance(facts, (list, tuple)) or not facts:
        raise ValueError("facts must be a non-empty sequence")
    for split in ("train", "validation", "test"):
        if split not in splits or not isinstance(splits[split], (list, tuple)):
            raise ValueError(f"{split} split must be a sequence")
        if not splits[split]:
            raise ValueError(f"{split} split must not be empty")
        if split not in input_hashes or not isinstance(input_hashes[split], str):
            raise ValueError(f"input_hashes must include {split}")
    if "facts" not in input_hashes:
        raise ValueError("input_hashes must include facts")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _read_json_array(path: Path, field: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field} JSON from {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} at {path} must be a non-empty JSON array")
    if any(not isinstance(record, dict) for record in value):
        raise ValueError(f"every {field} record at {path} must be an object")
    return value


def _read_yaml_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a YAML object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    artifact_root = Path(
        "outputs/experiments/graph_path_sanity_test_pilot/input_artifacts"
    )
    parser = argparse.ArgumentParser(
        description="Compare SELR answers before and after explanation training."
    )
    parser.add_argument(
        "--facts", type=Path, default=artifact_root / "processed" / "facts.json"
    )
    parser.add_argument(
        "--train-split", type=Path, default=artifact_root / "splits" / "train.json"
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=artifact_root / "splits" / "val.json",
    )
    parser.add_argument(
        "--test-split", type=Path, default=artifact_root / "splits" / "test.json"
    )
    parser.add_argument(
        "--answer-checkpoint",
        type=Path,
        default=Path(
            "outputs/experiments/task44_learned_comparison/"
            "selr_seed_17/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_config.yaml")
    )
    parser.add_argument(
        "--retrieval-config", type=Path, default=Path("configs/retrieval_config.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/task53_explanation_comparison"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--explanation-loss-weight", type=float, default=0.5)
    args = parser.parse_args(argv)

    paths = {
        "facts": args.facts,
        "train": args.train_split,
        "validation": args.validation_split,
        "test": args.test_split,
    }
    arrays = {name: _read_json_array(path, name) for name, path in paths.items()}
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    result = run_explanation_comparison(
        arrays["facts"],
        {name: arrays[name] for name in ("train", "validation", "test")},
        input_hashes=hashes,
        answer_checkpoint=args.answer_checkpoint,
        model_config=_read_yaml_mapping(args.model_config, "model config"),
        retrieval_config=_read_yaml_mapping(args.retrieval_config, "retrieval config"),
        output_dir=args.output_dir,
        seed=args.seed,
        explanation_loss_weight=args.explanation_loss_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
