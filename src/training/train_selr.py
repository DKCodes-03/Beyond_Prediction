"""CLI and data preparation for answer-only SELR training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.evaluation.metrics import normalize_answer_ids
from src.graph.graph_builder import build_temporal_graph
from src.graph.temporal_graph import TemporalGraph
from src.retrieval.subgraph_retriever import retrieve_subgraph
from src.training.trainer import SELRAnswerModel, TrainingResult, train_selr
from src.utils.logging_utils import (
    create_run_metadata,
    setup_logger,
    write_run_metadata,
)
from src.utils.seed import set_seed


class EncodedQADataset(Dataset[dict[str, Any]]):
    """Encode retrieved QA examples for the shared baseline/SELR batch contract."""

    def __init__(
        self,
        graph: TemporalGraph,
        encoder: GraphQuestionEncoder,
        records: Sequence[Mapping[str, Any]],
        *,
        split: str,
        graph_id: str,
        retrieval_settings: Mapping[str, Any],
        max_candidates: int,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError(f"{split} records must be a non-empty list")
        seen_ids: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"{split} record {index} must be a mapping")
            qa_id = record.get("qa_id")
            if not isinstance(qa_id, (str, int)) or isinstance(qa_id, bool):
                raise ValueError(f"{split} record {index} qa_id must be a string or integer")
            normalized_id = str(qa_id).strip()
            if not normalized_id:
                raise ValueError(f"{split} record {index} qa_id must not be empty")
            if normalized_id in seen_ids:
                raise ValueError(f"duplicate qa_id {normalized_id!r} in {split} records")
            seen_ids.add(normalized_id)
            if record.get("split") != split:
                raise ValueError(
                    f"{split} record {index} must have split={split!r}"
                )
            normalize_answer_ids(
                record.get("accepted_answer_ids"),
                field=f"{split} record {index} accepted_answer_ids",
            )
        self.graph = graph
        self.encoder = encoder
        self.records = tuple(records)
        self.split = split
        self.graph_id = graph_id
        self.retrieval_settings = dict(retrieval_settings)
        self.max_candidates = max_candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        qa_id = str(record["qa_id"]).strip()
        retrieved = retrieve_subgraph(
            self.graph,
            record,
            graph_id=self.graph_id,
            settings=self.retrieval_settings,
        )
        candidate_ids = tuple(retrieved["candidate_answer_ids"])
        if not candidate_ids:
            raise ValueError(
                f"{self.split} QA {qa_id!r} has no retrieved candidates "
                f"(status={retrieved['retrieval_status']!r})"
            )
        if len(candidate_ids) > self.max_candidates:
            raise ValueError(
                f"{self.split} QA {qa_id!r} has {len(candidate_ids)} candidates, "
                f"exceeding configured max_candidates={self.max_candidates}"
            )
        accepted_ids = normalize_answer_ids(
            record["accepted_answer_ids"],
            field=f"{self.split} QA {qa_id!r} accepted_answer_ids",
        )
        missing_answers = sorted(set(accepted_ids) - set(candidate_ids))
        if missing_answers:
            raise ValueError(
                f"{self.split} QA {qa_id!r} retrieved candidates are missing "
                f"accepted answers: {missing_answers!r}"
            )

        with torch.no_grad():
            encoded = self.encoder(retrieved)
        memory = torch.cat(
            (
                encoded.context_embedding.unsqueeze(0),
                encoded.node_embeddings,
                encoded.edge_embeddings,
            ),
            dim=0,
        )
        memory_type_ids = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=memory.device),
                torch.ones(
                    encoded.node_embeddings.shape[0],
                    dtype=torch.int64,
                    device=memory.device,
                ),
                torch.full(
                    (encoded.edge_embeddings.shape[0],),
                    2,
                    dtype=torch.int64,
                    device=memory.device,
                ),
            )
        )
        targets = torch.tensor(
            [
                float(candidate_id in accepted_ids)
                for candidate_id in encoded.candidate_ids
            ],
            dtype=encoded.candidate_embeddings.dtype,
            device=encoded.candidate_embeddings.device,
        )
        return {
            "encoder_memory": memory,
            "encoder_memory_type_ids": memory_type_ids,
            "context_embeddings": encoded.context_embedding,
            "candidate_embeddings": encoded.candidate_embeddings,
            "answer_targets": targets,
            "qa_id": qa_id,
            "candidate_ids": encoded.candidate_ids,
        }


def collate_encoded_examples(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad variable-size encoded examples and preserve their masks and types."""
    if not isinstance(examples, (list, tuple)) or not examples:
        raise ValueError("examples must be a non-empty list")
    first_memory = examples[0].get("encoder_memory")
    first_context = examples[0].get("context_embeddings")
    first_candidates = examples[0].get("candidate_embeddings")
    if (
        not isinstance(first_memory, Tensor)
        or first_memory.ndim != 2
        or not isinstance(first_context, Tensor)
        or first_context.ndim != 1
        or not isinstance(first_candidates, Tensor)
        or first_candidates.ndim != 2
    ):
        raise ValueError(
            "each example must contain rank-2 memory and candidate tensors "
            "and one context vector"
        )
    embedding_dim = first_memory.shape[1]
    if first_context.shape != (embedding_dim,):
        raise ValueError("context embedding dimension must match encoder memory")
    batch_size = len(examples)
    max_memory = max(example["encoder_memory"].shape[0] for example in examples)
    max_candidates = max(
        example["candidate_embeddings"].shape[0] for example in examples
    )
    if max_memory < 1 or max_candidates < 1:
        raise ValueError("every example must contain memory and candidate vectors")
    if (
        first_candidates.shape[1] != embedding_dim
        or first_candidates.dtype != first_memory.dtype
        or first_candidates.device != first_memory.device
        or first_context.dtype != first_memory.dtype
        or first_context.device != first_memory.device
    ):
        raise ValueError(
            "context, memory, and candidate embeddings must share dimension, "
            "dtype, and device"
        )

    memory = first_memory.new_zeros((batch_size, max_memory, embedding_dim))
    contexts = first_context.new_zeros((batch_size, embedding_dim))
    memory_mask = torch.zeros((batch_size, max_memory), dtype=torch.bool)
    memory_type_ids = torch.zeros((batch_size, max_memory), dtype=torch.int64)
    candidates = first_candidates.new_zeros(
        (batch_size, max_candidates, embedding_dim)
    )
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    targets = first_candidates.new_zeros((batch_size, max_candidates))

    for row, example in enumerate(examples):
        row_memory = example["encoder_memory"]
        row_types = example["encoder_memory_type_ids"]
        row_context = example["context_embeddings"]
        row_candidates = example["candidate_embeddings"]
        row_targets = example["answer_targets"]
        qa_id = example.get("qa_id")
        candidate_ids = example.get("candidate_ids")
        if (
            not isinstance(row_memory, Tensor)
            or not isinstance(row_types, Tensor)
            or not isinstance(row_context, Tensor)
            or not isinstance(row_candidates, Tensor)
            or not isinstance(row_targets, Tensor)
            or row_memory.ndim != 2
            or row_memory.shape[1] != embedding_dim
            or row_context.shape != (embedding_dim,)
            or row_types.shape != row_memory.shape[:1]
            or row_types.dtype != torch.int64
            or row_candidates.ndim != 2
            or row_candidates.shape[1] != embedding_dim
            or row_targets.shape != row_candidates.shape[:1]
            or row_memory.dtype != memory.dtype
            or row_context.dtype != contexts.dtype
            or row_candidates.dtype != candidates.dtype
            or row_targets.dtype != targets.dtype
            or row_memory.device != memory.device
            or row_context.device != contexts.device
            or row_candidates.device != candidates.device
            or row_targets.device != targets.device
            or row_types.device != memory_type_ids.device
        ):
            raise ValueError(
                f"example {row} has incompatible tensor shape, dtype, or device"
            )
        if (
            not isinstance(qa_id, str)
            or not qa_id.strip()
            or not isinstance(candidate_ids, (list, tuple))
            or len(candidate_ids) != row_candidates.shape[0]
            or any(not isinstance(candidate_id, str) for candidate_id in candidate_ids)
        ):
            raise ValueError(f"example {row} has invalid QA or candidate IDs")
        memory_count = row_memory.shape[0]
        candidate_count = row_candidates.shape[0]
        memory[row, :memory_count] = row_memory
        memory_mask[row, :memory_count] = True
        memory_type_ids[row, :memory_count] = row_types
        contexts[row] = row_context
        candidates[row, :candidate_count] = row_candidates
        candidate_mask[row, :candidate_count] = True
        targets[row, :candidate_count] = row_targets

    return {
        "encoder_memory": memory,
        "encoder_memory_mask": memory_mask,
        "encoder_memory_type_ids": memory_type_ids,
        "context_embeddings": contexts,
        "candidate_embeddings": candidates,
        "candidate_mask": candidate_mask,
        "answer_targets": targets,
        "qa_ids": tuple(str(example["qa_id"]) for example in examples),
        "candidate_ids": tuple(
            tuple(example["candidate_ids"]) for example in examples
        ),
    }


def run_selr_training(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    resume_from: str | Path | None = None,
    logger: logging.Logger | None = None,
    runtime_started_at: float | None = None,
) -> TrainingResult:
    """Prepare graph/QA batches and fit the answer-only SELR model.

    The graph encoder is initialized from the run seed and frozen while SELR is
    trained. QLoRA and explanation training are not part of this entry point.
    """
    training = _validate_training_config(training_config)
    (
        encoder_dim,
        dropout,
        latent_dim,
        max_steps,
        max_candidates,
        bottleneck_enabled,
    ) = (
        _validate_model_config(model_config)
    )
    retrieval_settings = _retrieval_settings(retrieval_config)
    set_seed(
        training["seed"],
        deterministic=training["deterministic"],
        num_threads=training["num_threads"],
    )
    graph = build_temporal_graph(facts)
    vocabulary = build_encoder_vocabulary(graph, train_records)
    encoder = GraphQuestionEncoder(
        vocabulary,
        embedding_dim=encoder_dim,
        dropout=dropout,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    graph_digest = hashlib.sha256(
        json.dumps(facts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    train_dataset = EncodedQADataset(
        graph,
        encoder,
        train_records,
        split="train",
        graph_id=graph_digest,
        retrieval_settings=retrieval_settings,
        max_candidates=max_candidates,
    )
    validation_dataset = EncodedQADataset(
        graph,
        encoder,
        validation_records,
        split="validation",
        graph_id=graph_digest,
        retrieval_settings=retrieval_settings,
        max_candidates=max_candidates,
    )
    loader_settings = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "collate_fn": collate_encoded_examples,
    }
    train_batches = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_settings,
    )
    validation_batches = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_settings,
    )
    model = SELRAnswerModel(
        encoder_dim=encoder_dim,
        latent_dim=latent_dim,
        max_steps=max_steps,
        bottleneck_enabled=bottleneck_enabled,
    )
    return train_selr(
        model,
        train_batches,
        validation_batches,
        config=training,
        seed=training["seed"],
        resume_from=resume_from,
        logger=logger,
        runtime_started_at=runtime_started_at,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the answer-only SELR model on prepared QA splits."
    )
    parser.add_argument("--facts", type=Path, default=Path("data/processed/facts.json"))
    parser.add_argument(
        "--train-split", type=Path, default=Path("data/splits/train.json")
    )
    parser.add_argument(
        "--validation-split", type=Path, default=Path("data/splits/val.json")
    )
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/training_config.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_config.yaml")
    )
    parser.add_argument(
        "--retrieval-config", type=Path, default=Path("configs/retrieval_config.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/selr"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args(argv)

    facts = _read_json_array(args.facts, "facts")
    train_records = _read_json_array(args.train_split, "training split")
    validation_records = _read_json_array(args.validation_split, "validation split")
    training_config = _read_yaml_mapping(args.training_config, "training config")
    model_config = _read_yaml_mapping(args.model_config, "model config")
    retrieval_config = _read_yaml_mapping(args.retrieval_config, "retrieval config")
    if args.seed is not None:
        training_config["seed"] = args.seed
    training_config = _validate_training_config(training_config)
    facts_sha256 = hashlib.sha256(args.facts.read_bytes()).hexdigest()
    manifest_path = args.facts.parent / "manifest.json"
    manifest_data: Mapping[str, Any] = {}
    if manifest_path.is_file():
        manifest_value = _read_json_object(manifest_path, "processed-artifact manifest")
        dataset_value = manifest_value.get("dataset", {})
        if not isinstance(dataset_value, Mapping):
            raise ValueError("processed-artifact manifest dataset must be an object")
        manifest_data = dataset_value
    dataset_id = manifest_data.get("id", "unknown")
    revision = manifest_data.get("revision")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("processed-artifact manifest dataset id must be non-empty")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ValueError("processed-artifact manifest revision must be a string or null")
    logger = setup_logger(
        "beyond_prediction.selr",
        log_dir=training_config["log_dir"],
    )
    result = run_selr_training(
        facts,
        train_records,
        validation_records,
        training_config=training_config,
        model_config=model_config,
        retrieval_config=retrieval_config,
        resume_from=args.resume_from,
        logger=logger,
    )

    run_metadata = create_run_metadata(
        config={
            "training": dict(training_config),
            "model": dict(model_config),
            "retrieval": dict(retrieval_config),
            "inputs": {
                "facts": str(args.facts),
                "train_split": str(args.train_split),
                "validation_split": str(args.validation_split),
            },
        },
        seed=training_config["seed"],
        split="validation",
        model_variant="selr",
        data_manifest={
            "dataset_id": dataset_id,
            "version": revision or facts_sha256,
            "checksum_sha256": facts_sha256,
        },
        metrics=asdict(result),
        measurement_conditions={
            "training_examples": len(train_records),
            "validation_examples": len(validation_records),
            "encoder_policy": "frozen",
            "qlora_used": False,
        },
        checkpoint_location=result.best_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = write_run_metadata(args.output_dir / "run.json", run_metadata)
    print(
        json.dumps(
            {"result": asdict(result), "run_metadata_path": str(run_path)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def _read_yaml_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a YAML object")
    return value


def _read_json_array(path: Path, field: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field} JSON from {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} at {path} must be a non-empty JSON array")
    if any(not isinstance(record, dict) for record in value):
        raise ValueError(f"every {field} record at {path} must be a JSON object")
    return value


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a JSON object")
    return value


def _validate_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("training config must be a mapping")
    values = dict(config)
    if values.get("model_variant") != "selr":
        raise ValueError("training config model_variant must be set to 'selr'")
    if values.get("objective") != "answer_only":
        raise ValueError("training config objective must be 'answer_only'")
    seed = values.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(values.get("deterministic"), bool):
        raise ValueError("deterministic must be a boolean")
    for field in (
        "num_threads",
        "max_epochs",
        "batch_size",
        "max_steps",
        "max_runtime_seconds",
    ):
        _positive_int(values.get(field), field)
    workers = values.get("num_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    device = values.get("device")
    if not isinstance(device, str) or device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    for field in ("checkpoint_dir", "log_dir"):
        if not isinstance(values.get(field), str) or not values[field].strip():
            raise ValueError(f"{field} must be a non-empty path")
    return values


def _validate_model_config(
    config: Mapping[str, Any],
) -> tuple[int, float, int, int, int, bool]:
    if not isinstance(config, Mapping):
        raise ValueError("model config must be a mapping")
    if config.get("variant") != "selr":
        raise ValueError("model config variant must be set to 'selr'")
    encoder = config.get("encoder")
    answer_head = config.get("answer_head")
    selr = config.get("selr")
    if not isinstance(encoder, Mapping) or not isinstance(answer_head, Mapping):
        raise ValueError("model config must include encoder and answer_head objects")
    if not isinstance(selr, Mapping) or selr.get("enabled") is not True:
        raise ValueError("model config must enable SELR")
    if encoder.get("type") != "entity_relation_time_embeddings":
        raise ValueError("encoder type must be entity_relation_time_embeddings")
    if encoder.get("aggregation") != "mean":
        raise ValueError("encoder aggregation must be mean")
    if answer_head.get("candidate_policy") != "retrieved_entities":
        raise ValueError("answer_head candidate_policy must be retrieved_entities")
    embedding_dim = _positive_int(encoder.get("embedding_dim"), "encoder.embedding_dim")
    latent_dim = _positive_int(selr.get("latent_dim"), "selr.latent_dim")
    max_steps = _positive_int(selr.get("max_steps"), "selr.max_steps")
    bottleneck_enabled = selr.get("bottleneck_enabled", True)
    if not isinstance(bottleneck_enabled, bool):
        raise ValueError("selr.bottleneck_enabled must be a boolean")
    max_candidates = _positive_int(
        answer_head.get("max_candidates"), "answer_head.max_candidates"
    )
    dropout = encoder.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("encoder.dropout must be a number between 0 and 1")
    if not 0 <= dropout <= 1:
        raise ValueError("encoder.dropout must be a number between 0 and 1")
    return (
        embedding_dim,
        float(dropout),
        latent_dim,
        max_steps,
        max_candidates,
        bottleneck_enabled,
    )


def _retrieval_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("retrieval config must be a mapping")
    return {
        name: config[name]
        for name in ("strategy", "max_hops", "max_nodes", "max_edges")
        if name in config
    }


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
