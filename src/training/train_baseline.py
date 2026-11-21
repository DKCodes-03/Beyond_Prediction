"""Prepare and train the matched non-SELR candidate-answer baseline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.graph.graph_builder import build_temporal_graph
from src.training.train_selr import EncodedQADataset, collate_encoded_examples
from src.training.trainer import BaselineAnswerModel, TrainingResult, train_baseline
from src.utils.seed import set_seed


def run_baseline_training(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    runtime_started_at: float | None = None,
) -> TrainingResult:
    """Train the non-SELR model with the same frozen encoder and QA pipeline."""
    training = dict(training_config)
    if training.get("model_variant") != "baseline":
        raise ValueError("training config model_variant must be 'baseline'")
    if training.get("objective") != "answer_only":
        raise ValueError("training config objective must be 'answer_only'")
    encoder_config = model_config.get("encoder")
    answer_head_config = model_config.get("answer_head")
    if not isinstance(encoder_config, Mapping) or not isinstance(
        answer_head_config, Mapping
    ):
        raise ValueError("model config must contain encoder and answer_head objects")
    if (
        model_config.get("variant") != "baseline"
        or encoder_config.get("type") != "entity_relation_time_embeddings"
        or encoder_config.get("aggregation") != "mean"
        or answer_head_config.get("type") != "candidate_scoring"
        or answer_head_config.get("candidate_policy") != "retrieved_entities"
    ):
        raise ValueError("model config does not match the baseline comparison policy")
    embedding_dim = _positive_int(encoder_config.get("embedding_dim"), "embedding_dim")
    max_candidates = _positive_int(
        answer_head_config.get("max_candidates"), "max_candidates"
    )
    dropout = encoder_config.get("dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= dropout <= 1
    ):
        raise ValueError("encoder.dropout must be between 0 and 1")
    retrieval_settings = {
        name: retrieval_config[name]
        for name in ("strategy", "max_hops", "max_nodes", "max_edges")
        if name in retrieval_config
    }

    seed = training.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training seed must be a non-negative integer")
    set_seed(
        seed,
        deterministic=training.get("deterministic", True),
        num_threads=training.get("num_threads", 1),
    )
    graph = build_temporal_graph(facts)
    vocabulary = build_encoder_vocabulary(graph, train_records)
    encoder = GraphQuestionEncoder(
        vocabulary,
        embedding_dim=embedding_dim,
        dropout=float(dropout),
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
        "batch_size": _positive_int(training.get("batch_size"), "batch_size"),
        "num_workers": training.get("num_workers", 0),
        "collate_fn": collate_encoded_examples,
    }
    train_batches = DataLoader(train_dataset, shuffle=True, **loader_settings)
    validation_batches = DataLoader(
        validation_dataset, shuffle=False, **loader_settings
    )
    model = BaselineAnswerModel(embedding_dim)
    return train_baseline(
        model,
        train_batches,
        validation_batches,
        config=training,
        seed=seed,
        runtime_started_at=runtime_started_at,
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value
