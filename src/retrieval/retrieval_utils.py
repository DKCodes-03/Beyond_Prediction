"""Utilities for retrieval cache keys and supporting-path diagnostics."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from src.graph.graph_utils import canonical_id
from src.retrieval.subgraph_retriever import _validate_query, _validate_settings


_SPLITS = frozenset({"train", "validation", "test"})
_MODEL_STATUSES = frozenset(
    {"ok", "retrieval_miss", "no_candidates", "model_error"}
)
_FAILURE_CATEGORIES = frozenset(
    {"correct", "retrieval_miss", "reasoning_error", "model_error"}
)


def retrieval_cache_key(
    dataset_version: str,
    graph_version: str,
    split: str,
    qa_record: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    example_id: str | None = None,
) -> str:
    """Return a stable digest isolating a retrieval result's inputs.

    The query is normalized with the same validation used by
    ``retrieve_subgraph``. QA labels and provenance are intentionally excluded.
    """
    normalized_dataset_version = _non_empty_string(
        dataset_version, "dataset_version"
    )
    normalized_graph_version = _non_empty_string(graph_version, "graph_version")
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")

    query = _validate_query(qa_record)
    if "split" in qa_record and qa_record["split"] != split:
        raise ValueError("split must match qa_record split")
    resolved_settings = _validate_settings(settings, len(query["relation_sequence"]))
    normalized_example_id = canonical_id(
        query["qa_id"] if example_id is None else example_id, "example_id"
    )
    key_payload = {
        "key_schema": 1,
        "dataset_version": normalized_dataset_version,
        "graph_version": normalized_graph_version,
        "split": split,
        "query": {
            **query,
            "example_id": normalized_example_id,
        },
        "settings": resolved_settings,
    }
    serialized = json.dumps(
        key_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assess_retrieval_support(
    qa_record: Mapping[str, Any],
    retrieved_subgraph: Mapping[str, Any],
    model_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure witness-path coverage and attribute one prediction outcome.

    A query is a retrieval miss when its complete declared witness path is
    absent from the retrieved facts. Otherwise, a wrong or unavailable answer
    is a reasoning error, except an explicit model error which remains separate.
    """
    qa_id = _record_id(qa_record, "qa_record")
    if _record_id(retrieved_subgraph, "retrieved_subgraph") != qa_id:
        raise ValueError("retrieved_subgraph qa_id must match qa_record")
    if _record_id(model_output, "model_output") != qa_id:
        raise ValueError("model_output qa_id must match qa_record")

    path_values = qa_record.get("supporting_path_fact_ids")
    if (
        not isinstance(path_values, (list, tuple))
        or not path_values
    ):
        raise ValueError("supporting_path_fact_ids must be a non-empty list")
    supporting_path = tuple(
        canonical_id(value, "supporting_path_fact_id") for value in path_values
    )

    retrieved_facts = retrieved_subgraph.get("facts")
    if not isinstance(retrieved_facts, (list, tuple)):
        raise ValueError("retrieved_subgraph facts must be a list")
    retrieved_fact_ids = {
        canonical_id(fact["fact_id"], f"retrieved fact {index} fact_id")
        for index, fact in enumerate(retrieved_facts)
        if isinstance(fact, Mapping) and "fact_id" in fact
    }
    if any(
        not isinstance(fact, Mapping) or "fact_id" not in fact
        for fact in retrieved_facts
    ):
        raise ValueError("each retrieved fact must include fact_id")

    covered_path = tuple(
        fact_id for fact_id in supporting_path if fact_id in retrieved_fact_ids
    )
    missing_path = tuple(
        dict.fromkeys(
            fact_id for fact_id in supporting_path if fact_id not in retrieved_fact_ids
        )
    )
    path_complete = not missing_path

    answer_values = qa_record.get("accepted_answer_ids")
    prediction_values = model_output.get("predicted_answer_ids")
    if not isinstance(answer_values, (list, tuple)) or not answer_values:
        raise ValueError("accepted_answer_ids must be a non-empty list")
    if not isinstance(prediction_values, (list, tuple)):
        raise ValueError("predicted_answer_ids must be a list")
    accepted_answers = {
        canonical_id(value, "accepted_answer_id") for value in answer_values
    }
    predicted_answers = {
        canonical_id(value, "predicted_answer_id") for value in prediction_values
    }
    prediction_correct = predicted_answers == accepted_answers

    status = model_output.get("status")
    if not isinstance(status, str) or status not in _MODEL_STATUSES:
        raise ValueError(f"model_output status must be one of {sorted(_MODEL_STATUSES)}")
    if status == "ok" and not prediction_values:
        raise ValueError("model_output with status 'ok' must contain predictions")
    if status != "ok" and prediction_values:
        raise ValueError("failed model_output statuses must not contain predictions")

    if not path_complete:
        failure_category = "retrieval_miss"
    elif status == "model_error":
        failure_category = "model_error"
    elif status == "ok" and prediction_correct:
        failure_category = "correct"
    else:
        failure_category = "reasoning_error"

    return {
        "qa_id": qa_id,
        "supporting_path_length": len(supporting_path),
        "covered_supporting_path_fact_count": len(covered_path),
        "covered_supporting_path_fact_ids": list(covered_path),
        "missing_supporting_path_fact_ids": list(missing_path),
        "supporting_path_coverage": len(covered_path) / len(supporting_path),
        "supporting_path_complete": path_complete,
        "prediction_correct": prediction_correct,
        "model_status": status,
        "failure_category": failure_category,
    }


def summarize_support_diagnostics(
    diagnostics: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Aggregate path coverage, exact accuracy, and error attribution."""
    if not isinstance(diagnostics, (list, tuple)) or not diagnostics:
        raise ValueError("diagnostics must be a non-empty list")

    total = len(diagnostics)
    path_edges = 0
    covered_path_edges = 0
    complete_paths = 0
    correct_predictions = 0
    failure_counts = {category: 0 for category in sorted(_FAILURE_CATEGORIES)}
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"diagnostic {index} must be a mapping")
        path_length = diagnostic.get("supporting_path_length")
        covered_path_length = diagnostic.get("covered_supporting_path_fact_count")
        path_complete = diagnostic.get("supporting_path_complete")
        category = diagnostic.get("failure_category")
        prediction_correct = diagnostic.get("prediction_correct")
        if (
            isinstance(path_length, bool)
            or not isinstance(path_length, int)
            or path_length < 1
        ):
            raise ValueError(f"diagnostic {index} supporting_path_length must be >= 1")
        if (
            isinstance(covered_path_length, bool)
            or not isinstance(covered_path_length, int)
            or not 0 <= covered_path_length <= path_length
        ):
            raise ValueError(
                f"diagnostic {index} covered supporting path count is invalid"
            )
        if not isinstance(path_complete, bool) or (
            path_complete != (covered_path_length == path_length)
        ):
            raise ValueError(
                f"diagnostic {index} supporting_path_complete is inconsistent"
            )
        if not isinstance(category, str) or category not in _FAILURE_CATEGORIES:
            raise ValueError(f"diagnostic {index} has an invalid failure_category")
        if not isinstance(prediction_correct, bool):
            raise ValueError(f"diagnostic {index} prediction_correct must be boolean")
        path_edges += path_length
        covered_path_edges += covered_path_length
        complete_paths += path_complete
        correct_predictions += prediction_correct
        failure_counts[category] += 1

    return {
        "example_count": total,
        "supporting_path_coverage": complete_paths / total,
        "supporting_path_edge_coverage": covered_path_edges / path_edges,
        "exact_accuracy": correct_predictions / total,
        "retrieval_miss_rate": failure_counts["retrieval_miss"] / total,
        "reasoning_error_rate": failure_counts["reasoning_error"] / total,
        "model_error_rate": failure_counts["model_error"] / total,
        "failure_counts": failure_counts,
    }


def _record_id(record: Mapping[str, Any], name: str) -> str:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return canonical_id(record["qa_id"], f"{name} qa_id")
    except KeyError as exc:
        raise ValueError(f"{name} is missing qa_id") from exc


def _non_empty_string(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
