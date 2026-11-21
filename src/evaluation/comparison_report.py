"""Diagnostic and efficiency summaries for the frozen learned comparison."""

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.encoder.node_embeddings import tokenize_question
from src.evaluation.metrics import compute_answer_metrics, normalize_answer_ids


def build_task45_report(
    comparison: Mapping[str, Any],
    qa_records: Sequence[Mapping[str, Any]],
    measured_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the Task 45 report from frozen predictions and measured evaluations."""
    if not isinstance(comparison, Mapping):
        raise ValueError("comparison must be a mapping")
    if not isinstance(qa_records, (list, tuple)) or not qa_records:
        raise ValueError("qa_records must be a non-empty list")
    if not isinstance(measured_runs, (list, tuple)) or not measured_runs:
        raise ValueError("measured_runs must be a non-empty list")

    run_reports: list[dict[str, Any]] = []
    for run in measured_runs:
        if not isinstance(run, Mapping):
            raise ValueError("every measured run must be a mapping")
        run_reports.append(
            _build_run_report(
                qa_records,
                run.get("predictions"),
                run.get("measurement"),
                seed=run.get("seed"),
                variant=run.get("variant"),
            )
        )
    return {
        "schema_version": 1,
        "benchmark": comparison.get("benchmark"),
        "dataset_id": comparison.get("dataset_id"),
        "dataset_revision": comparison.get("dataset_revision"),
        "source_code_revision": comparison.get("code_revision"),
        "source_tree_sha256": comparison.get("source_tree_sha256"),
        "input_sha256": comparison.get("input_sha256"),
        "test_split": comparison.get("test_split", "test"),
        "test_summary": comparison.get("test_summary"),
        "input_question_tokens": _question_token_statistics(qa_records),
        "run_reports": run_reports,
        "efficiency_summary": _efficiency_summary(run_reports),
    }


def _build_run_report(
    qa_records: Sequence[Mapping[str, Any]],
    predictions: Any,
    measurement: Any,
    *,
    seed: Any,
    variant: Any,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("run seed must be a non-negative integer")
    if variant not in {"baseline", "selr"}:
        raise ValueError("run variant must be baseline or selr")
    if not isinstance(measurement, Mapping):
        raise ValueError(f"{variant} seed {seed} measurement must be a mapping")
    metrics = compute_answer_metrics(qa_records, predictions)
    labels_by_id = {str(record["qa_id"]).strip(): record for record in qa_records}
    predictions_by_id = {
        str(record["qa_id"]).strip(): record for record in predictions
    }
    if set(labels_by_id) != set(predictions_by_id):
        raise ValueError("QA and prediction IDs must match exactly")

    hop_groups: dict[str, list[str]] = defaultdict(list)
    temporal_groups: dict[str, list[str]] = defaultdict(list)
    for qa_id, record in labels_by_id.items():
        relation_sequence = record.get("relation_sequence")
        if not isinstance(relation_sequence, (list, tuple)) or not relation_sequence:
            raise ValueError(f"QA {qa_id!r} must have a non-empty relation_sequence")
        temporal_constraint = record.get("temporal_constraint")
        if not isinstance(temporal_constraint, Mapping):
            raise ValueError(f"QA {qa_id!r} must have a temporal_constraint mapping")
        operator = temporal_constraint.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError(f"QA {qa_id!r} temporal operator must be non-empty")
        hop_groups[str(len(relation_sequence))].append(qa_id)
        temporal_groups[operator.strip()].append(qa_id)

    return {
        "seed": seed,
        "variant": variant,
        "accuracy": metrics,
        "hop_count_breakdown": _group_metrics(
            hop_groups, labels_by_id, predictions_by_id
        ),
        "temporal_operator_breakdown": _group_metrics(
            temporal_groups, labels_by_id, predictions_by_id
        ),
        "retrieval_coverage": _retrieval_coverage(
            labels_by_id, predictions_by_id
        ),
        "efficiency": dict(measurement),
    }


def _group_metrics(
    groups: Mapping[str, Sequence[str]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        group: compute_answer_metrics(
            [labels_by_id[qa_id] for qa_id in qa_ids],
            [predictions_by_id[qa_id] for qa_id in qa_ids],
        )
        for group, qa_ids in sorted(groups.items())
    }


def _retrieval_coverage(
    labels_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total_gold_answers = covered_gold_answers = 0
    examples_with_candidates = examples_with_any_gold = examples_with_all_gold = 0
    total_candidates = 0
    for qa_id, record in labels_by_id.items():
        prediction = predictions_by_id[qa_id]
        scores = prediction.get("answer_scores")
        if not isinstance(scores, (list, tuple)):
            raise ValueError(f"prediction for QA {qa_id!r} must contain answer_scores")
        candidate_ids: list[Any] = []
        for index, row in enumerate(scores):
            if not isinstance(row, Mapping) or "answer_id" not in row:
                raise ValueError(
                    f"prediction for QA {qa_id!r} answer_scores[{index}] "
                    "must contain answer_id"
                )
            candidate_ids.append(row["answer_id"])
        candidates = set(
            normalize_answer_ids(
                candidate_ids,
                field=f"prediction for QA {qa_id!r} candidate answer_ids",
                allow_empty=True,
            )
        )
        gold = set(
            normalize_answer_ids(
                record.get("accepted_answer_ids"),
                field=f"QA {qa_id!r} accepted_answer_ids",
            )
        )
        covered = gold & candidates
        total_gold_answers += len(gold)
        covered_gold_answers += len(covered)
        total_candidates += len(candidates)
        examples_with_candidates += bool(candidates)
        examples_with_any_gold += bool(covered)
        examples_with_all_gold += covered == gold

    example_count = len(labels_by_id)
    return {
        "example_count": example_count,
        "examples_with_candidates": examples_with_candidates,
        "candidate_coverage_rate": examples_with_candidates / example_count,
        "examples_with_any_gold_answer_retrieved": examples_with_any_gold,
        "any_gold_answer_coverage_rate": examples_with_any_gold / example_count,
        "examples_with_all_gold_answers_retrieved": examples_with_all_gold,
        "all_gold_answers_coverage_rate": examples_with_all_gold / example_count,
        "total_gold_answers": total_gold_answers,
        "covered_gold_answers": covered_gold_answers,
        "gold_answer_recall": (
            covered_gold_answers / total_gold_answers if total_gold_answers else 0.0
        ),
        "total_retrieved_candidates": total_candidates,
        "mean_candidates_per_example": total_candidates / example_count,
    }


def _question_token_statistics(
    qa_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts: list[int] = []
    for index, record in enumerate(qa_records):
        question = record.get("question")
        if not isinstance(question, str):
            raise ValueError(f"QA record {index} question must be a string")
        counts.append(len(tokenize_question(question)))
    return {
        "tokenizer": (
            "src.encoder.node_embeddings.tokenize_question: case-folded regex "
            "word, number, and punctuation tokens"
        ),
        "question_count": len(counts),
        "total_question_input_tokens": sum(counts),
        "mean_tokens_per_question": statistics.mean(counts),
        "max_tokens_per_question": max(counts),
        "encoder_tokenizer_used": True,
        "generated_tokens": "not applicable; answer decoders emit entity IDs",
    }


def _efficiency_summary(run_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in ("baseline", "selr"):
        runs = [run for run in run_reports if run["variant"] == variant]
        if not runs:
            continue
        latency_fields = (
            "evaluation_wall_seconds",
            "amortized_ms_per_example",
            "examples_per_second",
        )
        memory_fields = (
            "python_heap_peak_bytes",
            "model_parameter_bytes",
            "optimizer_state_tensor_bytes",
            "max_batch_tensor_storage_bytes",
        )
        summary[variant] = {
            "latency": {
                field: _mean_and_std(
                    [float(run["efficiency"]["latency"][field]) for run in runs]
                )
                for field in latency_fields
            },
            "memory": {
                field: _mean_and_std(
                    [int(run["efficiency"]["memory"][field]) for run in runs]
                )
                for field in memory_fields
            },
        }
    return summary


def _mean_and_std(values: Sequence[float | int]) -> dict[str, Any]:
    return {
        "per_seed": list(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
