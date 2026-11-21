"""Set-based answer normalization and evaluation metrics."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.graph.graph_utils import canonical_id


_STATUSES = ("ok", "retrieval_miss", "no_candidates", "model_error")


def normalize_answer_ids(
    answer_ids: Sequence[Any],
    *,
    field: str = "answer_ids",
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Canonicalize IDs, remove duplicates, and return a stable sorted tuple."""
    if not isinstance(answer_ids, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    normalized = tuple(
        sorted({canonical_id(answer_id, field) for answer_id in answer_ids})
    )
    if not normalized and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    return normalized


def compute_answer_metrics(
    qa_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate exact set match and answer-level precision/recall/F1.

    Every accepted answer is treated as correct. Failed outputs are retained as
    empty predictions and therefore count as incorrect rather than disappearing
    from the denominator.
    """
    if not isinstance(qa_records, (list, tuple)) or not qa_records:
        raise ValueError("qa_records must be a non-empty list")
    if not isinstance(prediction_records, (list, tuple)) or not prediction_records:
        raise ValueError("prediction_records must be a non-empty list")

    labels_by_id: dict[str, tuple[str, ...]] = {}
    for index, record in enumerate(qa_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"QA record {index} must be a mapping")
        qa_id = _record_id(record, f"QA record {index}")
        if qa_id in labels_by_id:
            raise ValueError(f"duplicate qa_id {qa_id!r} in QA records")
        labels_by_id[qa_id] = normalize_answer_ids(
            record.get("accepted_answer_ids"),
            field=f"QA record {index} accepted_answer_ids",
        )

    predictions_by_id: dict[str, tuple[str, ...]] = {}
    status_counts = Counter({status: 0 for status in _STATUSES})
    for index, record in enumerate(prediction_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"prediction record {index} must be a mapping")
        qa_id = _record_id(record, f"prediction record {index}")
        if qa_id in predictions_by_id:
            raise ValueError(f"duplicate qa_id {qa_id!r} in predictions")
        status = record.get("status")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError(
                f"prediction record {index} status must be one of {list(_STATUSES)}"
            )
        predicted_ids = normalize_answer_ids(
            record.get("predicted_answer_ids"),
            field=f"prediction record {index} predicted_answer_ids",
            allow_empty=True,
        )
        if status == "ok" and not predicted_ids:
            raise ValueError(
                f"prediction record {index} with status 'ok' must contain predictions"
            )
        if status != "ok" and predicted_ids:
            raise ValueError(
                f"failed prediction record {index} must not contain predictions"
            )
        predictions_by_id[qa_id] = predicted_ids
        status_counts[status] += 1

    missing = sorted(set(labels_by_id) - set(predictions_by_id))
    extra = sorted(set(predictions_by_id) - set(labels_by_id))
    if missing or extra:
        raise ValueError(
            f"QA/prediction IDs must match exactly; missing={missing!r}, extra={extra!r}"
        )

    true_positives = false_positives = false_negatives = 0
    exact_matches = 0
    per_example_precision: list[float] = []
    per_example_recall: list[float] = []
    per_example_f1: list[float] = []
    total_gold = total_predicted = 0
    for qa_id, accepted in labels_by_id.items():
        predicted = predictions_by_id[qa_id]
        accepted_set, predicted_set = set(accepted), set(predicted)
        true_positive = len(accepted_set & predicted_set)
        false_positive = len(predicted_set - accepted_set)
        false_negative = len(accepted_set - predicted_set)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        true_positives += true_positive
        false_positives += false_positive
        false_negatives += false_negative
        total_gold += len(accepted_set)
        total_predicted += len(predicted_set)
        exact_matches += predicted_set == accepted_set
        per_example_precision.append(precision)
        per_example_recall.append(recall)
        per_example_f1.append(f1)

    count = len(labels_by_id)
    micro_precision = _ratio(true_positives, true_positives + false_positives)
    micro_recall = _ratio(true_positives, true_positives + false_negatives)
    micro_f1 = _ratio(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )
    return {
        "example_count": count,
        "exact_match_accuracy": exact_matches / count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": sum(per_example_precision) / count,
        "macro_recall": sum(per_example_recall) / count,
        "macro_f1": sum(per_example_f1) / count,
        "total_gold_answers": total_gold,
        "total_predicted_answers": total_predicted,
        "status_counts": {status: status_counts[status] for status in _STATUSES},
    }


def _record_id(record: Mapping[str, Any], name: str) -> str:
    try:
        return canonical_id(record["qa_id"], f"{name} qa_id")
    except KeyError as exc:
        raise ValueError(f"{name} is missing qa_id") from exc


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
