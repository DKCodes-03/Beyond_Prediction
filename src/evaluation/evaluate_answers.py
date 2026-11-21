"""Validate and atomically export answer predictions and evaluation summaries."""

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.metrics import normalize_answer_ids
from src.graph.graph_utils import canonical_id


_STATUSES = frozenset({"ok", "retrieval_miss", "no_candidates", "model_error"})
_OUTPUT_FIELDS = frozenset(
    {
        "qa_id",
        "model_variant",
        "status",
        "predicted_answer_ids",
        "answer_scores",
        "failure_reason",
        "latent_states",
    }
)


def export_predictions(
    prediction_records: Sequence[Mapping[str, Any]],
    path: str | os.PathLike[str],
) -> Path:
    """Write schema-shaped prediction records as a deterministic JSON array."""
    normalized_records = _normalize_predictions(prediction_records)
    destination = Path(path)
    _atomic_write(
        destination,
        json.dumps(
            normalized_records,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return destination


def write_evaluation_results(
    output_dir: str | os.PathLike[str],
    prediction_records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    run_info: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    """Export predictions, metrics, and run configuration to one directory."""
    directory = Path(output_dir)
    predictions_path = export_predictions(
        prediction_records, directory / "predictions.json"
    )
    metrics_path = directory / "metrics.json"
    run_path = directory / "run.json"
    _atomic_write(
        metrics_path,
        json.dumps(
            dict(metrics),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_write(
        run_path,
        json.dumps(
            dict(run_info),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return predictions_path, metrics_path, run_path


def _normalize_predictions(
    prediction_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(prediction_records, (list, tuple)) or not prediction_records:
        raise ValueError("prediction_records must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen_qa_ids: set[str] = set()
    for index, record in enumerate(prediction_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"prediction record {index} must be a mapping")
        unknown_fields = set(record) - _OUTPUT_FIELDS
        if unknown_fields:
            raise ValueError(
                f"prediction record {index} contains unsupported fields: "
                f"{sorted(unknown_fields)!r}"
            )
        qa_id = canonical_id(record.get("qa_id"), f"prediction record {index} qa_id")
        if qa_id in seen_qa_ids:
            raise ValueError(f"duplicate qa_id {qa_id!r} in predictions")
        seen_qa_ids.add(qa_id)
        model_variant = record.get("model_variant")
        if not isinstance(model_variant, str) or not model_variant.strip():
            raise ValueError(
                f"prediction record {index} model_variant must be non-empty"
            )
        status = record.get("status")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError(
                f"prediction record {index} status must be one of "
                f"{sorted(_STATUSES)}"
            )
        answer_ids = normalize_answer_ids(
            record.get("predicted_answer_ids"),
            field=f"prediction record {index} predicted_answer_ids",
            allow_empty=True,
        )
        if status == "ok" and not answer_ids:
            raise ValueError(
                f"prediction record {index} with status 'ok' must contain predictions"
            )
        if status != "ok" and answer_ids:
            raise ValueError(
                f"failed prediction record {index} must not contain predictions"
            )
        output: dict[str, Any] = {
            "qa_id": qa_id,
            "model_variant": model_variant.strip(),
            "status": status,
            "predicted_answer_ids": list(answer_ids),
            "answer_scores": _normalize_scores(
                record.get("answer_scores"), index
            ),
        }
        if status != "ok":
            failure_reason = record.get("failure_reason")
            if not isinstance(failure_reason, str) or not failure_reason.strip():
                raise ValueError(
                    f"failed prediction record {index} must include failure_reason"
                )
            output["failure_reason"] = failure_reason.strip()
        elif "failure_reason" in record:
            raise ValueError(
                f"successful prediction record {index} must not include failure_reason"
            )
        if "latent_states" in record:
            output["latent_states"] = _normalize_latent_states(
                record["latent_states"], index
            )
        normalized.append(output)
    return normalized


def _normalize_scores(values: Any, index: int) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"prediction record {index} answer_scores must be a list")
    normalized: list[dict[str, Any]] = []
    seen_answer_ids: set[str] = set()
    for score_index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"prediction record {index} answer_scores[{score_index}] must be a mapping"
            )
        if set(row) != {"answer_id", "score"}:
            raise ValueError(
                f"prediction record {index} answer_scores[{score_index}] must "
                "contain exactly answer_id and score"
            )
        answer_id = canonical_id(
            row.get("answer_id"),
            f"prediction record {index} answer_scores[{score_index}] answer_id",
        )
        score = row.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"prediction record {index} answer_scores[{score_index}] "
                "score must be finite"
            )
        if answer_id in seen_answer_ids:
            raise ValueError(
                f"prediction record {index} has duplicate answer score for "
                f"{answer_id!r}"
            )
        seen_answer_ids.add(answer_id)
        normalized.append({"answer_id": answer_id, "score": float(score)})
    return normalized


def _normalize_latent_states(values: Any, index: int) -> list[list[float]]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"prediction record {index} latent_states must be a list")
    normalized: list[list[float]] = []
    for state_index, state in enumerate(values):
        if not isinstance(state, (list, tuple)):
            raise ValueError(
                f"prediction record {index} latent_states[{state_index}] must be a list"
            )
        vector: list[float] = []
        for value_index, value in enumerate(state):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"prediction record {index} latent_states[{state_index}]"
                    f"[{value_index}] must be finite"
                )
            vector.append(float(value))
        normalized.append(vector)
    return normalized


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
