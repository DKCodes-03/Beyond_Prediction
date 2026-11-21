"""Create deterministic, leakage-checked splits for prepared temporal QA."""

import json
import math
import random
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any


_SPLIT_NAMES = ("train", "validation", "test")
_OUTPUT_FILENAMES = {
    "train": "train.json",
    "validation": "val.json",
    "test": "test.json",
}
_DEFAULT_FRACTIONS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


def split_qa_records(
    records: Iterable[Mapping[str, Any]],
    fractions: Mapping[str, float] | None = None,
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    """Assign prepared QA records to deterministic, path-disjoint splits.

    Duplicate QA IDs are rejected, and records with the same ordered witness
    path are kept in one group. Every query must declare its inclusive
    historical cutoff; the split only partitions QA labels and does not
    partition or filter evidence facts.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    split_fractions = _validate_fractions(fractions)

    normalized_records: list[dict[str, Any]] = []
    seen_qa_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"QA record {index}: expected a mapping")
        qa_id = record.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id.strip():
            raise ValueError(f"QA record {index}: qa_id must be a non-empty string")
        qa_id = qa_id.strip()
        if qa_id in seen_qa_ids:
            raise ValueError(f"duplicate qa_id {qa_id!r}")
        seen_qa_ids.add(qa_id)
        if "split" in record:
            raise ValueError(f"QA record {qa_id!r} is already assigned to a split")
        _validate_temporal_constraint(record, qa_id)
        path_signature = _path_signature(record, qa_id)
        normalized_records.append(
            {
                "qa_id": qa_id,
                "path_signature": path_signature,
                "record": dict(record),
            }
        )

    if len(normalized_records) < len(_SPLIT_NAMES):
        raise ValueError("at least three QA records are required for non-empty splits")

    groups_by_signature: dict[tuple[str, ...], list[int]] = {}
    for index, record in enumerate(normalized_records):
        groups_by_signature.setdefault(record["path_signature"], []).append(index)
    groups = list(groups_by_signature.values())
    if len(groups) < len(_SPLIT_NAMES):
        raise ValueError(
            "at least three distinct supporting-path groups are required "
            "for non-empty splits"
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_sizes = {
        split: len(normalized_records) * split_fractions[split]
        for split in _SPLIT_NAMES
    }
    assignments: dict[str, list[int]] = {split: [] for split in _SPLIT_NAMES}
    assigned_counts = {split: 0 for split in _SPLIT_NAMES}

    # Reserve one group per split so even small datasets produce all splits.
    initial_groups = groups[: len(_SPLIT_NAMES)]
    for split, group in zip(
        sorted(_SPLIT_NAMES, key=lambda name: split_fractions[name], reverse=True),
        initial_groups,
    ):
        assignments[split].extend(group)
        assigned_counts[split] += len(group)

    for group in groups[len(_SPLIT_NAMES) :]:
        scores = {
            split: sum(
                (assigned_counts[name] + (len(group) if name == split else 0)
                 - target_sizes[name]) ** 2
                / target_sizes[name]
                for name in _SPLIT_NAMES
            )
            for split in _SPLIT_NAMES
        }
        best_score = min(scores.values())
        best_splits = [
            split for split in _SPLIT_NAMES if math.isclose(scores[split], best_score)
        ]
        chosen_split = rng.choice(best_splits)
        assignments[chosen_split].extend(group)
        assigned_counts[chosen_split] += len(group)

    result: dict[str, list[dict[str, Any]]] = {}
    for split in _SPLIT_NAMES:
        result[split] = [
            {**normalized_records[index]["record"], "split": split}
            for index in sorted(assignments[split])
        ]
    return result


def write_qa_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    output_dir: str | Path,
) -> None:
    """Write all three assigned splits as UTF-8 JSON arrays."""
    if set(splits) != set(_SPLIT_NAMES):
        raise ValueError("splits must contain exactly train, validation, and test")

    serialized: dict[str, str] = {}
    for split in _SPLIT_NAMES:
        records = list(splits[split])
        for index, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("split") != split:
                raise ValueError(
                    f"{split} record {index}: record split must be {split!r}"
                )
        serialized[split] = json.dumps(records, ensure_ascii=False, indent=2) + "\n"

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split, contents in serialized.items():
        (destination / _OUTPUT_FILENAMES[split]).write_text(
            contents, encoding="utf-8"
        )


def _validate_fractions(
    fractions: Mapping[str, float] | None,
) -> dict[str, float]:
    supplied = _DEFAULT_FRACTIONS if fractions is None else fractions
    if set(supplied) != set(_SPLIT_NAMES):
        raise ValueError("fractions must contain exactly train, validation, and test")
    normalized: dict[str, float] = {}
    for split in _SPLIT_NAMES:
        value = supplied[split]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{split} fraction must be a finite number")
        fraction = float(value)
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError(f"{split} fraction must be between 0 and 1")
        normalized[split] = fraction
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    return normalized


def _validate_temporal_constraint(record: Mapping[str, Any], qa_id: str) -> None:
    constraint = record.get("temporal_constraint")
    if not isinstance(constraint, Mapping):
        raise ValueError(f"QA record {qa_id!r}: temporal_constraint must be a mapping")
    if constraint.get("operator") != "on_or_before":
        raise ValueError(
            f"QA record {qa_id!r}: temporal operator must be 'on_or_before'"
        )
    cutoff = constraint.get("cutoff_date")
    if not isinstance(cutoff, str):
        raise ValueError(f"QA record {qa_id!r}: cutoff_date must be an ISO date")
    try:
        parsed_cutoff = date.fromisoformat(cutoff)
    except ValueError as exc:
        raise ValueError(
            f"QA record {qa_id!r}: cutoff_date must be an ISO date"
        ) from exc
    if parsed_cutoff.isoformat() != cutoff:
        raise ValueError(
            f"QA record {qa_id!r}: cutoff_date must use YYYY-MM-DD format"
        )


def _path_signature(record: Mapping[str, Any], qa_id: str) -> tuple[str, ...]:
    path = record.get("supporting_path_fact_ids")
    if (
        not isinstance(path, list)
        or not path
        or any(not isinstance(fact_id, str) or not fact_id.strip() for fact_id in path)
    ):
        raise ValueError(
            f"QA record {qa_id!r}: supporting_path_fact_ids must be a "
            "non-empty list of fact IDs"
        )
    return tuple(fact_id.strip() for fact_id in path)
