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


def create_disjoint_benchmark_splits(
    records: Iterable[Mapping[str, Any]],
    fractions: Mapping[str, float] | None = None,
    seed: int = 17,
    *,
    excluded_qa_records: Iterable[Mapping[str, Any]] | None = None,
    excluded_qa_ids: Iterable[str] | None = None,
    pilot_test_records: Iterable[Mapping[str, Any]] | None = None,
    excluded_records: Iterable[Mapping[str, Any]] | None = None,
    pilot_test_qa_records: Iterable[Mapping[str, Any]] | None = None,
    exclude_qa_ids: Iterable[str] | None = None,
    split_seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Create benchmark splits after removing pilot-overlapping instances.

    Overlap is checked by QA ID, witness-path facts, and the complete query
    signature. The remaining records are assigned with the shared path-disjoint
    split implementation, so no supporting-path instance is shared between
    train, validation, and test.
    """
    if split_seed is not None:
        seed = split_seed
    normalized = list(records)
    excluded_sources: list[Iterable[Mapping[str, Any]]] = []
    if excluded_qa_records is not None:
        excluded_sources.append(excluded_qa_records)
    if excluded_records is not None:
        excluded_sources.append(excluded_records)
    if pilot_test_records is not None:
        excluded_sources.append(pilot_test_records)
    if pilot_test_qa_records is not None:
        excluded_sources.append(pilot_test_qa_records)

    excluded_keys: set[tuple[Any, ...]] = set()
    excluded_ids: set[str] = set(excluded_qa_ids or [])
    excluded_ids.update(exclude_qa_ids or [])
    for source in excluded_sources:
        for record in source:
            if not isinstance(record, Mapping):
                raise ValueError("excluded QA records must be mappings")
            qa_id = record.get("qa_id")
            if isinstance(qa_id, str):
                excluded_ids.add(qa_id)
            excluded_keys.update(_overlap_keys(record))

    eligible: list[dict[str, Any]] = []
    for index, record in enumerate(normalized):
        if not isinstance(record, Mapping):
            raise ValueError(f"QA record {index}: expected a mapping")
        qa_id = record.get("qa_id")
        if isinstance(qa_id, str) and qa_id in excluded_ids:
            continue
        if _overlap_keys(record).intersection(excluded_keys):
            continue
        eligible_record = dict(record)
        eligible_record.pop("split", None)
        eligible.append(eligible_record)

    splits = split_qa_records(eligible, fractions=fractions, seed=seed)
    _verify_benchmark_split_disjointness(splits)
    return splits


def prepare_disjoint_benchmark_splits(
    records: Iterable[Mapping[str, Any]],
    fractions: Mapping[str, float] | None = None,
    seed: int = 17,
    **kwargs: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Descriptive alias for :func:`create_disjoint_benchmark_splits`."""
    return create_disjoint_benchmark_splits(
        records,
        fractions=fractions,
        seed=seed,
        **kwargs,
    )


def run_benchmark_split_preparation(
    *,
    qa_records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    fractions: Mapping[str, float] | None = None,
    seed: int = 17,
    excluded_qa_records: Iterable[Mapping[str, Any]] | None = None,
    excluded_qa_ids: Iterable[str] | None = None,
    pilot_test_records: Iterable[Mapping[str, Any]] | None = None,
    excluded_records: Iterable[Mapping[str, Any]] | None = None,
    pilot_test_qa_records: Iterable[Mapping[str, Any]] | None = None,
    exclude_qa_ids: Iterable[str] | None = None,
    split_seed: int | None = None,
    benchmark: str = "hard_synthetic",
) -> dict[str, Any]:
    """Filter pilot overlap, create three splits, and export them atomically."""
    if split_seed is not None:
        seed = split_seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    all_records = list(qa_records)
    splits = create_disjoint_benchmark_splits(
        all_records,
        fractions=fractions,
        seed=seed,
        excluded_qa_records=excluded_qa_records,
        excluded_qa_ids=excluded_qa_ids,
        pilot_test_records=pilot_test_records,
        excluded_records=excluded_records,
        pilot_test_qa_records=pilot_test_qa_records,
        exclude_qa_ids=exclude_qa_ids,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_qa_splits(splits, destination)

    manifest = {
        "benchmark": benchmark,
        "status": "prepared",
        "seed": seed,
        "fractions": dict(fractions or _DEFAULT_FRACTIONS),
        "input_qa_count": len(all_records),
        "split_counts": {
            split: len(records)
            for split, records in splits.items()
        },
        "excluded_qa_count": len(all_records) - sum(
            len(records) for records in splits.values()
        ),
        "disjoint": True,
        "test_reserved_for_final_evaluation": True,
        "pilot_test_overlap_excluded": True,
        "artifacts": {
            "train": "train.json",
            "validation": "val.json",
            "test": "test.json",
        },
    }
    manifest_path = destination / "benchmark_splits_manifest.json"
    temporary = manifest_path.with_name(".benchmark_splits_manifest.json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "benchmark": benchmark,
        "status": "prepared",
        "split_counts": manifest["split_counts"],
        "disjoint": True,
    }


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


def _overlap_keys(record: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    """Return exact-instance signatures used to reject benchmark overlap."""
    keys: set[tuple[Any, ...]] = set()
    qa_id = record.get("qa_id")
    if isinstance(qa_id, str) and qa_id:
        keys.add(("qa_id", qa_id))

    path = record.get("supporting_path_fact_ids")
    if isinstance(path, list) and path:
        keys.add(("path", tuple(path)))

    constraint = record.get("temporal_constraint")
    cutoff = constraint.get("cutoff_date") if isinstance(constraint, Mapping) else None
    seeds = record.get("seed_entity_ids")
    relations = record.get("relation_sequence")
    answers = record.get("accepted_answer_ids")
    if (
        isinstance(seeds, list)
        and isinstance(relations, list)
        and isinstance(answers, list)
        and isinstance(cutoff, str)
    ):
        keys.add(
            (
                "query",
                tuple(seeds),
                tuple(relations),
                cutoff,
                tuple(sorted(answers)),
            )
        )
    return keys


def _verify_benchmark_split_disjointness(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    qa_ids: dict[str, set[str]] = {}
    path_signatures: dict[str, set[tuple[str, ...]]] = {}
    query_signatures: dict[str, set[tuple[Any, ...]]] = {}

    for split in _SPLIT_NAMES:
        qa_ids[split] = set()
        path_signatures[split] = set()
        query_signatures[split] = set()
        for record in splits[split]:
            qa_id = record["qa_id"]
            qa_ids[split].add(qa_id)
            path_signatures[split].add(_path_signature(record, qa_id))
            query_signatures[split].update(_overlap_keys(record))

    for left_index, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[left_index + 1:]:
            if qa_ids[left].intersection(qa_ids[right]):
                raise ValueError("train/validation/test QA IDs must be disjoint")
            if path_signatures[left].intersection(path_signatures[right]):
                raise ValueError("train/validation/test witness paths must be disjoint")
            if query_signatures[left].intersection(query_signatures[right]):
                raise ValueError("train/validation/test query instances must be disjoint")


split_benchmark_qa_records = create_disjoint_benchmark_splits
create_benchmark_splits = create_disjoint_benchmark_splits
split_new_benchmark = create_disjoint_benchmark_splits
create_disjoint_splits = create_disjoint_benchmark_splits
prepare_new_benchmark_splits = create_disjoint_benchmark_splits
