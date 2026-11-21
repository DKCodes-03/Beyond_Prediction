"""Persist canonical facts and split assignments with reproducibility metadata."""

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.graph.graph_utils import fact_signature, validate_fact


_SPLIT_NAMES = ("train", "validation", "test")
_SPLIT_FILENAMES = {
    "train": "train.json",
    "validation": "val.json",
    "test": "test.json",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def save_processed_artifacts(
    facts: Iterable[Mapping[str, Any]],
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    processed_dir: str | Path,
    splits_dir: str | Path,
    *,
    dataset_id: str,
    dataset_source: str,
    dataset_revision: str | None,
    source_checksum_sha256: str | None,
    split_seed: int,
    split_fractions: Mapping[str, float],
) -> Path:
    """Write facts, split arrays, and a manifest describing their provenance.

    The manifest is written last and contains hashes for every data file.
    Existing ``facts.json``, ``train.json``, ``val.json``, and ``test.json``
    formats remain plain JSON arrays; the manifest adds their schema version
    and split-generation provenance without changing those file contracts.
    """
    normalized_facts = _validate_facts(facts)
    normalized_splits = _validate_splits(splits)
    fractions = _validate_fractions(split_fractions)
    _validate_dataset_metadata(
        dataset_id, dataset_source, dataset_revision, source_checksum_sha256
    )
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")

    processed_path = Path(processed_dir)
    splits_path = Path(splits_dir)
    facts_text = _json_text(normalized_facts)
    split_texts = {
        split: _json_text(normalized_splits[split]) for split in _SPLIT_NAMES
    }

    facts_path = processed_path / "facts.json"
    split_paths = {
        split: splits_path / _SPLIT_FILENAMES[split] for split in _SPLIT_NAMES
    }
    _atomic_write(facts_path, facts_text)
    for split in _SPLIT_NAMES:
        _atomic_write(split_paths[split], split_texts[split])

    manifest = {
        "manifest_schema": "processed_artifacts.schema.json",
        "schema_version": 1,
        "dataset": {
            "id": dataset_id.strip(),
            "source": dataset_source.strip(),
            "revision": dataset_revision.strip() if dataset_revision else None,
            "source_checksum_sha256": source_checksum_sha256,
        },
        "record_schemas": {
            "facts": "data_contracts.schema.json#/$defs/fact",
            "qa": "data_contracts.schema.json#/$defs/qa_record",
        },
        "artifacts": {
            "facts": {
                "path": facts_path.name,
                "record_count": len(normalized_facts),
                "sha256": _sha256(facts_text),
            },
            "splits": {
                split: {
                    "path": _relative_posix_path(split_paths[split], processed_path),
                    "record_count": len(normalized_splits[split]),
                    "sha256": _sha256(split_texts[split]),
                    "qa_ids_sha256": _sha256(
                        _json_text(
                            sorted(
                                record["qa_id"]
                                for record in normalized_splits[split]
                            )
                        )
                    ),
                }
                for split in _SPLIT_NAMES
            },
        },
        "split_provenance": {
            "strategy": "deterministic_grouped_random",
            "group_by": "query_path_signature",
            "seed": split_seed,
            "fractions": fractions,
            "evidence_policy": "closed_knowledge_base",
            "label_usage": {
                "train": "fit",
                "validation": "model_selection",
                "test": "final_evaluation_only",
            },
        },
    }
    manifest_path = processed_path / "manifest.json"
    _atomic_write(manifest_path, _json_text(manifest))
    return manifest_path


def _validate_facts(
    facts: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_signatures: set[tuple[str, str, str, str]] = set()
    for index, record in enumerate(facts):
        fact = validate_fact(record, index)
        signature = fact_signature(fact)
        if fact["fact_id"] in seen_ids:
            raise ValueError(f"fact {index}: duplicate fact_id {fact['fact_id']!r}")
        if signature in seen_signatures:
            raise ValueError(f"fact {index}: duplicate event signature {signature!r}")
        seen_ids.add(fact["fact_id"])
        seen_signatures.add(signature)
        normalized.append(fact)
    if not normalized:
        raise ValueError("cannot save an empty fact collection")
    return sorted(
        normalized,
        key=lambda fact: (
            fact["timestamp"],
            fact["subject_id"],
            fact["relation_id"],
            fact["object_id"],
            fact["fact_id"],
        ),
    )


def _validate_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    if set(splits) != set(_SPLIT_NAMES):
        raise ValueError("splits must contain exactly train, validation, and test")

    normalized: dict[str, list[dict[str, Any]]] = {}
    seen_qa_ids: set[str] = set()
    seen_paths: dict[tuple[str, ...], str] = {}
    for split in _SPLIT_NAMES:
        records: list[dict[str, Any]] = []
        for index, record in enumerate(splits[split]):
            if not isinstance(record, Mapping):
                raise ValueError(f"{split} record {index}: expected a mapping")
            qa_id = record.get("qa_id")
            if not isinstance(qa_id, str) or not qa_id.strip():
                raise ValueError(
                    f"{split} record {index}: qa_id must be a non-empty string"
                )
            normalized_qa_id = qa_id.strip()
            if normalized_qa_id in seen_qa_ids:
                raise ValueError(f"duplicate qa_id {qa_id!r}")
            if record.get("split") != split:
                raise ValueError(
                    f"{split} record {index}: record split must be {split!r}"
                )
            path = record.get("supporting_path_fact_ids")
            if (
                not isinstance(path, list)
                or not path
                or any(not isinstance(fact_id, str) or not fact_id.strip() for fact_id in path)
            ):
                raise ValueError(
                    f"{split} record {index}: supporting_path_fact_ids must be "
                    "a non-empty list of fact IDs"
                )
            signature = tuple(fact_id.strip() for fact_id in path)
            previous_split = seen_paths.get(signature)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    "supporting path is assigned to multiple splits: "
                    f"{signature!r}"
                )
            seen_paths[signature] = split
            seen_qa_ids.add(normalized_qa_id)
            records.append(dict(record))
        if not records:
            raise ValueError(f"{split} split must not be empty")
        normalized[split] = records
    return normalized


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if set(fractions) != set(_SPLIT_NAMES):
        raise ValueError("split_fractions must contain train, validation, and test")
    normalized: dict[str, float] = {}
    for split in _SPLIT_NAMES:
        value = fractions[split]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{split} fraction must be a finite number")
        fraction = float(value)
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError(f"{split} fraction must be between 0 and 1")
        normalized[split] = fraction
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    return normalized


def _validate_dataset_metadata(
    dataset_id: str,
    dataset_source: str,
    dataset_revision: str | None,
    source_checksum_sha256: str | None,
) -> None:
    for field, value in (("dataset_id", dataset_id), ("dataset_source", dataset_source)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if dataset_revision is not None and (
        not isinstance(dataset_revision, str) or not dataset_revision.strip()
    ):
        raise ValueError("dataset_revision must be a non-empty string or None")
    if source_checksum_sha256 is not None and (
        not isinstance(source_checksum_sha256, str)
        or not _SHA256_PATTERN.fullmatch(source_checksum_sha256)
    ):
        raise ValueError("source_checksum_sha256 must be a 64-character hex digest")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative_posix_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, start=base)).as_posix()


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
