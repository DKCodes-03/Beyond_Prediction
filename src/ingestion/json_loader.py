"""Load temporal facts from the ICEWS14 TFLEX JSONL release."""

import json
import zipfile
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


_ICEWS14_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ICEWS14Metadata:
    entities: dict[int, str]
    relations: dict[int, str]
    timestamps: dict[int, str]


def load_icews14_metadata(metadata_path: str | Path) -> ICEWS14Metadata:
    """Load the integer-ID vocabularies used by the TFLEX release."""
    path = Path(metadata_path)
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")

    vocabularies = {
        "entities": _load_vocabulary(metadata, "entity2idx", path),
        "relations": _load_vocabulary(metadata, "relation2idx", path),
        "timestamps": _load_vocabulary(metadata, "timestamp2idx", path),
    }
    count_keys = {
        "entities": "entity_count",
        "relations": "relation_count",
        "timestamps": "timestamp_count",
    }
    for vocabulary_name, count_key in count_keys.items():
        declared_count = metadata.get(count_key)
        if declared_count != len(vocabularies[vocabulary_name]):
            raise ValueError(
                f"{path}: {count_key} is {declared_count!r}, but "
                f"{vocabulary_name} contains {len(vocabularies[vocabulary_name])} IDs"
            )

    return ICEWS14Metadata(**vocabularies)


def load_icews14_facts(
    archive_path: str | Path,
    metadata_path: str | Path,
    splits: tuple[str, ...] = _ICEWS14_SPLITS,
) -> list[dict[str, str]]:
    """Convert TFLEX ``Pt`` queries into dated fact records.

    Entity and relation IDs remain stable stringified vocabulary indices;
    human-readable labels are included for question rendering. The source split
    and row-derived source ID preserve provenance for later preprocessing.
    """
    invalid_splits = sorted(set(splits) - set(_ICEWS14_SPLITS))
    if invalid_splits:
        raise ValueError(f"Unsupported ICEWS14 split(s): {invalid_splits}")
    if len(set(splits)) != len(splits):
        raise ValueError("ICEWS14 splits must not contain duplicates")

    metadata = load_icews14_metadata(metadata_path)
    facts: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for split in splits:
            member_name = f"{split}.jsonl"
            try:
                member = archive.open(member_name)
            except KeyError as exc:
                raise ValueError(
                    f"{archive_path}: missing required archive member {member_name!r}"
                ) from exc

            with member:
                for line_number, raw_line in enumerate(member, start=1):
                    try:
                        record = json.loads(raw_line)
                    except JSONDecodeError as exc:
                        raise ValueError(
                            f"{member_name}:{line_number}: invalid JSON record"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"{member_name}:{line_number}: record must be a JSON object"
                        )
                    query_name = record.get("query_name")
                    if not isinstance(query_name, str):
                        raise ValueError(
                            f"{member_name}:{line_number}: missing string query_name"
                        )
                    if query_name != "Pt":
                        continue

                    query = record.get("query")
                    if (
                        not isinstance(query, list)
                        or len(query) != 3
                    ):
                        raise ValueError(
                            f"{member_name}:{line_number}: Pt query must contain "
                            "subject, relation, and object IDs"
                        )
                    subject_id = _vocabulary_id(
                        query[0], metadata.entities, "entity", member_name, line_number
                    )
                    relation_id = _vocabulary_id(
                        query[1], metadata.relations, "relation", member_name, line_number
                    )
                    object_id = _vocabulary_id(
                        query[2], metadata.entities, "entity", member_name, line_number
                    )
                    answers = record.get("answer")
                    if not isinstance(answers, list) or not answers:
                        raise ValueError(
                            f"{member_name}:{line_number}: Pt answer must be a non-empty list"
                        )

                    for answer_index, timestamp_id_value in enumerate(answers):
                        timestamp_id = _vocabulary_id(
                            timestamp_id_value,
                            metadata.timestamps,
                            "timestamp",
                            member_name,
                            line_number,
                        )
                        facts.append(
                            {
                                "subject_id": str(subject_id),
                                "relation_id": str(relation_id),
                                "object_id": str(object_id),
                                "timestamp": metadata.timestamps[timestamp_id],
                                "source_id": (
                                    f"ICEWS14:{split}:{line_number}:answer:{answer_index}"
                                ),
                                "source_split": split,
                                "subject_label": metadata.entities[subject_id],
                                "relation_label": metadata.relations[relation_id],
                                "object_label": metadata.entities[object_id],
                            }
                        )
    return facts


def _load_vocabulary(
    metadata: dict[str, Any],
    key: str,
    metadata_path: Path,
) -> dict[int, str]:
    records = metadata.get(key)
    if not isinstance(records, list):
        raise ValueError(f"{metadata_path}: {key} must be a list")

    vocabulary: dict[int, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{metadata_path}: {key}[{index}] must be an object")
        identifier = record.get("id")
        name = record.get("name")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
        ):
            raise ValueError(f"{metadata_path}: {key}[{index}].id must be a non-negative integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{metadata_path}: {key}[{index}].name must be a non-empty string")
        if identifier in vocabulary:
            raise ValueError(f"{metadata_path}: duplicate ID {identifier} in {key}")
        vocabulary[identifier] = name
    return vocabulary


def _vocabulary_id(
    value: object,
    vocabulary: dict[int, str],
    kind: str,
    member_name: str,
    line_number: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{member_name}:{line_number}: {kind} ID must be an integer, got {value!r}"
        )
    if value not in vocabulary:
        raise ValueError(
            f"{member_name}:{line_number}: unknown {kind} ID {value}"
        )
    return value
