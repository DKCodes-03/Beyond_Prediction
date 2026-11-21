"""Canonicalize and validate structured temporal fact records."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def preprocess_facts(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return canonical facts, deduplicated and sorted by their fact content.

    IDs are trimmed strings (integer IDs are stringified), dates are normalized
    to ``YYYY-MM-DD``, and duplicate subject/relation/object/date tuples are
    represented once. For duplicates, the lexicographically smallest source ID
    is retained so output does not depend on input ordering.
    """
    unique_facts: dict[tuple[str, str, str, str], str] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {record_index}: expected a mapping")

        values = {
            field: _required_text(record, field, record_index)
            for field in ("subject_id", "relation_id", "object_id", "source_id")
        }
        timestamp = _normalize_timestamp(record.get("timestamp"), record_index)
        key = (
            values["subject_id"],
            values["relation_id"],
            values["object_id"],
            timestamp,
        )
        source_id = values["source_id"]
        if key not in unique_facts or source_id < unique_facts[key]:
            unique_facts[key] = source_id

    facts = []
    for key in sorted(unique_facts):
        subject_id, relation_id, object_id, timestamp = key
        content = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        fact_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
        facts.append(
            {
                "fact_id": f"fact:{fact_id}",
                "subject_id": subject_id,
                "relation_id": relation_id,
                "object_id": object_id,
                "timestamp": timestamp,
                "source_id": unique_facts[key],
            }
        )
    return facts


def _required_text(record: Mapping[str, Any], field: str, record_index: int) -> str:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"record {record_index}: {field} must be a non-empty string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"record {record_index}: {field} must not be empty")
    return normalized


def _normalize_timestamp(value: Any, record_index: int) -> str:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        text = value.strip()
        if _DATE_PATTERN.fullmatch(text):
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError as exc:
                raise ValueError(
                    f"record {record_index}: invalid timestamp {value!r}"
                ) from exc
        try:
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"record {record_index}: timestamp must be an ISO 8601 date or datetime"
            ) from exc
    else:
        raise ValueError(
            f"record {record_index}: timestamp must be an ISO 8601 date or datetime"
        )

    if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.date().isoformat()
