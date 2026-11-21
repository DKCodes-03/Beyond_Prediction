"""Lightweight ingestion helpers for text-based temporal fact records."""

import csv
import json
import re
from datetime import date
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


_DATE = r"\d{4}-\d{2}-\d{2}"
_ARROW_PATTERN = re.compile(
    rf"^\s*(?P<subject>.*?)\s+-\[\s*(?P<relation>.*?)\s*\]->"
    rf"\s*(?P<object>.*?)\s+\((?P<timestamp>{_DATE})\)\s*$"
)
_ON_PATTERN = re.compile(
    rf"^\s*(?P<subject>.*?)\s+(?P<relation>[^\s]+)\s+"
    rf"(?P<object>.*?)\s+(?:on|at)\s+(?P<timestamp>{_DATE})\s*$",
    re.IGNORECASE,
)


def extract_temporal_facts_from_text(
    text: str,
    *,
    source_id: str = "TXT",
    strict: bool = False,
) -> list[dict[str, str]]:
    """Extract point temporal facts from explicit TXT records.

    Supported records are JSON objects, pipe/tab/comma-delimited fields,
    ``subject -[relation]-> object (YYYY-MM-DD)`` arrows, and
    ``subject relation object on YYYY-MM-DD`` clauses.  This deliberately uses
    no trained NER/relation/temporal model.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    facts: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _is_text_header(line):
            continue

        record = _parse_text_record(line)
        if record is None:
            if strict:
                raise ValueError(
                    f"line {line_number}: unsupported temporal fact record"
                )
            continue

        fact = _normalize_extracted_record(
            record,
            source_id=f"{source_id}:{line_number}",
        )
        if fact is None:
            if strict:
                raise ValueError(
                    f"line {line_number}: missing subject, relation, object, "
                    "or timestamp"
                )
            continue

        fact_key = (
            fact["subject_id"],
            fact["relation_id"],
            fact["object_id"],
            fact["timestamp"],
        )
        if not any(
            (
                existing["subject_id"],
                existing["relation_id"],
                existing["object_id"],
                existing["timestamp"],
            )
            == fact_key
            for existing in facts
        ):
            facts.append(fact)
    return facts


def load_txt_facts(path: str | Path) -> list[dict[str, str]]:
    """Load explicit point-fact records from a UTF-8 text file."""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"{source}: file does not exist")
    if source.suffix.lower() not in ("", ".txt", ".text"):
        raise ValueError(f"{source}: unsupported format {source.suffix!r}")
    text = source.read_text(encoding="utf-8")
    return extract_temporal_facts_from_text(
        text,
        source_id=str(source),
        strict=True,
    )


def evaluate_extraction_quality(
    extracted: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return extraction counts and, if references are supplied, fact-level metrics.

    Extraction metrics concern only the facts produced by a loader.  They are
    independent of graph reasoning, answer accuracy, and explanation faithfulness.
    """
    extracted_keys = {
        _fact_key(record)
        for record in extracted
        if isinstance(record, Mapping) and _fact_key(record) is not None
    }
    extracted_items = [
        record
        for record in extracted
        if isinstance(record, Mapping) and _fact_key(record) is not None
    ]
    extracted_count = len(extracted_items)

    report: dict[str, Any] = {
        "extracted_count": extracted_count,
        "unique_extracted_count": len(extracted_keys),
        "duplicate_count": extracted_count - len(extracted_keys),
    }

    if references is not None:
        reference_keys = {
            key
            for record in references
            if isinstance(record, Mapping)
            for key in (_fact_key(record),)
            if key is not None
        }
        true_positives = len(extracted_keys & reference_keys)
        false_positives = len(extracted_keys - reference_keys)
        false_negatives = len(reference_keys - extracted_keys)
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report.update(
            {
                "reference_count": len(reference_keys),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return report


# Descriptive aliases for callers that use ``text`` rather than ``txt`` names.
extract_temporal_facts = extract_temporal_facts_from_text
load_text_facts = load_txt_facts


def _parse_text_record(line: str) -> dict[str, Any] | Sequence[Any] | None:
    if line.startswith("{"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None
        return record if isinstance(record, dict) else None

    arrow_match = _ARROW_PATTERN.match(line)
    if arrow_match:
        subject = arrow_match.group("subject")
        relation = arrow_match.group("relation")
        object_id = arrow_match.group("object")
        timestamp = arrow_match.group("timestamp")
        if subject and relation and object_id:
            return {
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": object_id.strip(),
                "timestamp": timestamp,
            }

    on_match = _ON_PATTERN.match(line)
    if on_match:
        subject = on_match.group("subject")
        relation = on_match.group("relation")
        object_id = on_match.group("object")
        timestamp = on_match.group("timestamp")
        if subject and relation and object_id:
            return {
                "subject": subject.strip(),
                "relation": relation.strip(),
                "object": object_id.strip(),
                "timestamp": timestamp,
            }

    delimiter = "|" if "|" in line else ("\t" if "\t" in line else None)
    if delimiter:
        values = [value.strip() for value in line.split(delimiter)]
    elif "," in line:
        values = [value.strip() for value in next(csv.reader([line]))]
    else:
        values = []

    if len(values) >= 4:
        return {
            "subject": values[0],
            "relation": values[1],
            "object": values[2],
            "timestamp": values[3],
        }
    return None


def _normalize_extracted_record(
    record: dict[str, Any] | Sequence[Any],
    *,
    source_id: str,
) -> dict[str, str] | None:
    if isinstance(record, Mapping):
        subject = _first_value(
            record, ("subject_id", "subject", "head", "source")
        )
        relation = _first_value(
            record,
            ("relation_id", "relation", "predicate", "predicate_id", "edge"),
        )
        object_id = _first_value(
            record, ("object_id", "object", "tail", "target")
        )
        timestamp = _first_value(
            record, ("timestamp", "date", "time", "start_time")
        )
    else:
        values = list(record)
        if len(values) < 4:
            return None
        subject, relation, object_id, timestamp = values[:4]

    values = (subject, relation, object_id, timestamp)
    if any(value is None or not str(value).strip() for value in values):
        return None

    timestamp_text = str(timestamp).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp_text):
        return None
    try:
        date.fromisoformat(timestamp_text)
    except ValueError:
        return None
    return {
        "subject_id": str(subject).strip(),
        "relation_id": str(relation).strip(),
        "object_id": str(object_id).strip(),
        "timestamp": timestamp_text,
        "source_id": source_id,
    }


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _is_text_header(line: str) -> bool:
    normalized = line.lower().replace("_", "")
    for delimiter in ("|", "\t", ","):
        if delimiter in line:
            fields = {
                field.strip().lower().replace("_", "")
                for field in line.split(delimiter)
            }
            if {"subject", "relation", "object", "timestamp"} <= fields:
                return True
            if {"subjectid", "relationid", "objectid", "timestamp"} <= fields:
                return True
    return False


def _fact_key(
    record: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    subject = _first_value(record, ("subject_id", "subject"))
    relation = _first_value(record, ("relation_id", "relation", "predicate"))
    object_id = _first_value(record, ("object_id", "object"))
    timestamp = _first_value(record, ("timestamp", "date", "time", "start_time"))
    values = (subject, relation, object_id, timestamp)
    if any(value is None or not str(value).strip() for value in values):
        return None
    return tuple(str(value).strip() for value in values)
