"""Load structured CSV and JSON data with TimelineKGQA validation."""

import csv
import json
from pathlib import Path
from typing import Any


TIMELINEKGQA_FACT_FIELDS = (
    "id",
    "subject",
    "predicate",
    "object",
    "start_time",
    "end_time",
)
TIMELINEKGQA_QUESTION_FIELDS = (
    "id",
    "source_kg_id",
    "question",
    "answer",
    "answer_type",
    "split",
)


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"{source}: file does not exist")

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"{source}: CSV header is missing")
            records = [dict(row) for row in reader]
    elif source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("records")
            if records is None:
                records = payload.get("data")
        else:
            records = None
        if not isinstance(records, list):
            raise ValueError(
                f"{source}: JSON must be a list or contain a records/data list"
            )
    else:
        raise ValueError(f"{source}: unsupported format {source.suffix!r}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source}: record {index} must be an object")
    return records


def load_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV file as a list of object records."""
    if Path(path).suffix.lower() != ".csv":
        raise ValueError(f"{path}: expected a CSV file")
    return _read_records(path)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array or records/data object."""
    if Path(path).suffix.lower() != ".json":
        raise ValueError(f"{path}: expected a JSON file")
    return _read_records(path)


load_structured_records = _read_records


def load_timelinekgqa_facts(path: str | Path) -> list[dict[str, str]]:
    """Load the TimelineKGQA unified-KG schema into interval-preserving rows."""
    source = Path(path)
    records = _read_records(source)
    facts: list[dict[str, str]] = []
    for index, record in enumerate(records):
        missing = [field for field in TIMELINEKGQA_FACT_FIELDS if not record.get(field)]
        if missing:
            raise ValueError(
                f"{source}: record {index} missing required field(s): "
                f"{', '.join(missing)}"
            )

        start_time = str(record["start_time"]).strip()
        end_time = str(record["end_time"]).strip()
        facts.append(
            {
                "source_id": f"TimelineKGQA:fact:{record['id']}",
                "subject_id": str(record["subject"]).strip(),
                "relation_id": str(record["predicate"]).strip(),
                "object_id": str(record["object"]).strip(),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return facts


def load_timelinekgqa_questions(
    path: str | Path,
    *,
    splits: tuple[str, ...] | None = None,
    answer_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load generated questions and normalize TimelineKGQA's ``dev`` split."""
    source = Path(path)
    records = _read_records(source)
    questions: list[dict[str, Any]] = []

    split_names = splits or ("train", "validation", "test")
    allowed_splits = {
        "validation" if split.lower() == "dev" else split.lower()
        for split in split_names
    }
    allowed_answer_types = (
        {answer_type.lower() for answer_type in answer_types}
        if answer_types is not None
        else None
    )

    for index, record in enumerate(records):
        missing = [
            field for field in TIMELINEKGQA_QUESTION_FIELDS
            if record.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"{source}: record {index} missing required field(s): "
                f"{', '.join(missing)}"
            )

        split = str(record["split"]).strip().lower()
        split = "validation" if split == "dev" else split
        if split not in allowed_splits:
            continue

        answer_type = str(record["answer_type"]).strip().lower()
        if allowed_answer_types is not None and answer_type not in allowed_answer_types:
            continue

        question = dict(record)
        question["split"] = split
        question["source_id"] = f"TimelineKGQA:question:{record['id']}"
        questions.append(question)
    return questions


def validate_timelinekgqa_dataset(
    facts_path: str | Path,
    questions_path: str | Path,
) -> dict[str, Any]:
    """Validate both public files and report their compatibility summary."""
    facts = load_timelinekgqa_facts(facts_path)
    questions = load_timelinekgqa_questions(questions_path)
    split_counts = {
        split: sum(question["split"] == split for question in questions)
        for split in ("train", "validation", "test")
    }
    answer_type_counts: dict[str, int] = {}
    for question in questions:
        answer_type = str(question["answer_type"]).strip().lower()
        answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1

    return {
        "fact_count": len(facts),
        "question_count": len(questions),
        "split_counts": split_counts,
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "entity_answers_supported": "entity" in answer_type_counts,
        "interval_times_present": any(
            fact["start_time"] != fact["end_time"] for fact in facts
        ),
    }
