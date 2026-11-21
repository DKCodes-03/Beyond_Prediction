"""Format-dispatching loader for structured and explicit text fact sources."""

import json
from pathlib import Path
from typing import Any

from src.ingestion.text_loader import load_txt_facts


def load_fact_records(path: str | Path) -> list[dict[str, Any]]:
    """Dispatch a JSON, JSONL, CSV, or TXT fact file to an existing loader."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".txt" or suffix == ".text":
        return load_txt_facts(source)
    if suffix == ".jsonl":
        if not source.exists():
            raise ValueError(f"{source}: file does not exist")
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{line_number}: invalid JSON record"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{source}:{line_number}: record must be a JSON object"
                    )
                records.append(record)
        return records
    if suffix == ".json":
        from src.ingestion.csv_loader import _read_records

        return _read_records(source)
    if suffix == ".csv":
        from src.ingestion.csv_loader import _read_records

        return _read_records(source)
    raise ValueError(f"{source}: unsupported format {suffix!r}")


load_ingestion_records = load_fact_records
