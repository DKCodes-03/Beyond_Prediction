"""Shared validation helpers for canonical temporal graph facts."""

from collections.abc import Mapping
from datetime import date
from typing import Any


FACT_FIELDS = (
    "fact_id",
    "subject_id",
    "relation_id",
    "object_id",
    "timestamp",
    "source_id",
)


def canonical_id(value: Any, field: str) -> str:
    """Normalize a string or integer identifier and reject empty/boolean IDs."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a non-empty string or integer")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def normalize_date(value: Any, field: str = "timestamp") -> str:
    """Validate and return a canonical ``YYYY-MM-DD`` calendar date."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO 8601 date")
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 date") from exc
    if parsed_date.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD format")
    return value


def validate_fact(record: Any, index: int = 0) -> dict[str, str]:
    """Validate the required fields in one canonical fact record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"fact {index}: expected a mapping")
    fact: dict[str, str] = {}
    for field in FACT_FIELDS:
        try:
            value = record[field]
        except KeyError as exc:
            raise ValueError(f"fact {index}: missing {field}") from exc
        try:
            fact[field] = canonical_id(value, field)
        except ValueError as exc:
            raise ValueError(f"fact {index}: {exc}") from exc
    try:
        fact["timestamp"] = normalize_date(record["timestamp"])
    except ValueError as exc:
        raise ValueError(f"fact {index}: {exc}") from exc
    return fact


def fact_signature(fact: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Return the identity of an event independent of its provenance."""
    return (
        fact["subject_id"],
        fact["relation_id"],
        fact["object_id"],
        fact["timestamp"],
    )
