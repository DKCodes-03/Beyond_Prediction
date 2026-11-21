"""Filter point-dated facts against validated temporal constraints."""

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from src.graph.graph_utils import normalize_date


def filter_facts_by_temporal_constraint(
    facts: Iterable[Mapping[str, Any]],
    temporal_constraint: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return facts satisfying a point, interval, or historical-cutoff constraint.

    A ``point`` constraint matches exactly one calendar date. An ``interval``
    constraint uses ``start_date``, ``end_date``, ``include_start``, and
    ``include_end``; each boundary is compared independently. The existing
    ``on_or_before`` cutoff is inclusive.
    """
    constraint = _parse_constraint(temporal_constraint)
    selected = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, Mapping):
            raise ValueError(f"fact {index}: expected a mapping")
        if "timestamp" not in fact:
            raise ValueError(f"fact {index}: missing timestamp")
        timestamp = normalize_date(fact["timestamp"], f"fact {index} timestamp")
        if _matches(timestamp, constraint):
            selected.append(fact)
    return tuple(selected)


def _parse_constraint(
    constraint: Mapping[str, Any],
) -> tuple[str, date, date | None, bool, bool]:
    if not isinstance(constraint, Mapping):
        raise ValueError("temporal_constraint must be a mapping")

    kind = constraint.get("type")
    if kind == "point":
        if "date" not in constraint:
            raise ValueError("point constraint requires date")
        point_date = _parse_date(constraint["date"], "date")
        return "point", point_date, None, True, True

    if kind == "interval":
        if "start_date" not in constraint or "end_date" not in constraint:
            raise ValueError("interval constraint requires start_date and end_date")
        if not isinstance(constraint.get("include_start"), bool):
            raise ValueError("interval constraint include_start must be a boolean")
        if not isinstance(constraint.get("include_end"), bool):
            raise ValueError("interval constraint include_end must be a boolean")
        start = _parse_date(constraint["start_date"], "start_date")
        end = _parse_date(constraint["end_date"], "end_date")
        if start > end:
            raise ValueError("interval constraint start_date must not follow end_date")
        return (
            "interval",
            start,
            end,
            constraint["include_start"],
            constraint["include_end"],
        )

    if constraint.get("operator") == "on_or_before":
        if "cutoff_date" not in constraint:
            raise ValueError("on_or_before constraint requires cutoff_date")
        cutoff = _parse_date(constraint["cutoff_date"], "cutoff_date")
        return "on_or_before", cutoff, None, True, True

    raise ValueError(
        "temporal_constraint type must be 'point' or 'interval', or operator "
        "must be 'on_or_before'"
    )


def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(normalize_date(value, field))
    except ValueError as exc:
        raise ValueError(f"temporal_constraint {field} must be YYYY-MM-DD") from exc


def _matches(
    timestamp: str,
    constraint: tuple[str, date, date | None, bool, bool],
) -> bool:
    kind, start, end, include_start, include_end = constraint
    fact_date = date.fromisoformat(timestamp)
    if kind == "point":
        return fact_date == start
    if kind == "on_or_before":
        return fact_date <= start
    assert end is not None
    return (fact_date > start or include_start and fact_date == start) and (
        fact_date < end or include_end and fact_date == end
    )
