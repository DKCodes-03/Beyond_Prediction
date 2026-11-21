"""Generate deterministic, graph-verified temporal QA records."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any


_REQUIRED_FACT_FIELDS = (
    "fact_id",
    "subject_id",
    "relation_id",
    "object_id",
    "timestamp",
    "source_id",
)


def prepare_qa(
    facts: Iterable[Mapping[str, Any]],
    entity_labels: Mapping[str, str] | None = None,
    relation_labels: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Create one- and two-hop QA queries from canonical temporal facts.

    Each candidate query uses the latest timestamp on one of its supporting
    paths as its inclusive cutoff. Answers include every unique endpoint
    reachable under that cutoff, and each record carries all supporting fact
    IDs plus a deterministic witness path. Split assignment is left to the
    later split-generation stage.
    """
    normalized_facts, inferred_entities, inferred_relations = _validate_facts(facts)
    entities = _merge_labels(inferred_entities, entity_labels, "entity")
    relations = _merge_labels(inferred_relations, relation_labels, "relation")

    outgoing: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for fact in normalized_facts:
        outgoing[(fact["subject_id"], fact["relation_id"])].append(fact)
        by_subject[fact["subject_id"]].append(fact)
    for edges in outgoing.values():
        edges.sort(key=lambda fact: (fact["timestamp"], fact["fact_id"]))

    candidates: set[tuple[str, tuple[str, ...], str]] = set()
    for first in normalized_facts:
        candidates.add(
            (
                first["subject_id"],
                (first["relation_id"],),
                first["timestamp"],
            )
        )
        for second in by_subject.get(first["object_id"], []):
            candidates.add(
                (
                    first["subject_id"],
                    (first["relation_id"], second["relation_id"]),
                    max(first["timestamp"], second["timestamp"]),
                )
            )

    qa_records = []
    for seed, relation_sequence, cutoff in sorted(candidates):
        answer_ids: set[str] = set()
        supporting_fact_ids: set[str] = set()
        witness: tuple[dict[str, str], ...] | None = None
        for path in _matching_paths(outgoing, seed, relation_sequence, cutoff):
            answer_ids.add(path[-1]["object_id"])
            supporting_fact_ids.update(fact["fact_id"] for fact in path)
            if witness is None or tuple(
                fact["fact_id"] for fact in path
            ) < tuple(fact["fact_id"] for fact in witness):
                witness = path
        if witness is None or not answer_ids:
            raise ValueError(
                "candidate query has no graph-verified supporting path: "
                f"{seed!r}, {relation_sequence!r}, {cutoff!r}"
            )

        template_id = "one_hop" if len(relation_sequence) == 1 else "two_hop"
        qa_records.append(
            {
                "qa_id": _qa_id(seed, relation_sequence, cutoff),
                "question": _render_question(
                    seed, relation_sequence, cutoff, entities, relations
                ),
                "seed_entity_ids": [seed],
                "relation_sequence": list(relation_sequence),
                "temporal_constraint": {
                    "operator": "on_or_before",
                    "cutoff_date": cutoff,
                },
                "accepted_answer_ids": sorted(answer_ids),
                "provenance": {
                    "source": "synthetic_graph_derived",
                    "template_id": template_id,
                    "supporting_fact_ids": sorted(supporting_fact_ids),
                },
                "supporting_path_fact_ids": [
                    fact["fact_id"] for fact in witness
                ],
            }
        )
    return qa_records


def _validate_facts(
    facts: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    normalized: list[dict[str, str]] = []
    fact_ids: set[str] = set()
    entity_labels: dict[str, str] = {}
    relation_labels: dict[str, str] = {}

    for index, record in enumerate(facts):
        if not isinstance(record, Mapping):
            raise ValueError(f"fact {index}: expected a mapping")
        fact = {
            field: _required_text(record, field, index)
            for field in _REQUIRED_FACT_FIELDS
        }
        if fact["fact_id"] in fact_ids:
            raise ValueError(f"fact {index}: duplicate fact_id {fact['fact_id']!r}")
        fact_ids.add(fact["fact_id"])

        try:
            parsed_date = date.fromisoformat(fact["timestamp"])
        except ValueError as exc:
            raise ValueError(
                f"fact {index}: timestamp must be an ISO 8601 date"
            ) from exc
        if parsed_date.isoformat() != fact["timestamp"]:
            raise ValueError(f"fact {index}: timestamp must use YYYY-MM-DD format")

        for entity_id, label_field in (
            (fact["subject_id"], "subject_label"),
            (fact["object_id"], "object_label"),
        ):
            label = record.get(label_field)
            if label is not None:
                _record_label(entity_labels, entity_id, label, label_field, index)
        relation_label = record.get("relation_label")
        if relation_label is not None:
            _record_label(
                relation_labels,
                fact["relation_id"],
                relation_label,
                "relation_label",
                index,
            )
        normalized.append(fact)

    if not normalized:
        raise ValueError("cannot prepare QA from an empty fact collection")
    return normalized, entity_labels, relation_labels


def _required_text(record: Mapping[str, Any], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"fact {index}: {field} must be a non-empty string")
    return value.strip()


def _record_label(
    labels: dict[str, str],
    identifier: str,
    value: Any,
    field: str,
    index: int,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"fact {index}: {field} must be a non-empty string")
    label = value.strip()
    if identifier in labels and labels[identifier] != label:
        raise ValueError(f"fact {index}: conflicting labels for {identifier!r}")
    labels[identifier] = label


def _merge_labels(
    inferred: dict[str, str],
    supplied: Mapping[str, str] | None,
    kind: str,
) -> dict[str, str]:
    merged = dict(inferred)
    if supplied is None:
        return merged
    for identifier, label in supplied.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{kind} label IDs must be non-empty strings")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{kind} labels must be non-empty strings")
        merged[identifier.strip()] = label.strip()
    return merged


def _matching_paths(
    outgoing: dict[tuple[str, str], list[dict[str, str]]],
    seed: str,
    relation_sequence: tuple[str, ...],
    cutoff: str,
) -> Iterable[tuple[dict[str, str], ...]]:
    first_edges = [
        fact
        for fact in outgoing.get((seed, relation_sequence[0]), [])
        if fact["timestamp"] <= cutoff
    ]
    if len(relation_sequence) == 1:
        yield from ((fact,) for fact in first_edges)
        return

    for first in first_edges:
        second_edges = outgoing.get(
            (first["object_id"], relation_sequence[1]), []
        )
        yield from (
            (first, second)
            for second in second_edges
            if second["timestamp"] <= cutoff
        )


def _qa_id(seed: str, relations: tuple[str, ...], cutoff: str) -> str:
    signature = json.dumps(
        [seed, relations, cutoff], ensure_ascii=False, separators=(",", ":")
    )
    return f"qa:{hashlib.sha256(signature.encode('utf-8')).hexdigest()}"


def _render_question(
    seed: str,
    relations: tuple[str, ...],
    cutoff: str,
    entity_labels: dict[str, str],
    relation_labels: dict[str, str],
) -> str:
    subject = entity_labels.get(seed, seed)
    if len(relations) == 1:
        relation = relation_labels.get(relations[0], relations[0])
        return (
            f"Which entity is connected to {subject} by relation {relation} "
            f"on or before {cutoff}?"
        )
    relation_1 = relation_labels.get(relations[0], relations[0])
    relation_2 = relation_labels.get(relations[1], relations[1])
    return (
        f"Which entity is reached from {subject} by relation {relation_1} "
        f"followed by {relation_2}, using facts dated on or before {cutoff}?"
    )
