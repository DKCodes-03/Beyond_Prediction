"""Question-seeded, bounded temporal subgraph retrieval."""

from collections.abc import Mapping
from typing import Any

from src.graph.graph_utils import canonical_id
from src.graph.temporal_graph import TemporalGraph
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


_DEFAULT_SETTINGS = {
    "strategy": "relation_constrained_bfs",
    "max_hops": 2,
    "max_nodes": 128,
    "max_edges": 256,
}


def retrieve_subgraph(
    graph: TemporalGraph,
    qa_record: Mapping[str, Any],
    *,
    graph_id: str,
    settings: Mapping[str, Any] | None = None,
    example_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve a bounded subgraph for a directed, relation-constrained QA query.

    The query's relation sequence determines the exact traversal order, and all
    traversed facts must satisfy its temporal constraint. Seed entities are
    included in the node budget. The returned record follows the
    ``retrieved_subgraph`` contract and reports missing seeds, empty results,
    and whether either size limit truncated the evidence.
    """
    query = _validate_query(qa_record)
    graph_identifier = canonical_id(graph_id, "graph_id")
    resolved_settings = _validate_settings(settings, len(query["relation_sequence"]))
    max_nodes = resolved_settings["max_nodes"]
    max_edges = resolved_settings["max_edges"]

    present_seeds = tuple(
        entity_id
        for entity_id in query["seed_entity_ids"]
        if entity_id in graph.entity_ids
    )
    missing_seeds = tuple(
        entity_id
        for entity_id in query["seed_entity_ids"]
        if entity_id not in graph.entity_ids
    )
    if len(present_seeds) > max_nodes:
        raise ValueError(
            "max_nodes must be at least the number of seed entities present in the graph"
        )

    selected_facts: dict[str, dict[str, str]] = {}
    visited_entities = set(present_seeds)
    frontier = present_seeds
    candidate_answer_ids: set[str] = set()
    limits_applied = False

    for hop_index, relation_id in enumerate(query["relation_sequence"]):
        next_frontier: set[str] = set()
        for entity_id in frontier:
            outgoing = graph.outgoing_facts(entity_id, relation_id)
            for fact in filter_facts_by_temporal_constraint(
                outgoing, query["temporal_constraint"]
            ):
                fact_id = fact["fact_id"]
                target_id = fact["object_id"]
                if (
                    target_id not in visited_entities
                    and len(visited_entities) >= max_nodes
                ):
                    limits_applied = True
                    continue
                if fact_id not in selected_facts and len(selected_facts) >= max_edges:
                    limits_applied = True
                    continue
                selected_facts[fact_id] = dict(fact)
                visited_entities.add(target_id)
                next_frontier.add(target_id)

        if hop_index == len(query["relation_sequence"]) - 1:
            candidate_answer_ids.update(next_frontier)
        frontier = tuple(sorted(next_frontier))
        if not frontier:
            break

    ordered_facts = sorted(
        selected_facts.values(),
        key=lambda fact: (fact["timestamp"], fact["fact_id"]),
    )
    entity_ids = sorted(
        {
            entity_id
            for fact in ordered_facts
            for entity_id in (fact["subject_id"], fact["object_id"])
        }
        | set(present_seeds)
    )
    return {
        "example_id": canonical_id(
            query["qa_id"] if example_id is None else example_id, "example_id"
        ),
        "qa_id": query["qa_id"],
        "graph_id": graph_identifier,
        "query_context": {
            "question": query["question"],
            "seed_entity_ids": list(query["seed_entity_ids"]),
            "temporal_constraint": dict(query["temporal_constraint"]),
        },
        "facts": ordered_facts,
        "entity_id_map": {entity_id: entity_id for entity_id in entity_ids},
        "candidate_answer_ids": sorted(candidate_answer_ids),
        "retrieval_settings": resolved_settings,
        "retrieval_status": (
            "ok"
            if ordered_facts
            else "missing_entity"
            if not present_seeds
            else "empty_result"
        ),
        "missing_entity_ids": list(missing_seeds),
        "limits_applied": limits_applied,
    }


def _validate_query(qa_record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(qa_record, Mapping):
        raise ValueError("qa_record must be a mapping")
    try:
        qa_id = canonical_id(qa_record["qa_id"], "qa_id")
    except KeyError as exc:
        raise ValueError("qa_record is missing qa_id") from exc
    question = qa_record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("qa_record question must be a non-empty string")

    seed_values = qa_record.get("seed_entity_ids")
    if not isinstance(seed_values, (list, tuple)) or not seed_values:
        raise ValueError("qa_record seed_entity_ids must be a non-empty list")
    seed_ids = tuple(
        sorted(canonical_id(value, "seed_entity_id") for value in seed_values)
    )
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("qa_record seed_entity_ids must be unique")

    relation_values = qa_record.get("relation_sequence")
    if not isinstance(relation_values, (list, tuple)) or not 1 <= len(
        relation_values
    ) <= 2:
        raise ValueError("qa_record relation_sequence must contain one or two relations")
    relation_sequence = tuple(
        canonical_id(value, "relation_id") for value in relation_values
    )
    if not isinstance(qa_record.get("temporal_constraint"), Mapping):
        raise ValueError("qa_record temporal_constraint must be a mapping")
    temporal_constraint = dict(qa_record["temporal_constraint"])
    filter_facts_by_temporal_constraint((), temporal_constraint)
    return {
        "qa_id": qa_id,
        "question": question.strip(),
        "seed_entity_ids": seed_ids,
        "relation_sequence": relation_sequence,
        "temporal_constraint": temporal_constraint,
    }


def _validate_settings(
    settings: Mapping[str, Any] | None, relation_count: int
) -> dict[str, Any]:
    if settings is not None and not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping")
    values = {**_DEFAULT_SETTINGS, **(settings or {})}
    if values["strategy"] != "relation_constrained_bfs":
        raise ValueError("strategy must be 'relation_constrained_bfs'")
    for name, minimum, maximum in (
        ("max_hops", 1, 2),
        ("max_nodes", 1, None),
        ("max_edges", 0, None),
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}")
    if values["max_hops"] < relation_count:
        raise ValueError("max_hops must be at least the relation sequence length")
    return {
        "strategy": values["strategy"],
        "max_hops": values["max_hops"],
        "max_nodes": values["max_nodes"],
        "max_edges": values["max_edges"],
    }
