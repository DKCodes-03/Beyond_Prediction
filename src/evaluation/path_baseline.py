"""Deterministic graph-path sanity baseline for synthetic temporal QA."""

from collections.abc import Mapping
from typing import Any

from src.graph.temporal_graph import TemporalGraph
from src.retrieval.subgraph_retriever import _validate_query
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


def graph_path_sanity_baseline(
    graph: TemporalGraph,
    qa_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return all graph-reachable answers and one deterministic witness per answer.

    The baseline traverses the complete graph using the query's ordered
    relations and temporal constraint. It does not inspect accepted answer
    labels, and is intended as a graph/QA sanity check rather than a learned
    baseline.
    """
    relation_values = (
        qa_record.get("relation_sequence")
        if isinstance(qa_record, Mapping)
        else None
    )
    if not isinstance(relation_values, (list, tuple)) or not relation_values:
        raise ValueError("graph path sanity baseline requires one or two relations")
    query = _validate_query(qa_record)
    frontier: dict[str, tuple[str, ...]] = {
        seed_id: () for seed_id in query["seed_entity_ids"]
    }

    for relation_id in query["relation_sequence"]:
        next_frontier: dict[str, tuple[str, ...]] = {}
        for entity_id, path in sorted(frontier.items()):
            outgoing = graph.outgoing_facts(entity_id, relation_id)
            for fact in filter_facts_by_temporal_constraint(
                outgoing, query["temporal_constraint"]
            ):
                target_id = fact["object_id"]
                candidate_path = (*path, fact["fact_id"])
                existing_path = next_frontier.get(target_id)
                if existing_path is None or candidate_path < existing_path:
                    next_frontier[target_id] = candidate_path
        frontier = next_frontier
        if not frontier:
            break

    answer_ids = sorted(frontier)
    return {
        "qa_id": query["qa_id"],
        "status": "ok" if answer_ids else "no_candidates",
        "predicted_answer_ids": answer_ids,
        "witness_paths": [
            {"answer_id": answer_id, "fact_ids": list(frontier[answer_id])}
            for answer_id in answer_ids
        ],
        "missing_entity_ids": [
            seed_id
            for seed_id in query["seed_entity_ids"]
            if seed_id not in graph.entity_ids
        ],
    }
