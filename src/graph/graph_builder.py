"""Build a validated in-memory temporal knowledge graph."""

from collections.abc import Iterable, Mapping
from typing import Any

import networkx as nx

from src.graph.graph_utils import fact_signature, validate_fact
from src.graph.temporal_graph import TemporalGraph


def build_temporal_graph(
    facts: Iterable[Mapping[str, Any]],
) -> TemporalGraph:
    """Validate facts and build a deterministic directed multigraph.

    Identical duplicate events are represented once. Their provenance is
    resolved deterministically by retaining the lexicographically smallest
    source ID, matching ``preprocess_facts``.
    """
    unique_facts: dict[str, dict[str, str]] = {}
    fact_ids_by_signature: dict[tuple[str, str, str, str], str] = {}

    for index, record in enumerate(facts):
        fact = validate_fact(record, index)
        fact_id = fact["fact_id"]
        signature = fact_signature(fact)

        previous = unique_facts.get(fact_id)
        if previous is not None:
            if fact_signature(previous) != signature:
                raise ValueError(
                    f"fact {index}: conflicting duplicate fact_id {fact_id!r}"
                )
            previous["source_id"] = min(previous["source_id"], fact["source_id"])
            continue

        previous_id = fact_ids_by_signature.get(signature)
        if previous_id is not None:
            raise ValueError(
                f"fact {index}: duplicate event has conflicting fact IDs "
                f"{previous_id!r} and {fact_id!r}"
            )
        unique_facts[fact_id] = fact
        fact_ids_by_signature[signature] = fact_id

    ordered_facts = sorted(
        unique_facts.values(),
        key=lambda fact: (
            fact["timestamp"],
            fact["subject_id"],
            fact["relation_id"],
            fact["object_id"],
            fact["fact_id"],
        ),
    )
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        sorted(
            {
                entity_id
                for fact in ordered_facts
                for entity_id in (fact["subject_id"], fact["object_id"])
            }
        )
    )
    for fact in ordered_facts:
        graph.add_edge(
            fact["subject_id"],
            fact["object_id"],
            key=fact["fact_id"],
            **fact,
        )
    return TemporalGraph(graph, ordered_facts)
