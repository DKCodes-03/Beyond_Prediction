"""In-memory directed multigraph representation of dated facts."""

from collections.abc import Iterable, Mapping

import networkx as nx

from src.graph.graph_utils import canonical_id


class TemporalGraph:
    """A directed multigraph with stable fact IDs and adjacency indexes."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        facts: Iterable[Mapping[str, str]],
    ) -> None:
        self.graph = graph
        self._facts = {fact["fact_id"]: dict(fact) for fact in facts}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}
        for fact_id, fact in self._facts.items():
            self._outgoing.setdefault(fact["subject_id"], []).append(fact_id)
            self._incoming.setdefault(fact["object_id"], []).append(fact_id)
        for index in (self._outgoing, self._incoming):
            for fact_ids in index.values():
                fact_ids.sort(
                    key=lambda item: (
                        self._facts[item]["timestamp"],
                        self._facts[item]["fact_id"],
                    )
                )

    @property
    def facts(self) -> tuple[dict[str, str], ...]:
        """Return copies of facts in deterministic order."""
        return tuple(dict(self._facts[fact_id]) for fact_id in sorted(self._facts))

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    @property
    def entity_ids(self) -> frozenset[str]:
        return frozenset(self.graph.nodes)

    @property
    def relation_ids(self) -> frozenset[str]:
        return frozenset(fact["relation_id"] for fact in self._facts.values())

    def get_fact(self, fact_id: str | int) -> dict[str, str] | None:
        """Return a fact by ID, or ``None`` if it is not in this graph."""
        normalized_id = canonical_id(fact_id, "fact_id")
        fact = self._facts.get(normalized_id)
        return None if fact is None else dict(fact)

    def outgoing_facts(
        self,
        entity_id: str | int,
        relation_id: str | int | None = None,
    ) -> tuple[dict[str, str], ...]:
        """Return outgoing facts, optionally restricted to one relation."""
        normalized_entity = canonical_id(entity_id, "entity_id")
        normalized_relation = (
            None if relation_id is None else canonical_id(relation_id, "relation_id")
        )
        return self._indexed_facts(
            self._outgoing.get(normalized_entity, ()), normalized_relation
        )

    def incoming_facts(
        self,
        entity_id: str | int,
        relation_id: str | int | None = None,
    ) -> tuple[dict[str, str], ...]:
        """Return incoming facts, optionally restricted to one relation."""
        normalized_entity = canonical_id(entity_id, "entity_id")
        normalized_relation = (
            None if relation_id is None else canonical_id(relation_id, "relation_id")
        )
        return self._indexed_facts(
            self._incoming.get(normalized_entity, ()), normalized_relation
        )

    def _indexed_facts(
        self, fact_ids: Iterable[str], relation_id: str | None
    ) -> tuple[dict[str, str], ...]:
        return tuple(
            dict(self._facts[fact_id])
            for fact_id in fact_ids
            if relation_id is None
            or self._facts[fact_id]["relation_id"] == relation_id
        )
