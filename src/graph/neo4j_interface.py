"""Optional Neo4j persistence adapter.

The core graph implementation remains an in-memory :class:`TemporalGraph`.
This module imports the Neo4j driver only when an adapter is explicitly used,
and never changes core retrieval behavior.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from src.graph.temporal_graph import TemporalGraph


class Neo4jInterface:
    """Small, opt-in adapter for persisting and querying temporal facts."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        *,
        user: str | None = None,
        driver: Any | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
        self.username = (
            username
            or user
            or os.getenv("NEO4J_USERNAME")
            or os.getenv("NEO4J_USER")
        )
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE")

        if driver is not None:
            self._driver = driver
        else:
            if not self.uri:
                raise ValueError(
                    "Neo4j URI is required; set NEO4J_URI or pass uri="
                )
            if not self.username or not self.password:
                raise ValueError(
                    "Neo4j credentials are required; set NEO4J_USERNAME and "
                    "NEO4J_PASSWORD or pass username= and password="
                )
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "The optional Neo4j driver is not installed. "
                    "Install it with `pip install neo4j`."
                ) from exc
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )

    @classmethod
    def from_environment(cls, database: str | None = None) -> "Neo4jInterface":
        """Create an adapter from ``NEO4J_*`` environment variables."""
        return cls(database=database)

    def close(self) -> None:
        """Close the underlying driver."""
        close = getattr(self._driver, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Neo4jInterface":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def verify_connectivity(self) -> bool:
        """Verify the server connection, if the driver supports it."""
        verify = getattr(self._driver, "verify_connectivity", None)
        if callable(verify):
            verify()
            return True

        with self._session() as session:
            if hasattr(session, "run"):
                session.run("RETURN 1 AS ok").consume()
        return True

    def is_available(self) -> bool:
        """Return whether Neo4j is currently reachable."""
        try:
            return self.verify_connectivity()
        except Exception:
            return False

    def execute_query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return result records as dictionaries."""
        with self._session() as session:
            result = session.run(query, parameters or {})
            consume = getattr(result, "consume", None)
            records = [dict(record) for record in result]
            if callable(consume):
                consume()
            return records

    def persist_temporal_graph(self, graph: TemporalGraph) -> int:
        """Persist all facts in an in-memory graph and return their count."""
        facts = [
            dict(fact)
            for fact in graph.facts
        ]
        if not facts:
            return 0

        query = """
        UNWIND $facts AS fact
        MERGE (subject:Entity {id: fact.subject_id})
        MERGE (object:Entity {id: fact.object_id})
        MERGE (subject)-[edge:Fact {fact_id: fact.fact_id}]->(object)
        SET edge.relation_id = fact.relation_id,
            edge.timestamp = fact.timestamp,
            edge.source_id = fact.source_id
        """
        self.execute_query(query, {"facts": facts})
        return len(facts)

    write_graph = persist_temporal_graph
    persist_graph = persist_temporal_graph

    def clear_graph(self) -> None:
        """Delete Fact edges and Entity nodes managed by this adapter."""
        self.execute_query(
            """
            MATCH (:Entity)-[edge:Fact]->(:Entity)
            DELETE edge
            """
        )
        self.execute_query(
            """
            MATCH (entity:Entity)
            WHERE NOT (entity)--()
            DELETE entity
            """
        )

    def query_facts(
        self,
        *,
        cutoff: str | None = None,
        relation_id: str | None = None,
        subject_id: str | None = None,
        object_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Query persisted facts with optional temporal and identity filters."""
        clauses = [
            "MATCH (subject:Entity)-[edge:Fact]->(object:Entity)",
            "WHERE edge.fact_id IS NOT NULL",
        ]
        parameters: dict[str, Any] = {}
        if cutoff is not None:
            clauses.append("AND edge.timestamp <= $cutoff")
            parameters["cutoff"] = cutoff
        if relation_id is not None:
            clauses.append("AND edge.relation_id = $relation_id")
            parameters["relation_id"] = relation_id
        if subject_id is not None:
            clauses.append("AND subject.id = $subject_id")
            parameters["subject_id"] = subject_id
        if object_id is not None:
            clauses.append("AND object.id = $object_id")
            parameters["object_id"] = object_id
        clauses.append(
            "RETURN edge.fact_id AS fact_id, subject.id AS subject_id, "
            "edge.relation_id AS relation_id, object.id AS object_id, "
            "edge.timestamp AS timestamp, edge.source_id AS source_id "
            "ORDER BY timestamp, subject_id, relation_id, object_id, fact_id"
        )

        records = self.execute_query("\n".join(clauses), parameters)
        return [
            {key: str(value) for key, value in record.items() if value is not None}
            for record in records
        ]

    def load_facts(self) -> list[dict[str, str]]:
        """Load persisted facts in deterministic fact order."""
        return self.query_facts()

    def _session(self) -> Any:
        session = getattr(self._driver, "session", None)
        if callable(session):
            kwargs = {}
            if self.database:
                kwargs["database"] = self.database
            return session(**kwargs)
        raise RuntimeError("Neo4j driver does not provide a session() method")
