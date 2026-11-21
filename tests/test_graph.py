import json
from datetime import date
from pathlib import Path

import pytest

from src.graph.graph_builder import build_temporal_graph
from src.ingestion.preprocess import preprocess_facts


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "temporal_fixture.json"


def _find_paths(facts, seed_entity_id, relation_sequence, cutoff_date):
    cutoff = date.fromisoformat(cutoff_date)
    paths = [(seed_entity_id, [])]

    for relation_id in relation_sequence:
        next_paths = []
        for entity_id, fact_ids in paths:
            for fact in facts:
                if (
                    fact["subject_id"] == entity_id
                    and fact["relation_id"] == relation_id
                    and date.fromisoformat(fact["timestamp"]) <= cutoff
                ):
                    next_paths.append((fact["object_id"], fact_ids + [fact["fact_id"]]))
        paths = next_paths

    return paths


def test_tiny_temporal_fixture_has_hand_checkable_paths_and_boundaries():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    graph = build_temporal_graph(fixture["facts"])
    facts_by_id = {fact["fact_id"]: fact for fact in fixture["facts"]}

    assert len(facts_by_id) == 4
    assert graph.fact_count == 4
    for query in fixture["queries"]:
        actual_paths = _find_paths(
            graph.facts,
            query["seed_entity_id"],
            query["relation_sequence"],
            query["cutoff_date"],
        )
        actual_path_ids = sorted(path_ids for _, path_ids in actual_paths)
        expected_path_ids = sorted(query["expected_paths"])
        actual_answers = sorted({entity_id for entity_id, _ in actual_paths})

        assert actual_path_ids == expected_path_ids, query["query_id"]
        assert actual_answers == sorted(query["expected_answer_ids"]), query["query_id"]

        for path_ids in query["expected_paths"]:
            assert all(fact_id in facts_by_id for fact_id in path_ids)


def test_temporal_graph_builds_directed_multigraph_and_adjacency_indexes():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    graph = build_temporal_graph(fixture["facts"])

    assert graph.graph.is_directed()
    assert graph.graph.is_multigraph()
    assert graph.entity_ids == frozenset(
        {"entity_a", "entity_b", "entity_c", "entity_d", "entity_e"}
    )
    assert graph.relation_ids == frozenset({"relation_1", "relation_2"})
    assert [
        fact["fact_id"] for fact in graph.outgoing_facts("entity_a", "relation_1")
    ] == ["f_ab", "f_ad"]
    assert [
        fact["fact_id"] for fact in graph.incoming_facts("entity_c")
    ] == ["f_bc"]


def test_temporal_graph_deduplicates_events_and_resolves_provenance():
    fact = {
        "fact_id": "f_ab",
        "subject_id": "entity_a",
        "relation_id": "relation_1",
        "object_id": "entity_b",
        "timestamp": "2014-01-01",
        "source_id": "z-source",
    }

    graph = build_temporal_graph([fact, {**fact, "source_id": "a-source"}])

    assert graph.fact_count == 1
    assert graph.get_fact("f_ab")["source_id"] == "a-source"
    assert graph.graph.number_of_edges() == 1


def test_preprocessing_is_deterministic_for_duplicate_facts_and_graph_construction():
    duplicate_facts = [
        {
            "subject_id": " entity_a ",
            "relation_id": "relation_1",
            "object_id": "entity_b",
            "timestamp": "2014-01-01T02:00:00+02:00",
            "source_id": "z-source",
        },
        {
            "subject_id": "entity_a",
            "relation_id": "relation_1",
            "object_id": "entity_b",
            "timestamp": "2014-01-01",
            "source_id": "a-source",
        },
    ]

    processed = preprocess_facts(duplicate_facts)
    reordered = preprocess_facts(list(reversed(duplicate_facts)))
    graph = build_temporal_graph(processed)

    assert processed == reordered
    assert len(processed) == 1
    assert processed[0]["timestamp"] == "2014-01-01"
    assert processed[0]["source_id"] == "a-source"
    assert graph.fact_count == 1
    assert graph.get_fact(processed[0]["fact_id"]) == processed[0]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (None, "expected a mapping"),
        (
            {
                "fact_id": "f",
                "subject_id": "s",
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-01-01",
            },
            "missing source_id",
        ),
        (
            {
                "fact_id": "f",
                "subject_id": "s",
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-02-30",
                "source_id": "source",
            },
            "timestamp must be an ISO 8601 date",
        ),
        (
            {
                "fact_id": "f",
                "subject_id": True,
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-01-01",
                "source_id": "source",
            },
            "subject_id must be a non-empty string or integer",
        ),
    ],
)
def test_temporal_graph_rejects_malformed_facts(record, message):
    with pytest.raises(ValueError, match=message):
        build_temporal_graph([record])


def test_temporal_graph_rejects_conflicting_fact_ids():
    fact = {
        "fact_id": "f_ab",
        "subject_id": "entity_a",
        "relation_id": "relation_1",
        "object_id": "entity_b",
        "timestamp": "2014-01-01",
        "source_id": "fixture",
    }

    with pytest.raises(ValueError, match="conflicting duplicate fact_id"):
        build_temporal_graph([fact, {**fact, "object_id": "entity_c"}])
