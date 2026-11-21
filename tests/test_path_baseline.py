"""Tests for the deterministic graph-path sanity baseline."""

import pytest

from src.evaluation.path_baseline import graph_path_sanity_baseline
from src.graph.graph_builder import build_temporal_graph


@pytest.fixture
def graph():
    return build_temporal_graph(
        [
            {
                "fact_id": "f_ab",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "b",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_ab_alt",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "b",
                "timestamp": "2014-01-01",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_bc",
                "subject_id": "b",
                "relation_id": "r2",
                "object_id": "c",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_bd",
                "subject_id": "b",
                "relation_id": "r2",
                "object_id": "d",
                "timestamp": "2014-01-03",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_wrong_relation",
                "subject_id": "b",
                "relation_id": "r3",
                "object_id": "e",
                "timestamp": "2014-01-01",
                "source_id": "fixture",
            },
        ]
    )


@pytest.fixture
def query():
    return {
        "qa_id": "path-query",
        "question": "Which entity follows r1 and then r2?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
        "accepted_answer_ids": ["ignored-label"],
    }


def test_baseline_returns_all_temporally_valid_endpoints_and_witnesses(graph, query):
    result = graph_path_sanity_baseline(graph, query)

    assert result == {
        "qa_id": "path-query",
        "status": "ok",
        "predicted_answer_ids": ["c"],
        "witness_paths": [{"answer_id": "c", "fact_ids": ["f_ab", "f_bc"]}],
        "missing_entity_ids": [],
    }


def test_baseline_supports_one_hop_queries(graph, query):
    query["relation_sequence"] = ["r1"]

    result = graph_path_sanity_baseline(graph, query)

    assert result["predicted_answer_ids"] == ["b"]
    assert result["witness_paths"] == [
        {"answer_id": "b", "fact_ids": ["f_ab"]}
    ]


def test_baseline_selects_deterministic_witness_for_duplicate_routes(graph, query):
    query["temporal_constraint"] = {
        "operator": "on_or_before",
        "cutoff_date": "2014-01-03",
    }

    result = graph_path_sanity_baseline(graph, query)

    assert result["predicted_answer_ids"] == ["c", "d"]
    assert result["witness_paths"] == [
        {"answer_id": "c", "fact_ids": ["f_ab", "f_bc"]},
        {"answer_id": "d", "fact_ids": ["f_ab", "f_bd"]},
    ]


@pytest.mark.parametrize(
    ("seed_entity_ids", "expected_missing"),
    [(["missing"], ["missing"]), (["a"], [])],
)
def test_baseline_explicitly_reports_queries_without_answers(
    graph, query, seed_entity_ids, expected_missing
):
    query["seed_entity_ids"] = seed_entity_ids
    query["temporal_constraint"] = {
        "operator": "on_or_before",
        "cutoff_date": "2013-12-31",
    }

    result = graph_path_sanity_baseline(graph, query)

    assert result["status"] == "no_candidates"
    assert result["predicted_answer_ids"] == []
    assert result["witness_paths"] == []
    assert result["missing_entity_ids"] == expected_missing


def test_baseline_rejects_relation_sequences_outside_supported_hop_range(
    graph, query
):
    query["relation_sequence"] = []

    with pytest.raises(ValueError, match="one or two relations"):
        graph_path_sanity_baseline(graph, query)
