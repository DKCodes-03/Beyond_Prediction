"""Tests for temporal filtering, bounded retrieval, and cache isolation."""

import pytest

from src.graph.graph_builder import build_temporal_graph
from src.retrieval.retrieval_utils import (
    assess_retrieval_support,
    retrieval_cache_key,
    summarize_support_diagnostics,
)
from src.retrieval.subgraph_retriever import retrieve_subgraph
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


@pytest.fixture
def temporal_graph():
    return build_temporal_graph(
        [
            {
                "fact_id": "f_ab",
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
                "fact_id": "f_ad",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "d",
                "timestamp": "2014-01-03",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_de",
                "subject_id": "d",
                "relation_id": "r2",
                "object_id": "e",
                "timestamp": "2014-01-04",
                "source_id": "fixture",
            },
        ]
    )


@pytest.fixture
def two_hop_query():
    return {
        "qa_id": "query-1",
        "question": "Which entity is reached by r1 then r2?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
    }


@pytest.fixture
def dated_facts():
    return tuple(
        {"fact_id": fact_id, "timestamp": timestamp}
        for fact_id, timestamp in (
            ("before", "2014-01-01"),
            ("start", "2014-01-02"),
            ("middle", "2014-01-03"),
            ("end", "2014-01-04"),
            ("after", "2014-01-05"),
        )
    )


def _fact_ids(facts):
    return [fact["fact_id"] for fact in facts]


def test_point_constraint_matches_only_exact_calendar_date(dated_facts):
    selected = filter_facts_by_temporal_constraint(
        dated_facts, {"type": "point", "date": "2014-01-03"}
    )

    assert _fact_ids(selected) == ["middle"]


@pytest.mark.parametrize(
    ("include_start", "include_end", "expected"),
    [
        (True, True, ["start", "middle", "end"]),
        (True, False, ["start", "middle"]),
        (False, True, ["middle", "end"]),
        (False, False, ["middle"]),
    ],
)
def test_interval_boundary_inclusivity(
    dated_facts, include_start, include_end, expected
):
    selected = filter_facts_by_temporal_constraint(
        dated_facts,
        {
            "type": "interval",
            "start_date": "2014-01-02",
            "end_date": "2014-01-04",
            "include_start": include_start,
            "include_end": include_end,
        },
    )

    assert _fact_ids(selected) == expected


def test_existing_historical_cutoff_is_inclusive(dated_facts):
    selected = filter_facts_by_temporal_constraint(
        dated_facts,
        {"operator": "on_or_before", "cutoff_date": "2014-01-02"},
    )

    assert _fact_ids(selected) == ["before", "start"]


@pytest.mark.parametrize(
    "constraint",
    [
        {"type": "point", "date": "2014-02-30"},
        {
            "type": "interval",
            "start_date": "2014-01-04",
            "end_date": "2014-01-02",
            "include_start": True,
            "include_end": True,
        },
        {
            "type": "interval",
            "start_date": "2014-01-02",
            "end_date": "2014-01-04",
            "include_start": 1,
            "include_end": True,
        },
    ],
)
def test_invalid_temporal_constraints_raise(dated_facts, constraint):
    with pytest.raises(ValueError):
        filter_facts_by_temporal_constraint(dated_facts, constraint)


def test_invalid_fact_timestamp_raises():
    with pytest.raises(ValueError, match="fact 0 timestamp"):
        filter_facts_by_temporal_constraint(
            [{"timestamp": "2014-1-2"}],
            {"type": "point", "date": "2014-01-02"},
        )


def test_retrieval_follows_relation_sequence_and_inclusive_cutoff(
    temporal_graph, two_hop_query
):
    result = retrieve_subgraph(
        temporal_graph, two_hop_query, graph_id="fixture-v1"
    )

    assert [fact["fact_id"] for fact in result["facts"]] == ["f_ab", "f_bc"]
    assert result["candidate_answer_ids"] == ["c"]
    assert result["retrieval_status"] == "ok"
    assert result["missing_entity_ids"] == []
    assert result["limits_applied"] is False


def test_two_hop_retrieval_returns_all_endpoints_at_inclusive_cutoff(
    temporal_graph, two_hop_query
):
    query = {
        **two_hop_query,
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-04",
        },
    }

    result = retrieve_subgraph(temporal_graph, query, graph_id="fixture-v1")

    assert [fact["fact_id"] for fact in result["facts"]] == [
        "f_ab",
        "f_bc",
        "f_ad",
        "f_de",
    ]
    assert result["candidate_answer_ids"] == ["c", "e"]
    assert result["retrieval_status"] == "ok"
    assert result["limits_applied"] is False


def test_retrieval_returns_only_endpoints_of_the_final_relation_step():
    graph = build_temporal_graph(
        [
            {
                "fact_id": "f_ab",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "b",
                "timestamp": "2014-01-01",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_bc",
                "subject_id": "b",
                "relation_id": "r1",
                "object_id": "c",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
        ]
    )
    query = {
        "qa_id": "repeated-relation",
        "question": "Which entity follows two r1 relations?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r1"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-03",
        },
    }

    result = retrieve_subgraph(graph, query, graph_id="fixture-v1")

    assert result["candidate_answer_ids"] == ["c"]


@pytest.mark.parametrize(
    ("seed_ids", "expected_status", "expected_missing"),
    [
        (["not-in-graph"], "missing_entity", ["not-in-graph"]),
        (["a"], "empty_result", []),
    ],
)
def test_retrieval_reports_missing_seeds_and_no_matching_evidence(
    temporal_graph, two_hop_query, seed_ids, expected_status, expected_missing
):
    query = {
        **two_hop_query,
        "seed_entity_ids": seed_ids,
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2013-12-31",
        },
    }
    if seed_ids == ["a"]:
        query["relation_sequence"] = ["r1"]

    result = retrieve_subgraph(temporal_graph, query, graph_id="fixture-v1")

    assert result["facts"] == []
    assert result["candidate_answer_ids"] == []
    assert result["retrieval_status"] == expected_status
    assert result["missing_entity_ids"] == expected_missing


@pytest.mark.parametrize(
    ("settings", "expected_fact_ids", "expected_answers"),
    [
        ({"max_nodes": 2}, ["f_ab"], []),
        ({"max_edges": 1}, ["f_ab"], []),
    ],
)
def test_retrieval_applies_node_and_edge_limits(
    temporal_graph, two_hop_query, settings, expected_fact_ids, expected_answers
):
    result = retrieve_subgraph(
        temporal_graph,
        two_hop_query,
        graph_id="fixture-v1",
        settings=settings,
    )

    assert [fact["fact_id"] for fact in result["facts"]] == expected_fact_ids
    assert result["candidate_answer_ids"] == expected_answers
    assert result["limits_applied"] is True


def test_retrieval_rejects_limits_that_cannot_cover_query(temporal_graph, two_hop_query):
    with pytest.raises(ValueError, match="max_hops must be at least"):
        retrieve_subgraph(
            temporal_graph,
            two_hop_query,
            graph_id="fixture-v1",
            settings={"max_hops": 1},
        )


def test_retrieval_cache_key_is_stable_for_equivalent_inputs(two_hop_query):
    base = retrieval_cache_key(
        "dataset-v1", "graph-v1", "train", two_hop_query
    )
    equivalent = retrieval_cache_key(
        " dataset-v1 ",
        "graph-v1",
        "train",
        {**two_hop_query, "seed_entity_ids": ["a"]},
        settings={
            "max_edges": 256,
            "max_nodes": 128,
            "max_hops": 2,
            "strategy": "relation_constrained_bfs",
        },
    )

    assert equivalent == base


@pytest.mark.parametrize(
    ("dataset_version", "graph_version", "split", "query_changes", "settings"),
    [
        ("dataset-v2", "graph-v1", "train", {}, None),
        ("dataset-v1", "graph-v2", "train", {}, None),
        ("dataset-v1", "graph-v1", "validation", {}, None),
        (
            "dataset-v1",
            "graph-v1",
            "train",
            {"temporal_constraint": {"operator": "on_or_before", "cutoff_date": "2014-01-03"}},
            None,
        ),
        ("dataset-v1", "graph-v1", "train", {"seed_entity_ids": ["b"]}, None),
        ("dataset-v1", "graph-v1", "train", {"relation_sequence": ["r1"]}, None),
        ("dataset-v1", "graph-v1", "train", {}, {"max_edges": 1}),
    ],
)
def test_retrieval_cache_key_isolates_versions_split_query_and_settings(
    two_hop_query, dataset_version, graph_version, split, query_changes, settings
):
    baseline = retrieval_cache_key(
        "dataset-v1", "graph-v1", "train", two_hop_query
    )
    key = retrieval_cache_key(
        dataset_version,
        graph_version,
        split,
        {**two_hop_query, **query_changes},
        settings=settings,
    )

    assert key != baseline


def test_retrieval_cache_key_isolates_example_id(two_hop_query):
    default_key = retrieval_cache_key(
        "dataset-v1", "graph-v1", "train", two_hop_query
    )
    distinct_example_key = retrieval_cache_key(
        "dataset-v1",
        "graph-v1",
        "train",
        two_hop_query,
        example_id="query-1-copy",
    )

    assert distinct_example_key != default_key


def test_retrieval_cache_key_rejects_split_mismatch(two_hop_query):
    with pytest.raises(ValueError, match="match qa_record split"):
        retrieval_cache_key(
            "dataset-v1",
            "graph-v1",
            "validation",
            {**two_hop_query, "split": "train"},
        )


def test_retrieval_cache_key_rejects_invalid_split(two_hop_query):
    with pytest.raises(ValueError, match="split must be"):
        retrieval_cache_key("dataset-v1", "graph-v1", [], two_hop_query)


def test_support_diagnostics_separate_retrieval_miss_from_reasoning_error(
    temporal_graph, two_hop_query
):
    complete_retrieval = retrieve_subgraph(
        temporal_graph, two_hop_query, graph_id="fixture-v1"
    )
    incomplete_retrieval = retrieve_subgraph(
        temporal_graph,
        two_hop_query,
        graph_id="fixture-v1",
        settings={"max_edges": 1},
    )
    qa_record = {
        **two_hop_query,
        "accepted_answer_ids": ["c"],
        "supporting_path_fact_ids": ["f_ab", "f_bc"],
    }

    retrieval_miss = assess_retrieval_support(
        qa_record,
        incomplete_retrieval,
        {
            "qa_id": qa_record["qa_id"],
            "status": "ok",
            "predicted_answer_ids": ["c"],
        },
    )
    reasoning_error = assess_retrieval_support(
        qa_record,
        complete_retrieval,
        {
            "qa_id": qa_record["qa_id"],
            "status": "ok",
            "predicted_answer_ids": ["b"],
        },
    )

    assert retrieval_miss["supporting_path_coverage"] == 0.5
    assert retrieval_miss["missing_supporting_path_fact_ids"] == ["f_bc"]
    assert retrieval_miss["failure_category"] == "retrieval_miss"
    assert reasoning_error["supporting_path_complete"] is True
    assert reasoning_error["failure_category"] == "reasoning_error"

    summary = summarize_support_diagnostics([retrieval_miss, reasoning_error])
    assert summary["supporting_path_coverage"] == 0.5
    assert summary["supporting_path_edge_coverage"] == 0.75
    assert summary["retrieval_miss_rate"] == 0.5
    assert summary["reasoning_error_rate"] == 0.5


def test_support_diagnostics_keep_model_errors_separate(
    temporal_graph, two_hop_query
):
    retrieval = retrieve_subgraph(
        temporal_graph, two_hop_query, graph_id="fixture-v1"
    )
    qa_record = {
        **two_hop_query,
        "accepted_answer_ids": ["c"],
        "supporting_path_fact_ids": ["f_ab", "f_bc"],
    }

    diagnostic = assess_retrieval_support(
        qa_record,
        retrieval,
        {
            "qa_id": qa_record["qa_id"],
            "status": "model_error",
            "predicted_answer_ids": [],
        },
    )

    assert diagnostic["supporting_path_complete"] is True
    assert diagnostic["failure_category"] == "model_error"
