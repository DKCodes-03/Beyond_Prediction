import pytest

from src.evaluation.evaluate_explanations import evaluate_explanations
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
                "source_id": "source:ab",
            },
            {
                "fact_id": "f_bc",
                "subject_id": "b",
                "relation_id": "r2",
                "object_id": "c",
                "timestamp": "2014-01-02",
                "source_id": "source:bc",
            },
            {
                "fact_id": "f_bd",
                "subject_id": "b",
                "relation_id": "r2",
                "object_id": "d",
                "timestamp": "2014-01-03",
                "source_id": "source:bd",
            },
        ]
    )


@pytest.fixture
def qa_record():
    return {
        "qa_id": "qa-1",
        "question": "Which entity follows r1 and then r2?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
        "accepted_answer_ids": ["c"],
    }


def _claim(fact_id, subject_id, relation_id, object_id, timestamp):
    claim = {
        "subject_id": subject_id,
        "relation_id": relation_id,
        "object_id": object_id,
        "timestamp": timestamp,
    }
    if fact_id is not None:
        claim["fact_id"] = fact_id
    return claim


def test_evaluates_supported_temporal_path(graph, qa_record):
    explanation = {
        "qa_id": "qa-1",
        "answer_id": "c",
        "supporting_path": [
            _claim(None, "a", "r1", "b", "2014-01-02"),
            _claim(None, "b", "r2", "c", "2014-01-02"),
        ],
    }

    metrics = evaluate_explanations(graph, [qa_record], [explanation])

    assert metrics["factual_edge_support"] == 1.0
    assert metrics["temporal_consistency"] == 1.0
    assert metrics["supporting_path_validity"] == 1.0
    assert metrics["per_explanation"][0]["path_failure_reasons"] == []


def test_separates_graph_support_from_temporal_consistency(graph, qa_record):
    qa_record["temporal_constraint"] = {
        "operator": "on_or_before",
        "cutoff_date": "2014-01-01",
    }
    explanation = {
        "qa_id": "qa-1",
        "answer_id": "c",
        "supporting_path": [
            _claim("f_ab", "a", "r1", "b", "2014-01-02"),
            _claim("f_bc", "b", "r2", "c", "2014-01-02"),
        ],
    }

    metrics = evaluate_explanations(graph, [qa_record], [explanation])

    assert metrics["factual_edge_support"] == 1.0
    assert metrics["temporal_consistency"] == 0.0
    assert metrics["supporting_path_validity"] == 0.0
    assert metrics["per_explanation"][0]["path_failure_reasons"] == [
        "temporal_violation"
    ]


def test_flags_unsupported_edges_even_when_path_is_well_formed(graph, qa_record):
    explanation = {
        "qa_id": "qa-1",
        "answer_id": "c",
        "supporting_path": [
            _claim("invented", "a", "r1", "b", "2014-01-02"),
            _claim("f_bc", "b", "r2", "c", "2014-01-02"),
        ],
    }

    metrics = evaluate_explanations(graph, [qa_record], [explanation])

    assert metrics["factual_edge_support"] == 0.5
    assert metrics["temporal_consistency"] == 1.0
    assert metrics["supporting_path_validity"] == 0.0
    assert metrics["per_explanation"][0]["path_failure_reasons"] == [
        "unsupported_edge"
    ]


def test_rejects_missing_explanation_records(graph, qa_record):
    with pytest.raises(ValueError, match="missing explanations"):
        evaluate_explanations(graph, [qa_record], [])


def test_rejects_duplicate_explanations(graph, qa_record):
    explanation = {"qa_id": "qa-1", "answer_id": "c", "supporting_path": []}

    with pytest.raises(ValueError, match="duplicate explanation"):
        evaluate_explanations(graph, [qa_record], [explanation, explanation])
