import json

import pytest

from src.evaluation.benchmark_verification import (
    independently_verify_benchmark,
    run_benchmark_verification,
)


@pytest.fixture
def facts():
    return [
        {
            "fact_id": "f_ab",
            "subject_id": "a",
            "relation_id": "r1",
            "object_id": "b",
            "timestamp": "2014-01-01",
            "source_id": "fixture",
        },
        {
            "fact_id": "f_ab_later",
            "subject_id": "a",
            "relation_id": "r1",
            "object_id": "b",
            "timestamp": "2014-01-03",
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
            "timestamp": "2014-01-02",
            "source_id": "fixture",
        },
        {
            "fact_id": "f_c_future",
            "subject_id": "c",
            "relation_id": "r3",
            "object_id": "future",
            "timestamp": "2014-01-10",
            "source_id": "fixture",
        },
    ]


@pytest.fixture
def qa_record():
    return {
        "qa_id": "q1",
        "question": "Which entity follows r1 and r2?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
        "accepted_answer_ids": ["c", "d"],
        "supporting_path_fact_ids": ["f_ab", "f_bc"],
        "provenance": {
            "supporting_fact_ids": ["f_ab", "f_bc"],
        },
    }


def test_verification_finds_all_answers_paths_and_temporal_boundaries(
    facts, qa_record
):
    report = independently_verify_benchmark(facts, [qa_record])

    assert report["status"] == "verified"
    assert report["qa_count"] == 1
    assert report["answer_set_coverage"]["coverage_rate"] == 1.0
    assert report["temporal_boundary_coverage"]["coverage_rate"] == 1.0
    assert report["candidate_coverage"]["gold_answer_recall"] == 1.0
    assert report["supporting_evidence_coverage"]["alternative_valid_paths"] == 2
    record = report["records"][0]
    assert record["graph_answer_ids"] == ["c", "d"]
    assert record["declared_supporting_path_valid"] is True
    assert record["graph_path_sanity"]["matches_accepted_answers"] is True


def test_compact_verification_preserves_aggregate_metrics(facts, qa_record):
    full = independently_verify_benchmark(facts, [qa_record])
    compact = independently_verify_benchmark(
        facts,
        [qa_record],
        include_records=False,
    )

    assert compact["status"] == full["status"]
    assert compact["answer_set_coverage"] == full["answer_set_coverage"]
    assert compact["temporal_boundary_coverage"] == full["temporal_boundary_coverage"]
    assert compact["candidate_coverage"] == full["candidate_coverage"]
    assert compact["supporting_evidence_coverage"] == full["supporting_evidence_coverage"]
    assert compact["graph_path_sanity"] == full["graph_path_sanity"]
    assert compact["record_details_included"] is False
    assert "records" not in compact


def test_verification_reports_mismatched_answer_set(facts, qa_record):
    qa_record["accepted_answer_ids"] = ["c"]

    report = independently_verify_benchmark(facts, [qa_record])

    assert report["status"] == "failed"
    assert report["answer_set_coverage"]["matching_examples"] == 0
    assert report["records"][0]["graph_answer_ids"] == ["c", "d"]


def test_verification_rejects_future_supporting_edge(facts, qa_record):
    future_qa = {
        "qa_id": "future_q",
        "question": "Which entity follows r3?",
        "seed_entity_ids": ["c"],
        "relation_sequence": ["r3"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
        "accepted_answer_ids": ["future"],
        "supporting_path_fact_ids": ["f_c_future"],
        "provenance": {"supporting_fact_ids": ["f_c_future"]},
    }

    report = independently_verify_benchmark(facts, [future_qa])

    assert report["answer_set_coverage"]["coverage_rate"] == 0.0
    assert report["temporal_boundary_coverage"]["coverage_rate"] == 0.0
    assert report["records"][0]["temporal_violation_fact_ids"] == ["f_c_future"]


def test_verification_rejects_invalid_input():
    with pytest.raises(ValueError, match="non-empty sequence"):
        independently_verify_benchmark([], [])


def test_verification_export_is_atomic(tmp_path, facts, qa_record):
    result = run_benchmark_verification(
        facts=facts,
        qa_records=[qa_record],
        output_dir=tmp_path,
    )
    saved = json.loads(
        (tmp_path / "benchmark_verification.json").read_text(encoding="utf-8")
    )

    assert saved["status"] == "verified"
    assert result["report_path"].endswith("benchmark_verification.json")
