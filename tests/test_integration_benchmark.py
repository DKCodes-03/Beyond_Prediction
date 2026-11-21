import json
from pathlib import Path

import pytest

from src.evaluation.transfer_assessment import (
    assess_icews18_transfer,
    run_icews18_transfer_assessment,
)
from src.evaluation.integration_benchmark import (
    benchmark_icews18_transfer,
    benchmark_adopted_integrations,
    benchmark_ingestion,
    benchmark_qwen,
    run_integration_benchmark,
)


def test_ingestion_benchmark_is_separate_from_extraction_metrics(tmp_path):
    txt_path = tmp_path / "facts.txt"
    txt_path.write_text(
        "Alpha|met|Beta|2014-01-01\n",
        encoding="utf-8",
    )

    report = benchmark_ingestion(
        {"txt": txt_path},
        extracted_text="Alpha|met|Beta|2014-01-01\n",
        references=[
            {
                "subject": "Alpha",
                "relation": "met",
                "object": "Beta",
                "timestamp": "2014-01-01",
            }
        ],
    )

    assert report["benchmark"] == "optional_ingestion"
    assert report["formats"]["txt"]["record_count"] == 1
    assert report["text_extraction"]["precision"] == 1.0
    assert report["text_extraction"]["recall"] == 1.0
    assert report["text_extraction"]["f1"] == 1.0


def test_qwen_benchmark_constructs_prompt_without_loading_model():
    report = benchmark_qwen(
        "Which entity follows Alpha?",
        [{"subject": "Alpha", "relation": "met", "object": "Beta"}],
        ["Beta"],
    )

    assert report["benchmark"] == "optional_qwen"
    assert report["model_loaded"] is False
    assert report["inference_performed"] is False
    assert report["prompt_chars"] > 0


class _FakeSession:
    def __init__(self):
        self.calls = []
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def run(self, query, parameters=None):
        self.calls.append((query, parameters or {}))

        class Result:
            def __iter__(self):
                return iter(self.records)

        result = Result()
        result.records = self.records
        return result


class _FakeNeo4jAdapter:
    def __init__(self):
        self.persisted = []

    def persist_temporal_graph(self, graph):
        self.persisted = [dict(fact) for fact in graph.facts]
        return len(self.persisted)

    def load_facts(self):
        return list(self.persisted)


def test_combined_benchmark_runs_neo4j_and_qwen_separately(tmp_path):
    facts = [
        {
            "fact_id": "f1",
            "subject_id": "Alpha",
            "relation_id": "met",
            "object_id": "Beta",
            "timestamp": "2014-01-01",
            "source_id": "fixture",
        }
    ]
    report = benchmark_adopted_integrations(
        neo4j_adapter=_FakeNeo4jAdapter(),
        facts=facts,
        question="Which entity follows Alpha?",
        evidence=facts,
        answer_ids=["Beta"],
    )

    assert set(report["components"]) == {"neo4j", "qwen"}
    assert report["components"]["neo4j"]["round_trip_match"] is True
    assert report["components"]["qwen"]["model_loaded"] is False

    result = run_integration_benchmark(
        output_dir=tmp_path,
        report=report,
    )
    saved = json.loads(
        (tmp_path / "integration_benchmark.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert result["component_count"] == 2


def test_combined_benchmark_requires_a_selected_component():
    with pytest.raises(ValueError, match="at least one"):
        benchmark_adopted_integrations()


def test_icews18_assessment_is_deferred_when_data_is_not_supplied():
    report = benchmark_icews18_transfer()

    assert report["benchmark"] == "optional_icews18_transfer"
    assert report["status"] == "deferred"
    assert report["source"]["supplied"] is False
    assert report["target"]["supplied"] is False
    assert report["learned_transfer_performed"] is False
    assert report["core_dataset_adoption"] is False


def test_icews18_assessment_checks_mapping_and_writes_report(tmp_path):
    source_facts = [
        {
            "fact_id": "s1",
            "subject_id": "1",
            "relation_id": "10",
            "object_id": "2",
            "timestamp": "2014-01-01",
            "source_id": "ICEWS14",
        }
    ]
    target_facts = [
        {
            "fact_id": "t1",
            "subject_id": "A",
            "relation_id": "met",
            "object_id": "B",
            "timestamp": "2018-01-01",
            "source_id": "ICEWS18",
        }
    ]
    target_qa = [
        {
            "qa_id": "q1",
            "question": "Which entity is connected to A?",
            "seed_entity_ids": ["A"],
            "relation_sequence": ["met"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2018-01-01",
            },
            "accepted_answer_ids": ["B"],
            "supporting_path_fact_ids": ["t1"],
        }
    ]

    report = assess_icews18_transfer(
        source_facts,
        target_facts,
        target_qa,
        entity_mapping={"1": "A", "2": "B"},
        relation_mapping={"10": "met"},
    )

    assert report["status"] == "assessed"
    assert report["mapping"]["source_entity_coverage"] == 1.0
    assert report["mapping"]["source_relation_coverage"] == 1.0
    assert report["target_graph_sanity"]["prediction_count"] == 1
    assert report["target_graph_sanity"]["metrics"]["exact_match_accuracy"] == 1.0

    written = run_icews18_transfer_assessment(
        output_dir=tmp_path,
        report=report,
    )
    saved = json.loads(
        (tmp_path / "icews18_transfer_assessment.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == report
    assert written["status"] == "assessed"
