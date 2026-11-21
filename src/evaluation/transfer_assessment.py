"""Isolated compatibility and transfer assessment for ICEWS18.

ICEWS18 is optional and deliberately remains outside the learned ICEWS14
protocol unless matching data, mappings, and a separate comparison are supplied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.metrics import compute_answer_metrics
from src.evaluation.path_baseline import graph_path_sanity_baseline
from src.graph.graph_builder import build_temporal_graph
from src.graph.graph_utils import validate_fact


def assess_icews18_transfer(
    source_facts: Sequence[Mapping[str, Any]] | None = None,
    target_facts: Sequence[Mapping[str, Any]] | None = None,
    target_qa_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    icews14_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_qa_records: Sequence[Mapping[str, Any]] | None = None,
    entity_mapping: Mapping[str, str] | None = None,
    relation_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a data-compatibility report for a possible ICEWS18 transfer."""
    if source_facts is not None:
        if icews14_facts is not None:
            raise ValueError("provide source_facts or icews14_facts, not both")
        icews14_facts = source_facts
    if target_facts is not None:
        if icews18_facts is not None:
            raise ValueError("provide target_facts or icews18_facts, not both")
        icews18_facts = target_facts
    if target_qa_records is not None:
        if icews18_qa_records is not None:
            raise ValueError(
                "provide target_qa_records or icews18_qa_records, not both"
            )
        icews18_qa_records = target_qa_records
    if entity_mapping is None:
        entity_mapping = {}
    if relation_mapping is None:
        relation_mapping = {}
    if not isinstance(entity_mapping, Mapping):
        raise ValueError("entity_mapping must be a mapping")
    if not isinstance(relation_mapping, Mapping):
        raise ValueError("relation_mapping must be a mapping")

    source_stats = _fact_stats(icews14_facts, "ICEWS14")
    target_stats = _fact_stats(icews18_facts, "ICEWS18")

    mapped_entities = _validate_mapping(entity_mapping, "entity")
    mapped_relations = _validate_mapping(relation_mapping, "relation")
    source_entity_ids = set(source_stats["entity_ids"])
    source_relation_ids = set(source_stats["relation_ids"])

    report: dict[str, Any] = {
        "benchmark": "optional_icews18_transfer",
        "status": "assessed",
        "source": {"dataset_id": "ICEWS14", **source_stats},
        "target": {"dataset_id": "ICEWS18", **target_stats},
        "mapping": {
            "entity_mapping_count": len(mapped_entities),
            "relation_mapping_count": len(mapped_relations),
            "source_entities_mapped": len(
                source_entity_ids.intersection(mapped_entities)
            ),
            "source_relations_mapped": len(
                source_relation_ids.intersection(mapped_relations)
            ),
            "source_entity_coverage": (
                len(source_entity_ids.intersection(mapped_entities))
                / len(source_entity_ids)
                if source_entity_ids
                else None
            ),
            "source_relation_coverage": (
                len(source_relation_ids.intersection(mapped_relations))
                / len(source_relation_ids)
                if source_relation_ids
                else None
            ),
        },
        "learned_transfer_performed": False,
        "core_dataset_adoption": False,
        "limitations": [
            "No learned ICEWS14 checkpoint is reused by this assessment.",
            "Synthetic QA results are a target-data protocol sanity check, "
            "not evidence of learned cross-dataset generalization.",
            "ICEWS18 is not adopted unless a separate matched, mapped comparison "
            "passes the frozen protocol gates.",
        ],
    }

    if icews18_facts is not None and icews18_qa_records is not None:
        if not isinstance(icews18_qa_records, (list, tuple)):
            raise ValueError("icews18_qa_records must be a sequence")
        if not icews18_qa_records:
            raise ValueError("icews18_qa_records must be non-empty")
        if any(not isinstance(record, Mapping) for record in icews18_qa_records):
            raise ValueError("every ICEWS18 QA record must be a mapping")

        graph = build_temporal_graph(icews18_facts)
        predictions: list[dict[str, Any]] = []
        for qa_record in icews18_qa_records:
            result = graph_path_sanity_baseline(graph, qa_record)
            predictions.append(
                {
                    "qa_id": result["qa_id"],
                    "model_variant": "graph_path_sanity",
                    "status": result["status"],
                    "predicted_answer_ids": result["predicted_answer_ids"],
                    "answer_scores": [],
                }
            )
        report["target_graph_sanity"] = {
            "metrics": compute_answer_metrics(
                icews18_qa_records,
                predictions,
            ),
            "prediction_count": len(predictions),
        }
    else:
        report["status"] = "deferred"
        report["deferred_reason"] = (
            "ICEWS18 facts and graph-verified QA records were not supplied"
        )

    return report


def run_icews18_transfer_assessment(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an ICEWS18 assessment atomically and return its artifact paths."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "icews18_transfer_assessment.json"
    temporary_path = destination / ".icews18_transfer_assessment.json.tmp"
    temporary_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return {"report_path": str(report_path), "status": report.get("status")}


def _fact_stats(
    facts: Sequence[Mapping[str, Any]] | None,
    dataset_name: str,
) -> dict[str, Any]:
    if facts is None:
        return {
            "supplied": False,
            "fact_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "entity_ids": [],
            "relation_ids": [],
        }
    if not isinstance(facts, (list, tuple)):
        raise ValueError(f"{dataset_name} facts must be a sequence")

    entity_ids: set[str] = set()
    relation_ids: set[str] = set()
    for index, record in enumerate(facts):
        if not isinstance(record, Mapping):
            raise ValueError(f"{dataset_name} fact {index} must be a mapping")
        fact = validate_fact(record, index)
        entity_ids.update((fact["subject_id"], fact["object_id"]))
        relation_ids.add(fact["relation_id"])

    return {
        "supplied": True,
        "fact_count": len(facts),
        "entity_count": len(entity_ids),
        "relation_count": len(relation_ids),
        "entity_ids": sorted(entity_ids),
        "relation_ids": sorted(relation_ids),
    }


def _validate_mapping(
    mapping: Mapping[str, str],
    kind: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for source_id, target_id in mapping.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{kind}_mapping keys must be non-empty strings")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError(
                f"{kind}_mapping values must be non-empty strings"
            )
        normalized[source_id.strip()] = target_id.strip()
    return normalized


# Descriptive aliases for scripts and external callers.
assess_icews18 = assess_icews18_transfer
run_transfer_assessment = run_icews18_transfer_assessment
