"""Independent verification for graph-derived benchmark answers and evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.path_baseline import graph_path_sanity_baseline
from src.graph.graph_builder import build_temporal_graph
from src.graph.graph_utils import canonical_id
from src.retrieval.subgraph_retriever import (
    _validate_query,
    retrieve_subgraph,
)
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


def independently_verify_benchmark(
    facts: Sequence[Mapping[str, Any]],
    qa_records: Iterable[Mapping[str, Any]],
    *,
    retrieval_settings: Mapping[str, Any] | None = None,
    include_records: bool = True,
) -> dict[str, Any]:
    """Independently enumerate paths and verify benchmark answer/evidence contracts."""
    if not isinstance(facts, (list, tuple)):
        raise ValueError("facts must be a sequence")
    if isinstance(qa_records, (str, bytes, Mapping)):
        raise ValueError("qa_records must be a non-empty iterable of mappings")
    try:
        qa_iterator = iter(qa_records)
    except TypeError as exc:
        raise ValueError("qa_records must be a non-empty iterable of mappings") from exc
    if isinstance(qa_records, Sequence) and not qa_records:
        raise ValueError("qa_records must be a non-empty sequence")
    if retrieval_settings is not None and not isinstance(retrieval_settings, Mapping):
        raise ValueError("retrieval_settings must be a mapping")
    if not isinstance(include_records, bool):
        raise ValueError("include_records must be a boolean")

    graph = build_temporal_graph(facts)
    records: list[dict[str, Any]] = []
    answer_set_matches = 0
    temporal_valid_count = 0
    candidate_exact_matches = 0
    supporting_complete = 0
    all_gold_answer_examples = 0
    total_gold = 0
    covered_gold = 0
    sanity_matches = 0
    alternative_valid_paths = 0
    count = 0

    for index, qa_record in enumerate(qa_iterator):
        count += 1
        if not isinstance(qa_record, Mapping):
            raise ValueError(f"QA record {index} must be a mapping")
        query = _validate_query(qa_record)

        accepted_values = qa_record.get("accepted_answer_ids")
        if not isinstance(accepted_values, (list, tuple)) or not accepted_values:
            raise ValueError(
                f"QA record {index} accepted_answer_ids must be a non-empty list"
            )
        accepted = sorted(
            {canonical_id(value, "accepted_answer_id") for value in accepted_values}
        )

        valid_paths = _enumerate_valid_paths(graph, query)
        answer_paths: dict[str, int] = {}
        valid_fact_ids: set[str] = set()
        for path_fact_ids in valid_paths:
            if not path_fact_ids:
                continue
            endpoint = graph.get_fact(path_fact_ids[-1])["object_id"]
            answer_paths[endpoint] = answer_paths.get(endpoint, 0) + 1
            valid_fact_ids.update(path_fact_ids)

        graph_answers = sorted(answer_paths)
        answer_set_exact = accepted == graph_answers

        temporal_valid = True
        cutoff_mismatches: list[str] = []
        for path_fact_ids in valid_paths:
            for fact_id in path_fact_ids:
                fact = graph.get_fact(fact_id)
                if fact is None:
                    temporal_valid = False
                    cutoff_mismatches.append(fact_id)
                    continue
                violating = _temporal_violation(fact, query["temporal_constraint"])
                if violating:
                    temporal_valid = False
                    cutoff_mismatches.append(fact_id)

        retrieved = retrieve_subgraph(
            graph,
            qa_record,
            graph_id="benchmark_verification",
            settings=retrieval_settings,
        )
        candidate_ids = sorted(set(retrieved["candidate_answer_ids"]))
        candidate_answer_coverage = len(set(accepted).intersection(candidate_ids))
        candidate_coverage = (
            candidate_answer_coverage / len(accepted) if accepted else 0.0
        )

        declared_support = qa_record.get("supporting_path_fact_ids")
        declared_valid = isinstance(declared_support, (list, tuple)) and bool(declared_support)
        if declared_valid:
            declared_support_ids = [
                canonical_id(value, "supporting_path_fact_id")
                for value in declared_support
            ]
            declared_valid = _validate_declared_path(
                graph,
                query,
                declared_support_ids,
            )
        else:
            declared_support_ids = [
                canonical_id(value, "supporting_path_fact_id")
                for value in declared_support
            ] if isinstance(declared_support, (list, tuple)) else []

        for fact_id in declared_support_ids:
            fact = graph.get_fact(fact_id)
            if fact is not None and _temporal_violation(
                fact,
                query["temporal_constraint"],
            ):
                temporal_valid = False
                cutoff_mismatches.append(fact_id)

        provenance = qa_record.get("provenance")
        provenance_fact_ids: list[str] = []
        if isinstance(provenance, Mapping):
            values = provenance.get("supporting_fact_ids")
            if isinstance(values, (list, tuple)):
                provenance_fact_ids = sorted(
                    {
                        canonical_id(value, "supporting_fact_id")
                        for value in values
                    }
                )
        supporting_evidence_complete = (
            bool(valid_fact_ids)
            and bool(provenance_fact_ids)
            and all(
                fact_id in valid_fact_ids for fact_id in provenance_fact_ids
            )
        )

        sanity = graph_path_sanity_baseline(graph, qa_record)
        alternative_path_count = sum(answer_paths.values())
        sanity_matches_accepted = (
            sorted(set(sanity["predicted_answer_ids"])) == accepted
        )
        record_temporal_valid = temporal_valid
        answer_set_matches += int(answer_set_exact)
        temporal_valid_count += int(record_temporal_valid)
        candidate_set_exact = accepted == candidate_ids
        candidate_exact_matches += int(candidate_set_exact)
        supporting_complete += int(
            supporting_evidence_complete and declared_valid
        )
        all_gold_answer_examples += int(candidate_answer_coverage == len(accepted))
        total_gold += len(accepted)
        covered_gold += candidate_answer_coverage
        sanity_matches += int(sanity_matches_accepted)
        alternative_valid_paths += alternative_path_count

        if include_records:
            records.append(
                {
                    "qa_id": query["qa_id"],
                    "accepted_answer_ids": accepted,
                    "graph_answer_ids": graph_answers,
                    "answer_set_exact_match": answer_set_exact,
                    "temporal_boundary_valid": record_temporal_valid,
                    "temporal_cutoff": query["temporal_constraint"].get("cutoff_date"),
                    "temporal_violation_fact_ids": sorted(set(cutoff_mismatches)),
                    "candidate_answer_ids": candidate_ids,
                    "candidate_answer_set_exact_match": candidate_set_exact,
                    "covered_gold_answers": candidate_answer_coverage,
                    "gold_answer_recall": candidate_coverage,
                    "declared_supporting_path_valid": declared_valid,
                    "declared_supporting_path_fact_ids": (
                        declared_support_ids if declared_valid else list(declared_support or [])
                    ),
                    "alternative_valid_path_count": alternative_path_count,
                    "provenance_supporting_fact_ids": provenance_fact_ids,
                    "supporting_evidence_complete": supporting_evidence_complete,
                    "valid_supporting_fact_ids": sorted(valid_fact_ids),
                    "graph_path_sanity": {
                        "status": sanity["status"],
                        "predicted_answer_ids": sanity["predicted_answer_ids"],
                        "matches_accepted_answers": sanity_matches_accepted,
                        "witness_paths": sanity["witness_paths"],
                    },
                }
            )

    if count == 0:
        raise ValueError("qa_records must be a non-empty iterable of mappings")

    report = {
        "benchmark": "benchmark_answer_verification",
        "status": (
            "verified"
            if (
                answer_set_matches == count
                and temporal_valid_count == count
                and supporting_complete == count
                and sanity_matches == count
            )
            else "failed"
        ),
        "qa_count": count,
        "answer_set_coverage": {
            "matching_examples": answer_set_matches,
            "coverage_rate": answer_set_matches / count,
        },
        "answer_set_verification": {
            "matching_examples": answer_set_matches,
            "coverage_rate": answer_set_matches / count,
        },
        "temporal_boundary_coverage": {
            "valid_examples": temporal_valid_count,
            "coverage_rate": temporal_valid_count / count,
        },
        "temporal_boundary_verification": {
            "valid_examples": temporal_valid_count,
            "coverage_rate": temporal_valid_count / count,
        },
        "candidate_coverage": {
            "examples_with_all_gold_answers": all_gold_answer_examples,
            "all_gold_answer_coverage_rate": all_gold_answer_examples / count,
            "exact_candidate_set_examples": candidate_exact_matches,
            "exact_candidate_set_rate": candidate_exact_matches / count,
            "total_gold_answers": total_gold,
            "covered_gold_answers": covered_gold,
            "gold_answer_recall": covered_gold / total_gold if total_gold else 0.0,
        },
        "candidate_answer_coverage": {
            "examples_with_all_gold_answers": all_gold_answer_examples,
            "all_gold_answer_coverage_rate": all_gold_answer_examples / count,
            "gold_answer_recall": covered_gold / total_gold if total_gold else 0.0,
        },
        "supporting_evidence_coverage": {
            "complete_examples": supporting_complete,
            "coverage_rate": supporting_complete / count,
            "alternative_valid_paths": alternative_valid_paths,
        },
        "supporting_evidence": {
            "complete_examples": supporting_complete,
            "coverage_rate": supporting_complete / count,
            "alternative_valid_paths": alternative_valid_paths,
        },
        "graph_path_sanity": {
            "matching_examples": sanity_matches,
            "coverage_rate": sanity_matches / count,
        },
        "record_details_included": include_records,
    }
    if include_records:
        report["records"] = records
    return report


verify_benchmark_answer_sets = independently_verify_benchmark
verify_benchmark_answers = independently_verify_benchmark
verify_benchmark = independently_verify_benchmark
benchmark_answer_verification = independently_verify_benchmark


def run_benchmark_verification(
    *,
    facts: Sequence[Mapping[str, Any]],
    qa_records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    retrieval_settings: Mapping[str, Any] | None = None,
    include_records: bool = True,
) -> dict[str, Any]:
    """Write an independently computed verification report atomically."""
    report = independently_verify_benchmark(
        facts,
        qa_records,
        retrieval_settings=retrieval_settings,
        include_records=include_records,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "benchmark_verification.json"
    temporary_path = destination / ".benchmark_verification.json.tmp"
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return {
        "report_path": str(report_path),
        "benchmark": report["benchmark"],
        "status": report["status"],
        "qa_count": report["qa_count"],
        "record_details_included": report["record_details_included"],
    }


run_benchmark_answer_verification = run_benchmark_verification


def _enumerate_valid_paths(
    graph: Any,
    query: Mapping[str, Any],
) -> list[list[str]]:
    """Enumerate every relation-constrained, temporally valid fact-ID path."""
    paths: set[tuple[str, ...]] = set()

    def walk(entity_id: str, hop: int, path: tuple[str, ...]) -> None:
        if hop == len(query["relation_sequence"]):
            if path:
                paths.add(path)
            return
        relation_id = query["relation_sequence"][hop]
        outgoing = graph.outgoing_facts(entity_id, relation_id)
        outgoing = filter_facts_by_temporal_constraint(
            outgoing,
            query["temporal_constraint"],
        )
        for fact in outgoing:
            walk(
                fact["object_id"],
                hop + 1,
                (*path, fact["fact_id"]),
            )

    for seed_id in query["seed_entity_ids"]:
        walk(seed_id, 0, ())
    return [list(path) for path in sorted(paths)]


def _validate_declared_path(
    graph: Any,
    query: Mapping[str, Any],
    fact_ids: Sequence[str],
) -> bool:
    if len(fact_ids) != len(query["relation_sequence"]):
        return False
    current_entity_ids = set(query["seed_entity_ids"])
    for hop, fact_id in enumerate(fact_ids):
        fact = graph.get_fact(fact_id)
        if fact is None:
            return False
        if fact["subject_id"] not in current_entity_ids:
            return False
        if fact["relation_id"] != query["relation_sequence"][hop]:
            return False
        if _temporal_violation(fact, query["temporal_constraint"]):
            return False
        current_entity_ids = {fact["object_id"]}
    return True


def _temporal_violation(
    fact: Mapping[str, Any],
    temporal_constraint: Mapping[str, Any],
) -> bool:
    allowed = filter_facts_by_temporal_constraint(
        (fact,),
        temporal_constraint,
    )
    return len(allowed) != 1
