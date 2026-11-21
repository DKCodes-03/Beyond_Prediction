"""Evaluate structured explanation claims against temporal graph evidence."""

from collections.abc import Mapping, Sequence
from typing import Any

from src.graph.graph_utils import canonical_id, normalize_date
from src.graph.temporal_graph import TemporalGraph
from src.retrieval.subgraph_retriever import _validate_query
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


def evaluate_explanations(
    graph: TemporalGraph,
    qa_records: Sequence[Mapping[str, Any]],
    explanation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure graph support, temporal consistency, and path validity.

    Each explanation record represents one QA answer and supplies a structured
    ``supporting_path``: an ordered list of edge claims with subject, relation,
    object, and timestamp fields. ``fact_id`` may also be supplied to bind a
    claim to one particular graph fact. This evaluates factual claims, not
    natural-language fluency or semantic faithfulness.
    """
    if not isinstance(qa_records, (list, tuple)) or not qa_records:
        raise ValueError("qa_records must be a non-empty list")
    if not isinstance(explanation_records, (list, tuple)):
        raise ValueError("explanation_records must be a list")

    queries_by_id: dict[str, dict[str, Any]] = {}
    for record in qa_records:
        query = _validate_query(record)
        if query["qa_id"] in queries_by_id:
            raise ValueError(f"duplicate qa_id {query['qa_id']!r} in QA records")
        queries_by_id[query["qa_id"]] = query

    explanations_by_id: dict[str, tuple[str, tuple[dict[str, str], ...]]] = {}
    for index, record in enumerate(explanation_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"explanation record {index} must be a mapping")
        try:
            qa_id = canonical_id(record["qa_id"], f"explanation record {index} qa_id")
        except KeyError as exc:
            raise ValueError(f"explanation record {index} is missing qa_id") from exc
        if qa_id not in queries_by_id:
            raise ValueError(f"explanation record has unknown qa_id {qa_id!r}")
        if qa_id in explanations_by_id:
            raise ValueError(f"duplicate explanation for qa_id {qa_id!r}")
        try:
            answer_id = canonical_id(
                record["answer_id"], f"explanation record {index} answer_id"
            )
        except KeyError as exc:
            raise ValueError(
                f"explanation record {index} is missing answer_id"
            ) from exc
        path_values = record.get("supporting_path")
        if not isinstance(path_values, (list, tuple)):
            raise ValueError(
                f"explanation record {index} supporting_path must be a list"
            )
        path = tuple(
            _normalize_claim(claim, index, claim_index)
            for claim_index, claim in enumerate(path_values)
        )
        explanations_by_id[qa_id] = (answer_id, path)

    missing = sorted(set(queries_by_id) - set(explanations_by_id))
    if missing:
        raise ValueError(f"missing explanations for qa_ids {missing!r}")

    facts_by_signature: dict[tuple[str, str, str, str], set[str]] = {}
    for fact in graph.facts:
        signature = _fact_signature(fact)
        facts_by_signature.setdefault(signature, set()).add(fact["fact_id"])

    edge_count = supported_edge_count = temporally_consistent_edge_count = 0
    valid_path_count = 0
    results = []
    for qa_id, query in queries_by_id.items():
        answer_id, path = explanations_by_id[qa_id]
        supported = [
            _is_graph_supported(claim, facts_by_signature) for claim in path
        ]
        temporally_consistent = [
            bool(
                filter_facts_by_temporal_constraint(
                    (claim,), query["temporal_constraint"]
                )
            )
            for claim in path
        ]
        reasons = _path_failure_reasons(
            path,
            supported,
            temporally_consistent,
            query["seed_entity_ids"],
            query["relation_sequence"],
            answer_id,
        )
        path_valid = not reasons
        valid_path_count += path_valid
        edge_count += len(path)
        supported_edge_count += sum(supported)
        temporally_consistent_edge_count += sum(temporally_consistent)
        results.append(
            {
                "qa_id": qa_id,
                "answer_id": answer_id,
                "claimed_edge_count": len(path),
                "supported_edge_count": sum(supported),
                "temporally_consistent_edge_count": sum(temporally_consistent),
                "factual_edge_support": _ratio(sum(supported), len(path)),
                "temporal_consistency": _ratio(
                    sum(temporally_consistent), len(path)
                ),
                "supporting_path_valid": path_valid,
                "path_failure_reasons": reasons,
            }
        )

    return {
        "explanation_count": len(results),
        "claimed_edge_count": edge_count,
        "supported_edge_count": supported_edge_count,
        "temporally_consistent_edge_count": temporally_consistent_edge_count,
        "factual_edge_support": _ratio(supported_edge_count, edge_count),
        "temporal_consistency": _ratio(
            temporally_consistent_edge_count, edge_count
        ),
        "valid_supporting_path_count": valid_path_count,
        "supporting_path_validity": valid_path_count / len(results),
        "per_explanation": results,
    }


def _normalize_claim(claim: Any, record_index: int, claim_index: int) -> dict[str, str]:
    label = f"explanation record {record_index} supporting_path[{claim_index}]"
    if not isinstance(claim, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized = {
        field: canonical_id(claim.get(field), f"{label} {field}")
        for field in ("subject_id", "relation_id", "object_id")
    }
    normalized["timestamp"] = normalize_date(
        claim.get("timestamp"), f"{label} timestamp"
    )
    if "fact_id" in claim:
        normalized["fact_id"] = canonical_id(claim["fact_id"], f"{label} fact_id")
    return normalized


def _fact_signature(fact: Mapping[str, str]) -> tuple[str, str, str, str]:
    return (
        fact["subject_id"],
        fact["relation_id"],
        fact["object_id"],
        fact["timestamp"],
    )


def _is_graph_supported(
    claim: Mapping[str, str],
    facts_by_signature: Mapping[tuple[str, str, str, str], set[str]],
) -> bool:
    fact_ids = facts_by_signature.get(_fact_signature(claim), set())
    return bool(fact_ids) and (
        "fact_id" not in claim or claim["fact_id"] in fact_ids
    )


def _path_failure_reasons(
    path: tuple[dict[str, str], ...],
    supported: list[bool],
    temporally_consistent: list[bool],
    seed_entity_ids: Sequence[str],
    relation_sequence: Sequence[str],
    answer_id: str,
) -> list[str]:
    reasons = []
    if not path:
        reasons.append("empty_supporting_path")
    if len(path) != len(relation_sequence):
        reasons.append("wrong_path_length")
    if any(not is_supported for is_supported in supported):
        reasons.append("unsupported_edge")
    if any(not is_consistent for is_consistent in temporally_consistent):
        reasons.append("temporal_violation")
    if any(
        claim["relation_id"] != relation_sequence[index]
        for index, claim in enumerate(path[: len(relation_sequence)])
    ):
        reasons.append("wrong_relation_sequence")
    if path:
        if path[0]["subject_id"] not in seed_entity_ids:
            reasons.append("path_does_not_start_at_seed")
        if any(
            left["object_id"] != right["subject_id"]
            for left, right in zip(path, path[1:])
        ):
            reasons.append("disconnected_path")
        if path[-1]["object_id"] != answer_id:
            reasons.append("wrong_answer_endpoint")
    return reasons


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
