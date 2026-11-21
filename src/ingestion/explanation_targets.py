"""Generate train-only explanation targets from verified temporal graph paths."""

from collections.abc import Iterable, Mapping
from typing import Any

from src.graph.temporal_graph import TemporalGraph
from src.retrieval.subgraph_retriever import _validate_query
from src.retrieval.temporal_filter import filter_facts_by_temporal_constraint


def generate_training_explanation_targets(
    graph: TemporalGraph,
    train_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create one deterministic path target per accepted answer in train QA.

    Only records explicitly assigned to ``train`` are accepted. Each answer
    must exactly match a reachable endpoint under the query's ordered
    relations and temporal constraint; unsupported or incomplete labels fail
    rather than producing a misleading target.
    """
    targets: list[dict[str, Any]] = []
    seen_qa_ids: set[str] = set()

    for index, record in enumerate(train_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"training QA record {index}: expected a mapping")
        if record.get("split") != "train":
            raise ValueError(
                f"training QA record {index}: split must be 'train'"
            )

        query = _validate_query(record)
        qa_id = query["qa_id"]
        if qa_id in seen_qa_ids:
            raise ValueError(f"duplicate training qa_id {qa_id!r}")
        seen_qa_ids.add(qa_id)

        answer_values = record.get("accepted_answer_ids")
        if not isinstance(answer_values, (list, tuple)) or not answer_values:
            raise ValueError(
                f"training QA {qa_id!r}: accepted_answer_ids must be a non-empty list"
            )
        answer_ids = tuple(
            sorted(
                _canonical_answer_id(value, qa_id)
                for value in answer_values
            )
        )
        if len(set(answer_ids)) != len(answer_ids):
            raise ValueError(
                f"training QA {qa_id!r}: accepted_answer_ids must be unique"
            )

        qa_provenance = record.get("provenance")
        if not isinstance(qa_provenance, Mapping):
            raise ValueError(
                f"training QA {qa_id!r}: provenance must be a mapping"
            )

        paths_by_answer = _find_temporal_paths(graph, query)
        reachable_answers = set(paths_by_answer)
        accepted_answers = set(answer_ids)
        if reachable_answers != accepted_answers:
            missing = sorted(accepted_answers - reachable_answers)
            unexpected = sorted(reachable_answers - accepted_answers)
            raise ValueError(
                f"training QA {qa_id!r}: accepted answers do not match "
                f"temporally reachable endpoints (missing={missing}, "
                f"unexpected={unexpected})"
            )

        for answer_id in answer_ids:
            fact_ids = min(paths_by_answer[answer_id])
            path = []
            for fact_id in fact_ids:
                fact = graph.get_fact(fact_id)
                if fact is None:
                    raise ValueError(
                        f"training QA {qa_id!r}: supporting fact {fact_id!r} "
                        "is missing from the graph"
                    )
                path.append(fact)

            targets.append(
                {
                    "qa_id": qa_id,
                    "answer_id": answer_id,
                    "split": "train",
                    "target_text": _render_path(path),
                    "supporting_path": path,
                    "provenance": {
                        "source": "temporal_graph_path",
                        "qa_provenance": dict(qa_provenance),
                        "fact_ids": [fact["fact_id"] for fact in path],
                        "source_ids": [fact["source_id"] for fact in path],
                        "temporal_constraint": dict(query["temporal_constraint"]),
                    },
                }
            )

    if not seen_qa_ids:
        raise ValueError("train_records must contain at least one training QA record")
    return targets


def _find_temporal_paths(
    graph: TemporalGraph,
    query: Mapping[str, Any],
) -> dict[str, set[tuple[str, ...]]]:
    frontier = [(seed_id, ()) for seed_id in query["seed_entity_ids"]]
    for relation_id in query["relation_sequence"]:
        next_frontier = []
        for entity_id, path in frontier:
            facts = filter_facts_by_temporal_constraint(
                graph.outgoing_facts(entity_id, relation_id),
                query["temporal_constraint"],
            )
            next_frontier.extend(
                (fact["object_id"], (*path, fact["fact_id"])) for fact in facts
            )
        frontier = next_frontier
        if not frontier:
            break

    paths_by_answer: dict[str, set[tuple[str, ...]]] = {}
    for answer_id, path in frontier:
        paths_by_answer.setdefault(answer_id, set()).add(path)
    return paths_by_answer


def _canonical_answer_id(value: Any, qa_id: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"training QA {qa_id!r}: accepted answer IDs must be strings or integers"
        )
    answer_id = str(value).strip()
    if not answer_id:
        raise ValueError(
            f"training QA {qa_id!r}: accepted answer IDs must not be empty"
        )
    return answer_id


def _render_path(path: list[dict[str, str]]) -> str:
    steps = [
        f"{fact['subject_id']} -[{fact['relation_id']}]-> "
        f"{fact['object_id']} ({fact['timestamp']})"
        for fact in path
    ]
    return "Evidence path: " + "; ".join(steps) + "."
