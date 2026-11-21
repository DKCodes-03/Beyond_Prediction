import pytest

from src.graph.graph_builder import build_temporal_graph
from src.ingestion.explanation_targets import generate_training_explanation_targets


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
                "fact_id": "f_ab_alt",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "b",
                "timestamp": "2014-01-01",
                "source_id": "source:ab-alt",
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


def _record(**overrides):
    return {
        "qa_id": "train-qa",
        "question": "Which entity follows r1 and then r2?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-03",
        },
        "accepted_answer_ids": ["c", "d"],
        "split": "train",
        "provenance": {
            "source": "synthetic_graph_derived",
            "template_id": "two_hop",
        },
        **overrides,
    }


def test_generates_temporal_path_targets_with_full_provenance(graph):
    targets = generate_training_explanation_targets(graph, [_record()])

    assert [target["answer_id"] for target in targets] == ["c", "d"]
    first, second = targets
    assert first["split"] == second["split"] == "train"
    assert first["provenance"]["fact_ids"] == ["f_ab", "f_bc"]
    assert first["provenance"]["source_ids"] == ["source:ab", "source:bc"]
    assert first["provenance"]["qa_provenance"]["template_id"] == "two_hop"
    assert first["provenance"]["temporal_constraint"]["cutoff_date"] == "2014-01-03"
    assert first["target_text"] == (
        "Evidence path: a -[r1]-> b (2014-01-02); "
        "b -[r2]-> c (2014-01-02)."
    )
    assert second["provenance"]["fact_ids"] == ["f_ab", "f_bd"]


def test_generates_one_hop_target_and_includes_cutoff_boundary(graph):
    record = _record(
        qa_id="one-hop",
        relation_sequence=["r1"],
        accepted_answer_ids=["b"],
        temporal_constraint={
            "operator": "on_or_before",
            "cutoff_date": "2014-01-01",
        },
    )

    target = generate_training_explanation_targets(graph, [record])[0]

    assert target["provenance"]["fact_ids"] == ["f_ab_alt"]
    assert target["supporting_path"][0]["timestamp"] == "2014-01-01"


@pytest.mark.parametrize("split", ["validation", "test", None])
def test_rejects_non_training_supervision(graph, split):
    record = _record(split=split)

    with pytest.raises(ValueError, match="split must be 'train'"):
        generate_training_explanation_targets(graph, [record])


def test_rejects_labels_that_do_not_match_temporally_reachable_answers(graph):
    record = _record(accepted_answer_ids=["c"])

    with pytest.raises(ValueError, match="do not match temporally reachable endpoints"):
        generate_training_explanation_targets(graph, [record])


def test_rejects_empty_training_records(graph):
    with pytest.raises(ValueError, match="at least one training QA record"):
        generate_training_explanation_targets(graph, [])
