import pytest

from src.ingestion.qa_preparation import prepare_qa


def _facts():
    return [
        {
            "fact_id": "f_sm",
            "subject_id": "s",
            "relation_id": "r1",
            "object_id": "m",
            "timestamp": "2014-01-01",
            "source_id": "source:1",
            "subject_label": "Seed",
            "relation_label": "first",
            "object_label": "Middle",
        },
        {
            "fact_id": "f_mx",
            "subject_id": "m",
            "relation_id": "r2",
            "object_id": "x",
            "timestamp": "2014-01-02",
            "source_id": "source:2",
            "object_label": "Answer X",
            "relation_label": "second",
        },
        {
            "fact_id": "f_my",
            "subject_id": "m",
            "relation_id": "r2",
            "object_id": "y",
            "timestamp": "2014-01-03",
            "source_id": "source:3",
            "object_label": "Answer Y",
            "relation_label": "second",
        },
        {
            "fact_id": "f_sn",
            "subject_id": "s",
            "relation_id": "r1",
            "object_id": "n",
            "timestamp": "2014-01-04",
            "source_id": "source:4",
            "object_label": "Other Middle",
            "relation_label": "first",
        },
        {
            "fact_id": "f_nx",
            "subject_id": "n",
            "relation_id": "r2",
            "object_id": "x",
            "timestamp": "2014-01-02",
            "source_id": "source:5",
            "object_label": "Answer X",
            "relation_label": "second",
        },
    ]


def test_prepare_qa_finds_all_temporal_answers_and_tracks_witnesses():
    records = prepare_qa(_facts())
    two_hop = [
        record
        for record in records
        if record["relation_sequence"] == ["r1", "r2"]
    ]
    at_jan_3 = next(
        record
        for record in two_hop
        if record["temporal_constraint"]["cutoff_date"] == "2014-01-03"
    )

    assert at_jan_3["accepted_answer_ids"] == ["x", "y"]
    assert at_jan_3["question"] == (
        "Which entity is reached from Seed by relation first followed by "
        "second, using facts dated on or before 2014-01-03?"
    )
    assert at_jan_3["provenance"]["template_id"] == "two_hop"
    assert at_jan_3["provenance"]["supporting_fact_ids"] == [
        "f_mx",
        "f_my",
        "f_sm",
    ]
    assert at_jan_3["supporting_path_fact_ids"] == ["f_sm", "f_mx"]
    assert "split" not in at_jan_3


def test_prepare_qa_is_order_independent_and_ids_are_stable():
    facts = _facts()

    assert prepare_qa(facts) == prepare_qa(reversed(facts))


def test_prepare_qa_uses_only_edges_on_or_before_the_cutoff():
    records = prepare_qa(_facts())
    one_hop_jan_1 = next(
        record
        for record in records
        if record["relation_sequence"] == ["r1"]
        and record["temporal_constraint"]["cutoff_date"] == "2014-01-01"
    )

    assert one_hop_jan_1["accepted_answer_ids"] == ["m"]
    assert one_hop_jan_1["supporting_path_fact_ids"] == ["f_sm"]


def test_prepare_qa_preserves_repeated_fact_ids_in_ordered_self_loop_witness():
    facts = [
        {
            "fact_id": "loop",
            "subject_id": "s",
            "relation_id": "r",
            "object_id": "s",
            "timestamp": "2014-01-01",
            "source_id": "source",
        }
    ]

    two_hop = next(
        record
        for record in prepare_qa(facts)
        if record["relation_sequence"] == ["r", "r"]
    )

    assert two_hop["accepted_answer_ids"] == ["s"]
    assert two_hop["supporting_path_fact_ids"] == ["loop", "loop"]
    assert two_hop["provenance"]["supporting_fact_ids"] == ["loop"]


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ([], "empty fact collection"),
        ([None], "fact 0: expected a mapping"),
        (
            [
                {
                    "fact_id": "f",
                    "subject_id": "s",
                    "relation_id": "r",
                    "object_id": "o",
                    "timestamp": "2014-02-30",
                    "source_id": "source",
                }
            ],
            "timestamp must be an ISO 8601 date",
        ),
    ],
)
def test_prepare_qa_rejects_invalid_fact_inputs(facts, message):
    with pytest.raises(ValueError, match=message):
        prepare_qa(facts)


def test_prepare_qa_rejects_conflicting_labels():
    facts = _facts()
    facts[1]["subject_label"] = "Not Middle"

    with pytest.raises(ValueError, match="conflicting labels for 'm'"):
        prepare_qa(facts)
