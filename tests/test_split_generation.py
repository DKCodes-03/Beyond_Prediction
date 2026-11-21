import json

import pytest

from src.ingestion.split_generation import split_qa_records, write_qa_splits


def _records():
    return [
        {
            "qa_id": f"qa:{index}",
            "question": f"question {index}",
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2014-01-01",
            },
            "supporting_path_fact_ids": path,
        }
        for index, path in enumerate(
            (["shared"], ["shared"], ["f2"], ["f3"], ["f4"], ["f5"])
        )
    ]


def test_split_qa_records_is_deterministic_and_keeps_paths_together():
    records = _records()

    splits = split_qa_records(records)

    assert splits == split_qa_records(records)
    assert splits["train"] and splits["validation"] and splits["test"]
    assignments = {
        record["qa_id"]: split
        for split, split_records in splits.items()
        for record in split_records
    }
    assert assignments["qa:0"] == assignments["qa:1"]
    assert all(
        record["temporal_constraint"]["operator"] == "on_or_before"
        for split_records in splits.values()
        for record in split_records
    )
    assert all(
        record["split"] == split
        for split, split_records in splits.items()
        for record in split_records
    )


def test_write_qa_splits_uses_contract_filenames_and_json_arrays(tmp_path):
    splits = split_qa_records(_records())

    write_qa_splits(splits, tmp_path)

    for split, filename in (
        ("train", "train.json"),
        ("validation", "val.json"),
        ("test", "test.json"),
    ):
        records = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
        assert records == splits[split]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "at least three QA records"),
        (
            [
                {
                    "qa_id": "qa:1",
                    "temporal_constraint": {
                        "operator": "future",
                        "cutoff_date": "2014-01-01",
                    },
                    "supporting_path_fact_ids": ["f1"],
                },
                {
                    "qa_id": "qa:2",
                    "temporal_constraint": {
                        "operator": "on_or_before",
                        "cutoff_date": "2014-01-01",
                    },
                    "supporting_path_fact_ids": ["f2"],
                },
                {
                    "qa_id": "qa:3",
                    "temporal_constraint": {
                        "operator": "on_or_before",
                        "cutoff_date": "2014-01-01",
                    },
                    "supporting_path_fact_ids": ["f3"],
                },
            ],
            "temporal operator",
        ),
    ],
)
def test_split_qa_records_rejects_insufficient_or_invalid_data(records, message):
    with pytest.raises(ValueError, match=message):
        split_qa_records(records)


def test_split_qa_records_rejects_duplicate_ids_and_paths_cannot_fill_splits():
    records = _records()
    duplicate_id = [*records, {**records[0], "supporting_path_fact_ids": ["other"]}]
    same_path = [
        {**record, "supporting_path_fact_ids": ["same"]}
        for record in records[:3]
    ]

    with pytest.raises(ValueError, match="duplicate qa_id"):
        split_qa_records(duplicate_id)
    with pytest.raises(ValueError, match="three distinct supporting-path groups"):
        split_qa_records(same_path)


def test_split_qa_records_validates_fraction_sum():
    with pytest.raises(ValueError, match="must sum to 1"):
        split_qa_records(
            _records(),
            fractions={"train": 0.7, "validation": 0.1, "test": 0.1},
        )
