import json
import zipfile

import pytest

from src.ingestion.json_loader import load_icews14_facts
from src.ingestion.preprocess import preprocess_facts


def _write_metadata(path):
    metadata = {
        "entity_count": 2,
        "relation_count": 1,
        "timestamp_count": 2,
        "entity2idx": [{"name": "Alpha", "id": 0}, {"name": "Beta", "id": 1}],
        "relation2idx": [{"name": "met", "id": 0}],
        "timestamp2idx": [
            {"name": "2014-01-01", "id": 0},
            {"name": "2014-01-02", "id": 1},
        ],
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _write_archive(path, train_records):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "train.jsonl",
            "\n".join(json.dumps(record) for record in train_records) + "\n",
        )


def test_load_icews14_pt_records_decodes_timestamps_and_preserves_provenance(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    archive_path = tmp_path / "icews14.zip"
    _write_metadata(metadata_path)
    _write_archive(
        archive_path,
        [
            {"query_name": "Pe", "query": [0, 0, 1], "answer": [0]},
            {"query_name": "Pt", "query": [0, 0, 1], "answer": [0, 1]},
        ],
    )

    facts = load_icews14_facts(
        archive_path, metadata_path, splits=("train",)
    )

    assert facts == [
        {
            "subject_id": "0",
            "relation_id": "0",
            "object_id": "1",
            "timestamp": "2014-01-01",
            "source_id": "ICEWS14:train:2:answer:0",
            "source_split": "train",
            "subject_label": "Alpha",
            "relation_label": "met",
            "object_label": "Beta",
        },
        {
            "subject_id": "0",
            "relation_id": "0",
            "object_id": "1",
            "timestamp": "2014-01-02",
            "source_id": "ICEWS14:train:2:answer:1",
            "source_split": "train",
            "subject_label": "Alpha",
            "relation_label": "met",
            "object_label": "Beta",
        },
    ]


def test_load_icews14_rejects_unknown_vocabulary_ids(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    archive_path = tmp_path / "icews14.zip"
    _write_metadata(metadata_path)
    _write_archive(
        archive_path,
        [{"query_name": "Pt", "query": [0, 4, 1], "answer": [0]}],
    )

    with pytest.raises(ValueError, match="unknown relation ID 4"):
        load_icews14_facts(archive_path, metadata_path, splits=("train",))


def test_load_icews14_rejects_metadata_count_mismatch(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    archive_path = tmp_path / "icews14.zip"
    _write_metadata(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["entity_count"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_archive(archive_path, [])

    with pytest.raises(ValueError, match="entity_count is 3"):
        load_icews14_facts(archive_path, metadata_path, splits=("train",))


def test_preprocess_facts_canonicalizes_dates_ids_and_duplicate_provenance():
    records = [
        {
            "subject_id": " 01 ",
            "relation_id": 2,
            "object_id": "3",
            "timestamp": "2014-01-02T01:00:00+02:00",
            "source_id": "z-source",
        },
        {
            "subject_id": "01",
            "relation_id": "2",
            "object_id": 3,
            "timestamp": "2014-01-01",
            "source_id": "a-source",
        },
    ]

    facts = preprocess_facts(records)

    assert len(facts) == 1
    assert facts[0]["subject_id"] == "01"
    assert facts[0]["relation_id"] == "2"
    assert facts[0]["object_id"] == "3"
    assert facts[0]["timestamp"] == "2014-01-01"
    assert facts[0]["source_id"] == "a-source"
    assert facts[0]["fact_id"].startswith("fact:")
    assert preprocess_facts(list(reversed(records))) == facts


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {
                "subject_id": "",
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-01-01",
                "source_id": "source",
            },
            "subject_id",
        ),
        (
            {
                "subject_id": "s",
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-02-30",
                "source_id": "source",
            },
            "invalid timestamp",
        ),
        (
            {
                "subject_id": True,
                "relation_id": "r",
                "object_id": "o",
                "timestamp": "2014-01-01",
                "source_id": "source",
            },
            "subject_id",
        ),
        (
            {
                "subject_id": "s",
                "relation_id": "r",
                "object_id": "o",
                "source_id": "source",
            },
            "timestamp",
        ),
    ],
)
def test_preprocess_facts_rejects_invalid_records(record, message):
    with pytest.raises(ValueError, match=message):
        preprocess_facts([record])


def test_preprocess_facts_rejects_non_mapping_records():
    with pytest.raises(ValueError, match="record 0: expected a mapping"):
        preprocess_facts([None])
