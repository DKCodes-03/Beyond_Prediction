import hashlib
import json

import pytest

from src.ingestion.artifact_store import save_processed_artifacts


def _facts():
    return [
        {
            "fact_id": "f1",
            "subject_id": "s1",
            "relation_id": "r1",
            "object_id": "o1",
            "timestamp": "2014-01-01",
            "source_id": "source:1",
        }
    ]


def _splits():
    return {
        split: [
            {
                "qa_id": f"qa:{split}",
                "question": f"question {split}",
                "split": split,
                "supporting_path_fact_ids": [f"fact:{split}"],
            }
        ]
        for split in ("train", "validation", "test")
    }


def _save(tmp_path, splits=None):
    return save_processed_artifacts(
        _facts(),
        _splits() if splits is None else splits,
        tmp_path / "processed",
        tmp_path / "splits",
        dataset_id="ICEWS14",
        dataset_source="huggingface:linxy/ICEWS14",
        dataset_revision="revision-1",
        source_checksum_sha256="a" * 64,
        split_seed=17,
        split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
    )


def test_save_processed_artifacts_writes_versioned_manifest_and_data(tmp_path):
    manifest_path = _save(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema"] == "processed_artifacts.schema.json"
    assert manifest["schema_version"] == 1
    assert manifest["dataset"]["id"] == "ICEWS14"
    assert manifest["dataset"]["revision"] == "revision-1"
    assert manifest["record_schemas"]["facts"].endswith("#/$defs/fact")
    assert manifest["split_provenance"] == {
        "strategy": "deterministic_grouped_random",
        "group_by": "query_path_signature",
        "seed": 17,
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "evidence_policy": "closed_knowledge_base",
        "label_usage": {
            "train": "fit",
            "validation": "model_selection",
            "test": "final_evaluation_only",
        },
    }
    for split, filename in (
        ("train", "train.json"),
        ("validation", "val.json"),
        ("test", "test.json"),
    ):
        artifact = manifest["artifacts"]["splits"][split]
        content = (manifest_path.parent / artifact["path"]).read_bytes()
        assert json.loads(content) == _splits()[split]
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
        assert artifact["record_count"] == 1
        assert (tmp_path / "splits" / filename).exists()

    facts_artifact = manifest["artifacts"]["facts"]
    facts_content = (manifest_path.parent / facts_artifact["path"]).read_bytes()
    assert json.loads(facts_content) == _facts()
    assert facts_artifact["sha256"] == hashlib.sha256(facts_content).hexdigest()

    original_bytes = manifest_path.read_bytes()
    _save(tmp_path)
    assert manifest_path.read_bytes() == original_bytes


def test_save_processed_artifacts_rejects_split_path_leakage(tmp_path):
    splits = _splits()
    splits["validation"][0]["supporting_path_fact_ids"] = ["fact:train"]

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        _save(tmp_path, splits)


def test_save_processed_artifacts_rejects_invalid_source_checksum(tmp_path):
    with pytest.raises(ValueError, match="64-character hex digest"):
        save_processed_artifacts(
            _facts(),
            _splits(),
            tmp_path / "processed",
            tmp_path / "splits",
            dataset_id="ICEWS14",
            dataset_source="source",
            dataset_revision=None,
            source_checksum_sha256="not-a-checksum",
            split_seed=17,
            split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        )
