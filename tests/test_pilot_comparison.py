import json

import pytest

from src.evaluation.pilot_comparison import (
    build_pilot_report,
    define_bottleneck_ablations,
    define_latent_step_ablations,
    run_pilot_comparison,
)


def _qa_records():
    return [
        {
            "qa_id": "train-1",
            "split": "train",
            "relation_sequence": ["r1"],
            "accepted_answer_ids": ["a1"],
            "cutoff": "2014-01-01",
        },
        {
            "qa_id": "train-2",
            "split": "train",
            "relation_sequence": ["r1", "r2", "r3"],
            "accepted_answer_ids": ["a1", "a2", "a3"],
            "cutoff": "2014-06-01",
        },
        {
            "qa_id": "validation-1",
            "split": "validation",
            "relation_sequence": ["r1", "r2"],
            "accepted_answer_ids": ["a1", "a2"],
            "cutoff": "2014-12-01",
        },
    ]


def test_pilot_report_is_deterministic_and_rejects_test_split(tmp_path):
    records = _qa_records()
    report = build_pilot_report(records)

    assert report["status"] == "designed"
    assert report["test_evaluation_performed"] is False
    assert report["allowed_splits"] == ["train", "validation"]
    assert report["difficulty"]["difficulty_slices"]["dense_candidates"]["record_count"] >= 1
    assert report["configurations"]["baseline"]["model_variant"] == "baseline"

    result = run_pilot_comparison(
        output_dir=tmp_path,
        qa_records=records,
    )
    saved = json.loads(
        (tmp_path / "pilot_comparison.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert result["test_evaluation_performed"] is False

    test_records = [{**records[0], "split": "test"}]
    with pytest.raises(ValueError, match="test-split"):
        build_pilot_report(test_records)


def test_pilot_defines_separate_latent_and_bottleneck_ablations():
    latent = define_latent_step_ablations((1, 2, 4))
    assert [item["max_steps"] for item in latent] == [1, 2, 4]
    assert all(item["ablation_type"] == "latent_steps" for item in latent)

    bottleneck = define_bottleneck_ablations()
    assert [
        item["ablation_id"] for item in bottleneck
    ] == ["selr_bottleneck_enabled", "selr_bottleneck_disabled"]
    assert bottleneck[1]["bottleneck_enabled"] is False


def test_pilot_difficulty_slices_read_temporal_constraint_cutoffs():
    records = [
        {
            "qa_id": f"validation-{index}",
            "split": "validation",
            "relation_sequence": ["r1"],
            "accepted_answer_ids": ["a1"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": cutoff,
            },
        }
        for index, cutoff in enumerate(
            ("2014-01-01", "2014-06-01", "2014-12-01")
        )
    ]

    report = build_pilot_report(records)
    slices = report["difficulty"]["difficulty_slices"]

    assert slices["early_cutoff"]["record_count"] == 1
    assert slices["middle_cutoff"]["record_count"] == 1
    assert slices["late_cutoff"]["record_count"] == 1


@pytest.mark.parametrize("steps", [[], [0], [True], [2, 2]])
def test_pilot_rejects_invalid_latent_step_ablations(steps):
    with pytest.raises(ValueError, match="latent"):
        define_latent_step_ablations(steps)
