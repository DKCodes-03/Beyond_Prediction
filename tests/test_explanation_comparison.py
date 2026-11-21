"""Paired answer-regression accounting for Task 53."""

from src.evaluation.explanation_comparison import compare_answer_predictions


def test_compare_answer_predictions_saves_persistent_and_new_failures():
    qa_records = [
        {
            "qa_id": "still-correct",
            "accepted_answer_ids": ["a"],
            "seed_entity_ids": ["s"],
        },
        {
            "qa_id": "regressed",
            "accepted_answer_ids": ["b"],
            "relation_sequence": ["r"],
        },
        {
            "qa_id": "persistent",
            "accepted_answer_ids": ["c"],
        },
    ]
    before = [
        {"qa_id": "still-correct", "status": "ok", "predicted_answer_ids": ["a"]},
        {"qa_id": "regressed", "status": "ok", "predicted_answer_ids": ["b"]},
        {"qa_id": "persistent", "status": "ok", "predicted_answer_ids": ["x"]},
    ]
    after = [
        {"qa_id": "still-correct", "status": "ok", "predicted_answer_ids": ["a"]},
        {"qa_id": "regressed", "status": "ok", "predicted_answer_ids": ["x"]},
        {"qa_id": "persistent", "status": "ok", "predicted_answer_ids": ["x"]},
    ]

    summary, failures = compare_answer_predictions(qa_records, before, after)

    assert summary["before_explanation_training"]["exact_match_accuracy"] == 2 / 3
    assert summary["after_explanation_training"]["exact_match_accuracy"] == 1 / 3
    assert summary["exact_match_delta"] == -1 / 3
    assert summary["regression_count"] == 1
    assert summary["improved_count"] == 0
    assert [case["failure_type"] for case in failures["failures"]] == [
        "regression",
        "persistent_failure",
    ]
