"""Tests for Task 45 diagnostic and efficiency report aggregation."""

from src.evaluation.comparison_report import build_task45_report


def test_task45_report_breaks_down_metrics_and_measures_candidate_coverage():
    qa_records = [
        {
            "qa_id": "one",
            "question": "Which entity follows r1?",
            "relation_sequence": ["r1"],
            "temporal_constraint": {"operator": "on_or_before"},
            "accepted_answer_ids": ["a"],
        },
        {
            "qa_id": "two",
            "question": "Which entity follows r1 then r2?",
            "relation_sequence": ["r1", "r2"],
            "temporal_constraint": {"operator": "on_or_before"},
            "accepted_answer_ids": ["b", "c"],
        },
        {
            "qa_id": "empty",
            "question": "Which entity follows r3?",
            "relation_sequence": ["r3"],
            "temporal_constraint": {"operator": "on_or_after"},
            "accepted_answer_ids": ["d"],
        },
    ]
    predictions = [
        {
            "qa_id": "one",
            "status": "ok",
            "predicted_answer_ids": ["a"],
            "answer_scores": [{"answer_id": "a", "score": 0.9}],
        },
        {
            "qa_id": "two",
            "status": "ok",
            "predicted_answer_ids": ["b"],
            "answer_scores": [{"answer_id": "b", "score": 0.8}],
        },
        {
            "qa_id": "empty",
            "status": "no_candidates",
            "predicted_answer_ids": [],
            "answer_scores": [],
        },
    ]
    measurement = {
        "latency": {
            "evaluation_wall_seconds": 1.5,
            "amortized_ms_per_example": 500.0,
            "examples_per_second": 2.0,
        },
        "memory": {
            "python_heap_peak_bytes": 1000,
            "model_parameter_bytes": 200,
            "optimizer_state_tensor_bytes": 50,
            "max_batch_tensor_storage_bytes": 300,
        },
    }

    report = build_task45_report(
        {
            "benchmark": "matched_learned_baseline_vs_selr",
            "dataset_id": "ICEWS14",
            "test_summary": {"baseline": {"exact_match_accuracy": {"mean": 1.0}}},
        },
        qa_records,
        [
            {
                "seed": 17,
                "variant": "baseline",
                "predictions": predictions,
                "measurement": measurement,
            }
        ],
    )

    run = report["run_reports"][0]
    assert run["accuracy"]["exact_match_accuracy"] == 1 / 3
    assert run["hop_count_breakdown"]["1"]["example_count"] == 2
    assert run["hop_count_breakdown"]["2"]["exact_match_accuracy"] == 0
    assert run["temporal_operator_breakdown"]["on_or_after"]["example_count"] == 1
    assert run["retrieval_coverage"]["covered_gold_answers"] == 2
    assert run["retrieval_coverage"]["gold_answer_recall"] == 2 / 4
    assert run["retrieval_coverage"]["all_gold_answers_coverage_rate"] == 1 / 3
    assert report["input_question_tokens"]["total_question_input_tokens"] == 17
    assert report["input_question_tokens"]["encoder_tokenizer_used"] is True
    assert (
        report["efficiency_summary"]["baseline"]["latency"][
            "evaluation_wall_seconds"
        ]["mean"]
        == 1.5
    )
