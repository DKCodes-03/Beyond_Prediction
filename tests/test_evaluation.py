"""Tests for answer metrics, prediction exports, and benchmark execution."""

import json

import pytest

from src.evaluation.benchmark import main, run_path_baseline_benchmark
from src.evaluation.evaluate_answers import export_predictions
from src.evaluation.metrics import compute_answer_metrics, normalize_answer_ids


def test_normalize_answer_ids_canonicalizes_and_deduplicates():
    assert normalize_answer_ids([" b ", 1, "b"]) == ("1", "b")
    assert normalize_answer_ids([], allow_empty=True) == ()

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_answer_ids([])
    with pytest.raises(ValueError, match="must be a non-empty string or integer"):
        normalize_answer_ids([True])


def test_metrics_compare_answer_sets_and_count_failed_predictions():
    qa_records = [
        {"qa_id": "q1", "accepted_answer_ids": ["a", "b"]},
        {"qa_id": "q2", "accepted_answer_ids": ["c"]},
    ]
    predictions = [
        {
            "qa_id": " q1 ",
            "status": "ok",
            "predicted_answer_ids": [" b", "a", "a "],
        },
        {
            "qa_id": "q2",
            "status": "no_candidates",
            "predicted_answer_ids": [],
        },
    ]

    metrics = compute_answer_metrics(qa_records, predictions)

    assert metrics["example_count"] == 2
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["micro_precision"] == 1
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == 0.5
    assert metrics["status_counts"]["no_candidates"] == 1


def test_metrics_reject_missing_duplicate_or_inconsistent_predictions():
    qa_records = [{"qa_id": "q1", "accepted_answer_ids": ["a"]}]
    valid_prediction = {
        "qa_id": "q1",
        "status": "ok",
        "predicted_answer_ids": ["a"],
    }

    with pytest.raises(ValueError, match="IDs must match exactly"):
        compute_answer_metrics(
            qa_records,
            [{**valid_prediction, "qa_id": "extra"}],
        )
    with pytest.raises(ValueError, match="duplicate qa_id"):
        compute_answer_metrics(qa_records, [valid_prediction, valid_prediction])
    with pytest.raises(ValueError, match="must not contain predictions"):
        compute_answer_metrics(
            qa_records,
            [{**valid_prediction, "status": "model_error"}],
        )


def test_prediction_export_normalizes_ids_and_rejects_nonfinite_scores(tmp_path):
    destination = export_predictions(
        [
            {
                "qa_id": " q1 ",
                "model_variant": " baseline ",
                "status": "ok",
                "predicted_answer_ids": ["b", " a ", "a"],
                "answer_scores": [{"answer_id": " a ", "score": 0.75}],
            }
        ],
        tmp_path / "predictions.json",
    )
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved[0]["qa_id"] == "q1"
    assert saved[0]["model_variant"] == "baseline"
    assert saved[0]["predicted_answer_ids"] == ["a", "b"]
    assert saved[0]["answer_scores"] == [{"answer_id": "a", "score": 0.75}]

    with pytest.raises(ValueError, match="score must be finite"):
        export_predictions(
            [
                {
                    "qa_id": "q1",
                    "model_variant": "baseline",
                    "status": "ok",
                    "predicted_answer_ids": ["a"],
                    "answer_scores": [{"answer_id": "a", "score": float("nan")}],
                }
            ],
            tmp_path / "invalid.json",
        )


def test_path_benchmark_writes_predictions_metrics_and_run_config(tmp_path):
    facts = [
        {
            "fact_id": "f_ab",
            "subject_id": "a",
            "relation_id": "r1",
            "object_id": "b",
            "timestamp": "2014-01-01",
            "source_id": "fixture",
        }
    ]
    qa_records = [
        {
            "qa_id": "q1",
            "question": "Which entity follows r1?",
            "seed_entity_ids": ["a"],
            "relation_sequence": ["r1"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2014-01-01",
            },
            "accepted_answer_ids": ["b"],
            "split": "test",
        }
    ]

    result = run_path_baseline_benchmark(
        facts,
        qa_records,
        output_dir=tmp_path,
        split="test",
        seed=17,
        config={"max_edges": 100},
    )

    assert result["metrics"]["exact_match_accuracy"] == 1
    predictions = json.loads((tmp_path / "predictions.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    run = json.loads((tmp_path / "run.json").read_text())
    assert predictions[0]["predicted_answer_ids"] == ["b"]
    assert metrics["example_count"] == 1
    assert run["seed"] == 17
    assert run["split"] == "test"
    assert run["config"]["max_edges"] == 100


def test_benchmark_cli_reads_json_and_runs_selected_split(tmp_path, capsys):
    facts_path = tmp_path / "facts.json"
    split_path = tmp_path / "test.json"
    output_dir = tmp_path / "benchmark"
    facts_path.write_text(
        json.dumps(
            [
                {
                    "fact_id": "f_ab",
                    "subject_id": "a",
                    "relation_id": "r1",
                    "object_id": "b",
                    "timestamp": "2014-01-01",
                    "source_id": "fixture",
                }
            ]
        ),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(
            [
                {
                    "qa_id": "q1",
                    "question": "Which entity follows r1?",
                    "seed_entity_ids": ["a"],
                    "relation_sequence": ["r1"],
                    "temporal_constraint": {
                        "operator": "on_or_before",
                        "cutoff_date": "2014-01-01",
                    },
                    "accepted_answer_ids": ["b"],
                    "split": "test",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--facts",
            str(facts_path),
            "--split-file",
            str(split_path),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert json.loads(capsys.readouterr().out)["metrics"]["exact_match_accuracy"] == 1
    assert (output_dir / "predictions.json").exists()
