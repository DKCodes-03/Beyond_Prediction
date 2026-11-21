"""Focused tests for answer-only SELR training and checkpoint restoration."""

import copy
import json

import pytest
import torch
import yaml

from src.training.train_selr import main as train_selr_main, run_selr_training
from src.training.trainer import (
    BaselineAnswerModel,
    SELRAnswerModel,
    train_baseline,
    train_selr,
    train_selr_explanations,
    validate,
)
from src.utils.checkpoint import load_checkpoint


def _training_config(checkpoint_dir):
    return {
        "model_variant": "selr",
        "objective": "answer_only",
        "seed": 17,
        "device": "cpu",
        "deterministic": True,
        "num_threads": 1,
        "max_epochs": 1,
        "batch_size": 2,
        "max_steps": 10,
        "max_runtime_seconds": 30,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "validation": {"every_epochs": 1, "early_stopping_patience": 3},
        "checkpoint_dir": str(checkpoint_dir),
    }


def _batch():
    return {
        "encoder_memory": torch.randn(2, 3, 4),
        "encoder_memory_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "candidate_embeddings": torch.randn(2, 2, 4),
        "candidate_mask": torch.tensor([[True, True], [True, False]]),
        "answer_targets": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    }


def test_training_validates_and_saves_resumable_checkpoints(tmp_path):
    torch.manual_seed(29)
    config = _training_config(tmp_path / "checkpoints")
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    batches = [_batch()]

    result = train_selr(
        model, batches, batches, config=config, seed=17
    )

    assert result.global_step == 1
    assert result.epochs_completed == 1
    assert result.stop_reason == "max_epochs"
    assert 0 <= result.best_validation_exact_match <= 1
    assert len(result.history) == 1
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()

    restored_model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
    progress = load_checkpoint(
        result.latest_checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_config=config,
        expected_seed=17,
    )
    assert progress["epoch"] == 1
    assert progress["global_step"] == 1
    assert restored_optimizer.state
    for trained, restored in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(trained, restored)

    metrics = validate(restored_model, batches)
    assert metrics["example_count"] == 2
    assert metrics["candidate_count"] == 3
    assert 0 <= metrics["exact_match_accuracy"] <= 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_explanation_stage_restores_answer_checkpoint_and_trains_joint_loss(
    tmp_path,
):
    answer_config = _training_config(tmp_path / "answer-checkpoints")
    answer_batches = [_batch()]
    answer_model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    answer_result = train_selr(
        answer_model,
        answer_batches,
        answer_batches,
        config=answer_config,
        seed=17,
    )

    explanation_batch = {
        **_batch(),
        "explanation_target_ids": torch.tensor([[4, 2, 0], [5, 2, 0]]),
        "explanation_target_mask": torch.tensor(
            [[True, True, False], [True, True, False]]
        ),
    }
    explanation_config = {
        **_training_config(tmp_path / "explanation-checkpoints"),
        "objective": "answer_plus_explanation",
        "explanation": {
            "vocab_size": 8,
            "max_decode_steps": 3,
            "pad_token_id": 0,
            "start_token_id": 1,
            "end_token_id": 2,
            "loss_weight": 0.5,
        },
    }
    stage_two_model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    result = train_selr_explanations(
        stage_two_model,
        [explanation_batch],
        answer_batches,
        answer_checkpoint=answer_result.best_checkpoint,
        answer_config=answer_config,
        config=explanation_config,
        seed=17,
    )

    assert result.global_step == 1
    assert result.epochs_completed == 1
    assert result.best_validation_exact_match >= 0
    assert result.history[0]["training_explanation_loss"] > 0
    assert stage_two_model.explanation_decoder is not None
    assert (tmp_path / "explanation-checkpoints" / "best.pt").is_file()
    checkpoint = torch.load(result.best_checkpoint, weights_only=False)
    assert checkpoint["config"]["objective"] == "answer_plus_explanation"
    assert checkpoint["model_metadata"]["explanation_decoder"] == {
        "vocab_size": 8,
        "max_decode_steps": 3,
        "pad_token_id": 0,
    }
    assert validate(stage_two_model, answer_batches)["example_count"] == 2


def test_answer_stage_requires_the_answer_only_objective(tmp_path):
    config = {
        **_training_config(tmp_path / "checkpoints"),
        "objective": "answer_plus_explanation",
    }

    with pytest.raises(ValueError, match="use train_selr_explanations"):
        train_selr(
            SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2),
            [_batch()],
            [_batch()],
            config=config,
            seed=17,
        )


def test_baseline_trains_on_context_and_padded_candidates(tmp_path):
    torch.manual_seed(29)
    batch = _batch()
    batch["context_embeddings"] = torch.randn(2, 4)
    config = {
        **_training_config(tmp_path / "baseline-checkpoints"),
        "model_variant": "baseline",
    }
    model = BaselineAnswerModel(encoder_dim=4)

    result = train_baseline(model, [batch], [batch], config=config, seed=17)

    assert result.global_step == 1
    assert result.epochs_completed == 1
    assert result.stop_reason == "max_epochs"
    assert (tmp_path / "baseline-checkpoints" / "best.pt").is_file()
    metrics = validate(model, [batch])
    assert metrics["example_count"] == 2
    assert metrics["candidate_count"] == 3


def test_checkpoint_resume_rejects_mismatched_metadata(tmp_path):
    config = _training_config(tmp_path / "checkpoints")
    batches = [_batch()]
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    result = train_selr(model, batches, batches, config=config, seed=17)
    restored_model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
    wrong_config = copy.deepcopy(config)
    wrong_config["seed"] = 23

    with pytest.raises(ValueError, match="config does not match"):
        load_checkpoint(
            result.latest_checkpoint,
            model=restored_model,
            optimizer=optimizer,
            expected_config=wrong_config,
            expected_seed=17,
        )

    with pytest.raises(ValueError, match="seed does not match"):
        load_checkpoint(
            result.latest_checkpoint,
            model=restored_model,
            optimizer=optimizer,
            expected_config=config,
            expected_seed=23,
        )


def test_training_rejects_one_shot_batch_iterators(tmp_path):
    config = _training_config(tmp_path / "checkpoints")
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    one_shot = iter([_batch()])

    with pytest.raises(ValueError, match="reusable across epochs"):
        train_selr(model, one_shot, [_batch()], config=config, seed=17)


def test_selr_learns_controlled_tiny_cpu_fixture(tmp_path):
    positive = torch.tensor([1.0, 0.0, 0.0, 0.0])
    negative = -positive
    batch = {
        "encoder_memory": torch.tensor(
            [
                [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
                [[0.0, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
            ]
        ),
        "encoder_memory_mask": torch.ones((2, 2), dtype=torch.bool),
        "candidate_embeddings": torch.stack(
            (torch.stack((positive, negative)), torch.stack((negative, positive)))
        ),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "answer_targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    torch.manual_seed(17)
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    initial_metrics = validate(model, [batch])
    config = {
        **_training_config(tmp_path / "checkpoints"),
        "max_epochs": 30,
        "max_steps": 30,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
        },
        "validation": {"every_epochs": 1, "early_stopping_patience": 100},
    }

    result = train_selr(model, [batch], [batch], config=config, seed=17)
    final_metrics = validate(model, [batch])

    assert final_metrics["exact_match_accuracy"] == 1.0
    assert final_metrics["loss"] < initial_metrics["loss"] * 0.5
    assert result.epochs_completed > 1
    assert result.stop_reason == "max_steps"
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_selr_entry_point_trains_from_facts_and_qa_splits(tmp_path):
    facts = [
        {
            "fact_id": "f_ab",
            "subject_id": "a",
            "relation_id": "r",
            "object_id": "b",
            "timestamp": "2014-01-01",
            "source_id": "fixture",
        },
        {
            "fact_id": "f_xy",
            "subject_id": "x",
            "relation_id": "r",
            "object_id": "y",
            "timestamp": "2014-01-01",
            "source_id": "fixture",
        },
    ]
    train_records = [
        {
            "qa_id": "train-1",
            "split": "train",
            "question": "Which entity follows r?",
            "seed_entity_ids": ["a"],
            "relation_sequence": ["r"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2014-01-01",
            },
            "accepted_answer_ids": ["b"],
        }
    ]
    validation_records = [
        {
            "qa_id": "validation-1",
            "split": "validation",
            "question": "Which entity follows r?",
            "seed_entity_ids": ["x"],
            "relation_sequence": ["r"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2014-01-01",
            },
            "accepted_answer_ids": ["y"],
        }
    ]
    facts_path = tmp_path / "facts.json"
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    facts_path.write_text(json.dumps(facts), encoding="utf-8")
    train_path.write_text(json.dumps(train_records), encoding="utf-8")
    validation_path.write_text(json.dumps(validation_records), encoding="utf-8")

    training_config = {
        **_training_config(tmp_path / "checkpoints"),
        "model_variant": "selr",
        "num_workers": 0,
        "log_dir": str(tmp_path / "logs"),
    }
    model_config = {
        "variant": "selr",
        "encoder": {
            "type": "entity_relation_time_embeddings",
            "embedding_dim": 4,
            "aggregation": "mean",
            "dropout": 0.0,
        },
        "answer_head": {
            "type": "candidate_scoring",
            "candidate_policy": "retrieved_entities",
            "max_candidates": 8,
        },
        "selr": {"enabled": True, "latent_dim": 3, "max_steps": 2},
    }
    retrieval_config = {
        "strategy": "relation_constrained_bfs",
        "max_hops": 2,
        "max_nodes": 8,
        "max_edges": 8,
    }
    config_paths = []
    for name, value in (
        ("training.yaml", training_config),
        ("model.yaml", model_config),
        ("retrieval.yaml", retrieval_config),
    ):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        config_paths.append(path)

    assert train_selr_main(
        [
            "--facts",
            str(facts_path),
            "--train-split",
            str(train_path),
            "--validation-split",
            str(validation_path),
            "--training-config",
            str(config_paths[0]),
            "--model-config",
            str(config_paths[1]),
            "--retrieval-config",
            str(config_paths[2]),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    ) == 0

    run_metadata = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run_metadata["model_variant"] == "selr"
    assert run_metadata["metrics"]["global_step"] == 1
    assert run_metadata["metrics"]["epochs_completed"] == 1
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    training_log = (tmp_path / "logs" / "beyond_prediction.selr.log").read_text()
    assert "training_epoch" in training_log
    assert "training_complete" in training_log

    bad_train_records = [
        {**train_records[0], "accepted_answer_ids": ["outside-retrieval"]}
    ]
    with pytest.raises(ValueError, match="missing accepted answers"):
        run_selr_training(
            facts,
            bad_train_records,
            validation_records,
            training_config={
                **training_config,
                "checkpoint_dir": str(tmp_path / "bad-checkpoints"),
            },
            model_config=model_config,
            retrieval_config=retrieval_config,
        )
