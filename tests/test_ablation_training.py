from src.evaluation.ablation_training import (
    build_training_configurations,
    run_repeated_training,
    run_task65_pilot,
)


def _facts_and_splits():
    facts = [
        {
            "fact_id": "f1",
            "subject_id": "s1",
            "relation_id": "r1",
            "object_id": "a1",
            "timestamp": "2020-01-01",
            "source_id": "fixture",
        },
        {
            "fact_id": "f2",
            "subject_id": "s2",
            "relation_id": "r1",
            "object_id": "a2",
            "timestamp": "2020-01-01",
            "source_id": "fixture",
        },
        {
            "fact_id": "f3",
            "subject_id": "s3",
            "relation_id": "r1",
            "object_id": "a3",
            "timestamp": "2020-01-01",
            "source_id": "fixture",
        },
    ]

    def qa(qa_id, split, seed, answer):
        return {
            "qa_id": qa_id,
            "split": split,
            "question": f"Which entity follows r1 from {seed}?",
            "seed_entity_ids": [seed],
            "relation_sequence": ["r1"],
            "temporal_constraint": {
                "operator": "on_or_before",
                "cutoff_date": "2020-01-01",
            },
            "accepted_answer_ids": [answer],
            "supporting_path_fact_ids": [f"f{seed[-1]}"],
        }

    return facts, [qa("train-1", "train", "s1", "a1"), qa("train-2", "train", "s2", "a2")], [
        qa("validation-1", "validation", "s3", "a3")
    ]


def _configs():
    training = {
        "model_variant": "baseline",
        "objective": "answer_only",
        "seed": 17,
        "device": "cpu",
        "deterministic": True,
        "num_threads": 1,
        "max_epochs": 1,
        "batch_size": 1,
        "max_steps": 10,
        "max_runtime_seconds": 30,
        "num_workers": 0,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "validation": {"every_epochs": 1, "early_stopping_patience": 2},
        "checkpoint_dir": "unused",
        "log_dir": "unused",
    }
    model = {
        "variant": "baseline",
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
        "selr": {"enabled": False, "latent_dim": 3, "max_steps": 3},
    }
    retrieval = {
        "strategy": "relation_constrained_bfs",
        "max_hops": 1,
        "max_nodes": 8,
        "max_edges": 8,
    }
    return training, model, retrieval


def test_pilot_trains_validation_selected_configs_and_repeats_paired_seeds(tmp_path):
    facts, train, validation = _facts_and_splits()
    training, model, retrieval = _configs()
    pilot = run_task65_pilot(
        facts=facts,
        train_records=train,
        validation_records=validation,
        training_config=training,
        model_config=model,
        retrieval_config=retrieval,
        output_dir=tmp_path / "pilot",
        seed=61,
        latent_steps=(1, 3),
    )

    assert pilot["status"] == "complete"
    assert pilot["test_evaluation_performed"] is False
    assert pilot["selection"]["split"] == "validation"
    assert len(pilot["selected_configurations"]) == 4
    assert all(run["best_checkpoint"] for run in pilot["runs"])

    repeated = run_repeated_training(
        facts=facts,
        train_records=train,
        validation_records=validation,
        selected_configurations=pilot["selected_configurations"][:2],
        training_config=training,
        model_config=model,
        retrieval_config=retrieval,
        output_dir=tmp_path / "repeated",
        seeds=(17, 23),
    )

    assert repeated["status"] == "complete"
    assert repeated["test_evaluation_performed"] is False
    assert [run["seed"] for run in repeated["runs"]] == [17, 17, 23, 23]
    assert len(repeated["runs"]) == 4


def test_training_configuration_builder_omits_duplicate_default_step():
    model = _configs()[1]
    configurations = build_training_configurations(model, latent_steps=(1, 3, 4))

    assert [item["configuration_id"] for item in configurations] == [
        "matched_non_selr_baseline",
        "matched_selr",
        "selr_latent_steps_1",
        "selr_latent_steps_4",
        "selr_bottleneck_disabled",
    ]
