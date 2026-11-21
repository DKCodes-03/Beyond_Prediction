"""Run the frozen, paired learned-baseline versus SELR comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.evaluation.comparison_report import build_task45_report
from src.evaluation.metrics import compute_answer_metrics
from src.graph.graph_builder import build_temporal_graph
from src.training.train_baseline import run_baseline_training
from src.training.train_selr import (
    EncodedQADataset,
    collate_encoded_examples,
    run_selr_training,
)
from src.training.trainer import (
    BaselineAnswerModel,
    SELRAnswerModel,
    TrainingResult,
)
from src.utils.checkpoint import load_checkpoint
from src.utils.logging_utils import detect_code_revision
from src.utils.seed import set_seed


_SEEDS = (17, 23, 42)
_FROZEN_HASHES = {
    "facts": "b9a14c0124ba246dee9bb0e230b22370aaf0d0b7c9916c84ed9fee5bec172357",
    "train": "dd11dfc4943e265f776a3696dceb8e7d8d0ad1cdfc044516edd35e6859edf8b9",
    "validation": "5c069f2e323a6e49549f99124aec63ec037cc9c7c2bb3e91364bf67f3e8ebb71",
    "test": "03d787db7fe56aa51f7832982427f784c3134ebaa3accdd181800962e2aeea34",
}
_FROZEN_COUNTS = {"train": 14156, "validation": 1770, "test": 1770}


def run_learned_comparison(
    facts: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    input_hashes: Mapping[str, str],
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    output_dir: str | Path,
    seeds: Sequence[int] = _SEEDS,
) -> dict[str, Any]:
    """Train all paired runs, then evaluate test labels exactly once per run."""
    _validate_frozen_inputs(facts, splits, input_hashes)
    _validate_frozen_configs(training_config, model_config, retrieval_config)
    if tuple(seeds) != _SEEDS:
        raise ValueError(f"seeds must match the frozen sequence {_SEEDS!r}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "comparison.json"
    if result_path.exists():
        raise FileExistsError(
            f"{result_path} already exists; choose a fresh output directory"
        )

    train_records = splits["train"]
    validation_records = splits["validation"]
    test_records = splits["test"]
    run_results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "benchmark": "matched_learned_baseline_vs_selr",
        "status": "training",
        "dataset_id": "ICEWS14",
        "dataset_revision": "462141693b1c9c2ece49384fab41955eae89471a",
        "code_revision": detect_code_revision(),
        "source_tree_sha256": _source_tree_sha256(),
        "seeds": list(_SEEDS),
        "input_sha256": dict(input_hashes),
        "training_runs": run_results,
        "test_evaluation_performed": False,
        "measurement_conditions": {
            "device": training_config["device"],
            "host": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "retrieval": {
                name: retrieval_config[name]
                for name in ("strategy", "max_hops", "max_nodes", "max_edges")
            },
            "model_config": dict(model_config),
            "selr_model_config": _variant_model_config(model_config, "selr"),
            "encoder_policy": (
                "GraphQuestionEncoder initialized from each training seed, "
                "evaluation mode, frozen parameters"
            ),
            "answer_policy": (
                "retrieved final-hop candidates; multi-label BCE; threshold 0.5; "
                "top-candidate fallback"
            ),
            "training_budget": {
                "optimizer": dict(training_config["optimizer"]),
                "batch_size": training_config["batch_size"],
                "max_epochs": training_config["max_epochs"],
                "max_steps": training_config["max_steps"],
                "max_runtime_seconds": training_config["max_runtime_seconds"],
                "validation_every_epochs": training_config["validation"]["every_epochs"],
                "early_stopping_patience": training_config["validation"][
                    "early_stopping_patience"
                ],
            },
        },
    }
    _write_json(result_path, summary)

    for seed in _SEEDS:
        for variant in ("baseline", "selr"):
            run_config = dict(training_config)
            run_config["seed"] = seed
            run_config["model_variant"] = variant
            checkpoint_dir = destination / f"{variant}_seed_{seed}" / "checkpoints"
            run_config["checkpoint_dir"] = str(checkpoint_dir)
            variant_config = _variant_model_config(model_config, variant)
            started_at = time.monotonic()
            if variant == "baseline":
                result = run_baseline_training(
                    facts,
                    train_records,
                    validation_records,
                    training_config=run_config,
                    model_config=variant_config,
                    retrieval_config=retrieval_config,
                    runtime_started_at=started_at,
                )
            else:
                result = run_selr_training(
                    facts,
                    train_records,
                    validation_records,
                    training_config=run_config,
                    model_config=variant_config,
                    retrieval_config=retrieval_config,
                    runtime_started_at=started_at,
                )
            run_results.append(
                _training_run_record(
                    seed,
                    variant,
                    result,
                    elapsed_seconds=time.monotonic() - started_at,
                )
            )
            _write_json(result_path, summary)

    incomplete = [
        run
        for run in run_results
        if run["stop_reason"] == "max_runtime"
        or run["elapsed_seconds"] > training_config["max_runtime_seconds"]
    ]
    if incomplete:
        summary["status"] = "incomplete"
        summary["incomplete_runs"] = [
            {"variant": run["variant"], "seed": run["seed"]}
            for run in incomplete
        ]
        _write_json(result_path, summary)
        return summary

    test_metrics: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "selr": [],
    }
    for run in run_results:
        predictions = _evaluate_checkpoint(
            facts,
            train_records,
            test_records,
            seed=run["seed"],
            variant=run["variant"],
            checkpoint_path=run["best_checkpoint"],
            training_config={
                **training_config,
                "model_variant": run["variant"],
                "seed": run["seed"],
                "checkpoint_dir": str(
                    destination
                    / f"{run['variant']}_seed_{run['seed']}"
                    / "checkpoints"
                ),
            },
            model_config=_variant_model_config(model_config, run["variant"]),
            retrieval_config=retrieval_config,
        )
        metrics = compute_answer_metrics(test_records, predictions)
        test_metrics[run["variant"]].append(metrics)
        predictions_path = (
            destination
            / f"{run['variant']}_seed_{run['seed']}_predictions.json"
        )
        _write_json(predictions_path, predictions)
        run["test_metrics"] = metrics
        run["predictions_path"] = str(predictions_path)
        _write_json(result_path, summary)

    baseline_accuracies = [
        item["exact_match_accuracy"] for item in test_metrics["baseline"]
    ]
    selr_accuracies = [item["exact_match_accuracy"] for item in test_metrics["selr"]]
    summary["test_summary"] = {
        variant: _metric_summary(test_metrics[variant])
        for variant in ("baseline", "selr")
    }
    paired_differences = [
        selr - baseline
        for baseline, selr in zip(
            baseline_accuracies, selr_accuracies, strict=True
        )
    ]
    summary["test_summary"]["selr_minus_baseline_exact_match"] = {
        "per_seed": paired_differences,
        "mean": statistics.mean(paired_differences),
        "sample_std": statistics.stdev(paired_differences),
    }
    summary["test_evaluation_performed"] = True
    summary["status"] = "complete"
    _write_json(result_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen three-seed learned baseline versus SELR comparison."
    )
    artifact_root = Path(
        "outputs/experiments/graph_path_sanity_test_pilot/input_artifacts"
    )
    parser.add_argument(
        "--facts", type=Path, default=artifact_root / "processed" / "facts.json"
    )
    parser.add_argument(
        "--train-split", type=Path, default=artifact_root / "splits" / "train.json"
    )
    parser.add_argument(
        "--validation-split", type=Path, default=artifact_root / "splits" / "val.json"
    )
    parser.add_argument(
        "--test-split", type=Path, default=artifact_root / "splits" / "test.json"
    )
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/training_config.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_config.yaml")
    )
    parser.add_argument(
        "--retrieval-config", type=Path, default=Path("configs/retrieval_config.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/task44_learned_comparison"),
    )
    parser.add_argument(
        "--export-report",
        action="store_true",
        help="Measure saved Task 44 checkpoints and export the Task 45 report only.",
    )
    args = parser.parse_args(argv)
    input_paths = {
        "facts": args.facts,
        "train": args.train_split,
        "validation": args.validation_split,
        "test": args.test_split,
    }
    payloads = {name: _read_json_array(path, name) for name, path in input_paths.items()}
    input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in input_paths.items()
    }
    config = _read_yaml_mapping(args.training_config, "training config")
    model = _read_yaml_mapping(args.model_config, "model config")
    retrieval = _read_yaml_mapping(args.retrieval_config, "retrieval config")
    if args.export_report:
        result = export_task45_report(
            payloads["facts"],
            payloads["train"],
            payloads["validation"],
            payloads["test"],
            input_hashes=input_hashes,
            training_config=config,
            model_config=model,
            retrieval_config=retrieval,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    result = run_learned_comparison(
        payloads["facts"],
        {name: payloads[name] for name in ("train", "validation", "test")},
        input_hashes=input_hashes,
        training_config=config,
        model_config=model,
        retrieval_config=retrieval,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def export_task45_report(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    input_hashes: Mapping[str, str],
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure existing Task 44 checkpoints and export their diagnostic report."""
    splits = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }
    _validate_frozen_inputs(facts, splits, input_hashes)
    _validate_frozen_configs(training_config, model_config, retrieval_config)
    destination = Path(output_dir)
    comparison_path = destination / "comparison.json"
    comparison = _read_json_object(comparison_path, "Task 44 comparison")
    if comparison.get("status") != "complete" or not comparison.get(
        "test_evaluation_performed"
    ):
        raise ValueError("Task 44 comparison must have completed test evaluation")
    stored_runs = comparison.get("training_runs")
    if not isinstance(stored_runs, list) or len(stored_runs) != len(_SEEDS) * 2:
        raise ValueError("Task 44 comparison must contain all six trained runs")

    measured_runs: list[dict[str, Any]] = []
    for run in stored_runs:
        if not isinstance(run, Mapping):
            raise ValueError("every Task 44 training run must be an object")
        seed = run.get("seed")
        variant = run.get("variant")
        if seed not in _SEEDS or variant not in {"baseline", "selr"}:
            raise ValueError("Task 44 training run has an unexpected seed or variant")
        prediction_path = destination / f"{variant}_seed_{seed}_predictions.json"
        saved_predictions = _read_json_array(prediction_path, "saved predictions")
        measured_predictions, measurement = _evaluate_checkpoint(
            facts,
            train_records,
            test_records,
            seed=seed,
            variant=variant,
            checkpoint_path=str(
                destination / f"{variant}_seed_{seed}" / "checkpoints" / "best.pt"
            ),
            training_config={
                **training_config,
                "model_variant": variant,
                "seed": seed,
                "checkpoint_dir": str(
                    destination / f"{variant}_seed_{seed}" / "checkpoints"
                ),
            },
            model_config=_variant_model_config(model_config, variant),
            retrieval_config=retrieval_config,
        )
        if _prediction_signature(saved_predictions) != _prediction_signature(
            measured_predictions
        ):
            raise ValueError(
                f"re-evaluated predictions differ from Task 44 artifacts for "
                f"{variant} seed {seed}"
            )
        measured_runs.append(
            {
                "seed": seed,
                "variant": variant,
                "predictions": measured_predictions,
                "measurement": measurement,
            }
        )

    report = build_task45_report(comparison, test_records, measured_runs)
    report["measurement_conditions"] = {
        "device": training_config["device"],
        "host": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "batch_size": training_config["batch_size"],
        "num_threads": training_config["num_threads"],
        "latency": (
            "One un-warmed pass over the test split per checkpoint. Timing includes "
            "dataset retrieval and encoding, DataLoader collation, model forward, "
            "and answer decoding; graph/model/checkpoint setup is excluded. "
            "Per-example latency is evaluation wall time divided by the example "
            "count; the measured batch size is reported separately."
        ),
        "memory": (
            "Python peak bytes are measured by tracemalloc during the evaluation "
            "loop and exclude native PyTorch/C++ allocations. Model parameter, "
            "optimizer-state, and largest input-batch tensor storage are counted "
            "separately; these components are not a process RSS peak."
        ),
        "tokens": (
            "Question text is counted with the same case-folded regex tokenizer "
            "used by GraphQuestionEncoder. These counts are encoder input tokens; "
            "answer decoders emit entity IDs rather than generated language tokens."
        ),
    }
    report_path = destination / "task45_metrics.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _validate_frozen_inputs(
    facts: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    hashes: Mapping[str, str],
) -> None:
    if not isinstance(facts, (list, tuple)) or not facts:
        raise ValueError("facts must be a non-empty sequence")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("splits must contain train, validation, and test")
    if dict(hashes) != _FROZEN_HASHES:
        raise ValueError("input hashes do not match the frozen Task 35 artifacts")
    if len(facts) != 2000:
        raise ValueError("frozen pilot facts must contain exactly 2,000 records")
    for split, expected_count in _FROZEN_COUNTS.items():
        records = splits[split]
        if not isinstance(records, (list, tuple)) or len(records) != expected_count:
            raise ValueError(f"frozen {split} split must contain {expected_count} records")
        if any(record.get("split") != split for record in records):
            raise ValueError(f"every record in {split} must declare that split")


def _validate_frozen_configs(
    training: Mapping[str, Any],
    model: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> None:
    if training.get("objective") != "answer_only":
        raise ValueError("training objective must be answer_only")
    expected_training = {
        "device": "cpu",
        "deterministic": True,
        "num_threads": 2,
        "max_epochs": 10,
        "batch_size": 16,
        "max_steps": 2000,
        "max_runtime_seconds": 900,
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected:
            raise ValueError(f"training.{name} must match the frozen value {expected!r}")
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping) or dict(optimizer) != {
        "name": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }:
        raise ValueError("optimizer must match the frozen AdamW settings")
    validation = training.get("validation")
    if not isinstance(validation, Mapping) or dict(validation) != {
        "every_epochs": 1,
        "early_stopping_patience": 3,
    }:
        raise ValueError("validation must match the frozen selection settings")
    encoder = model.get("encoder")
    answer_head = model.get("answer_head")
    selr = model.get("selr")
    if not isinstance(encoder, Mapping) or dict(encoder) != {
        "type": "entity_relation_time_embeddings",
        "embedding_dim": 64,
        "aggregation": "mean",
        "dropout": 0.1,
    }:
        raise ValueError("encoder config must match the frozen policy")
    if not isinstance(answer_head, Mapping) or dict(answer_head) != {
        "type": "candidate_scoring",
        "candidate_policy": "retrieved_entities",
        "max_candidates": 128,
    }:
        raise ValueError("answer-head config must match the frozen candidate policy")
    if not isinstance(selr, Mapping) or dict(selr) != {
        "enabled": False,
        "latent_dim": 64,
        "max_steps": 3,
    }:
        raise ValueError("model config must contain the frozen baseline SELR section")
    if (
        retrieval.get("strategy") != "relation_constrained_bfs"
        or retrieval.get("max_hops") != 2
        or retrieval.get("max_nodes") != 128
        or retrieval.get("max_edges") != 256
    ):
        raise ValueError("retrieval config must match the frozen settings")


def _variant_model_config(
    config: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    result = dict(config)
    result["variant"] = variant
    selr = dict(config["selr"])
    selr["enabled"] = variant == "selr"
    result["selr"] = selr
    return result


def _training_run_record(
    seed: int,
    variant: str,
    result: TrainingResult,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "variant": variant,
        "global_step": result.global_step,
        "epochs_completed": result.epochs_completed,
        "best_validation_exact_match": result.best_validation_exact_match,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": elapsed_seconds,
        "best_checkpoint": result.best_checkpoint,
        "latest_checkpoint": result.latest_checkpoint,
        "history": list(result.history),
    }


def _evaluate_checkpoint(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    variant: str,
    checkpoint_path: str,
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(
        seed,
        deterministic=training_config["deterministic"],
        num_threads=training_config["num_threads"],
    )
    graph = build_temporal_graph(facts)
    vocabulary = build_encoder_vocabulary(graph, train_records)
    encoder_config = model_config["encoder"]
    answer_head_config = model_config["answer_head"]
    encoder_dim = int(encoder_config["embedding_dim"])
    encoder = GraphQuestionEncoder(
        vocabulary,
        embedding_dim=encoder_dim,
        dropout=float(encoder_config["dropout"]),
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model = (
        BaselineAnswerModel(encoder_dim)
        if variant == "baseline"
        else SELRAnswerModel(
            encoder_dim,
            int(model_config["selr"]["latent_dim"]),
            int(model_config["selr"]["max_steps"]),
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["optimizer"]["learning_rate"],
        weight_decay=training_config["optimizer"]["weight_decay"],
    )
    load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        expected_config=training_config,
        expected_seed=seed,
    )
    graph_digest = hashlib.sha256(
        json.dumps(facts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset = EncodedQADataset(
        graph,
        encoder,
        test_records,
        split="test",
        graph_id=graph_digest,
        retrieval_settings={
            name: retrieval_config[name]
            for name in ("strategy", "max_hops", "max_nodes", "max_edges")
        },
        max_candidates=int(answer_head_config["max_candidates"]),
    )
    batches = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=collate_encoded_examples,
    )
    predictions: list[dict[str, Any]] = []
    model.eval()
    owns_tracemalloc = not tracemalloc.is_tracing()
    if owns_tracemalloc:
        tracemalloc.start()
    tracemalloc.reset_peak()
    batch_latencies: list[float] = []
    max_batch_tensor_bytes = 0
    evaluation_started = time.perf_counter()
    try:
        iterator = iter(batches)
        with torch.no_grad():
            while True:
                batch_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                max_batch_tensor_bytes = max(
                    max_batch_tensor_bytes, _tensor_storage_bytes(batch)
                )
                logits = model(batch)
                probabilities = torch.sigmoid(logits)
                for row, (qa_id, candidate_ids) in enumerate(
                    zip(batch["qa_ids"], batch["candidate_ids"], strict=True)
                ):
                    count = len(candidate_ids)
                    row_probabilities = probabilities[row, :count]
                    if count == 0:
                        raise ValueError(f"test QA {qa_id!r} has no candidate answers")
                    predicted = [
                        candidate_id
                        for candidate_id, score in zip(
                            candidate_ids,
                            row_probabilities.tolist(),
                            strict=True,
                        )
                        if score >= 0.5
                    ]
                    if not predicted:
                        predicted = [
                            candidate_ids[int(row_probabilities.argmax().item())]
                        ]
                    predictions.append(
                        {
                            "qa_id": qa_id,
                            "model_variant": variant,
                            "status": "ok",
                            "predicted_answer_ids": predicted,
                            "answer_scores": [
                                {"answer_id": candidate_id, "score": float(score)}
                                for candidate_id, score in zip(
                                    candidate_ids,
                                    row_probabilities.tolist(),
                                    strict=True,
                                )
                            ],
                        }
                    )
                batch_latencies.append(time.perf_counter() - batch_started)
        elapsed = time.perf_counter() - evaluation_started
        _, python_heap_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if owns_tracemalloc:
            tracemalloc.stop()

    example_count = len(predictions)
    sorted_batch_latencies = sorted(batch_latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_batch_latencies)) - 1)
    measurement = {
        "latency": {
            "evaluation_wall_seconds": elapsed,
            "example_count": example_count,
            "examples_per_second": example_count / elapsed if elapsed else 0.0,
            "amortized_ms_per_example": (
                elapsed * 1000 / example_count if example_count else 0.0
            ),
            "batch_count": len(batch_latencies),
            "mean_batch_wall_ms": (
                statistics.mean(batch_latencies) * 1000 if batch_latencies else 0.0
            ),
            "p95_batch_wall_ms": (
                sorted_batch_latencies[p95_index] * 1000
                if sorted_batch_latencies
                else 0.0
            ),
        },
        "memory": {
            "python_heap_peak_bytes": python_heap_peak_bytes,
            "model_parameter_bytes": _tensor_storage_bytes(
                (tuple(encoder.parameters()), tuple(model.parameters()))
            ),
            "optimizer_state_tensor_bytes": _tensor_storage_bytes(optimizer.state),
            "max_batch_tensor_storage_bytes": max_batch_tensor_bytes,
        },
    }
    return predictions, measurement


def _prediction_signature(
    predictions: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, ...]]:
    return [
        (
            prediction.get("qa_id"),
            prediction.get("status"),
            tuple(prediction.get("predicted_answer_ids", ())),
            tuple(
                (score.get("answer_id"), score.get("score"))
                for score in prediction.get("answer_scores", ())
            ),
        )
        for prediction in predictions
    ]


def _tensor_storage_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_storage_bytes(item) for item in value)
    return 0


def _metric_summary(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "exact_match_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = [float(item[field]) for item in metrics]
        result[field] = {
            "per_seed": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def _read_json_array(path: Path, field: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field} JSON from {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} at {path} must be a non-empty JSON array")
    if any(not isinstance(record, dict) for record in value):
        raise ValueError(f"every {field} record at {path} must be a JSON object")
    return value


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a JSON object")
    return value


def _read_yaml_mapping(path: Path, field: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read {field} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a YAML object")
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    source_root = root / "src"
    digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
