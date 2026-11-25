"""Validation-only ablation pilot and paired repeated training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.pilot_comparison import (
    DEFAULT_LATENT_STEPS,
    build_pilot_report,
    difficulty_slice_counts,
)
from src.training.train_baseline import run_baseline_training
from src.training.train_selr import run_selr_training
from src.training.trainer import TrainingResult
from src.utils.logging_utils import detect_code_revision


DEFAULT_SEEDS = (17, 23, 42)


def build_training_configurations(
    model_config: Mapping[str, Any],
    *,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> list[dict[str, Any]]:
    """Return unique baseline, SELR, latent-step, and bottleneck variants."""
    selr_config = model_config.get("selr")
    if not isinstance(selr_config, Mapping):
        raise ValueError("model_config.selr must be a mapping")
    standard_steps = _positive_int(selr_config.get("max_steps"), "selr.max_steps")
    if not isinstance(latent_steps, (list, tuple)) or not latent_steps:
        raise ValueError("latent_steps must be a non-empty sequence")
    normalized_steps: list[int] = []
    for steps in latent_steps:
        normalized_steps.append(_positive_int(steps, "latent_steps"))
    if len(set(normalized_steps)) != len(normalized_steps):
        raise ValueError("latent_steps must be unique")

    configurations = [
        {
            "configuration_id": "matched_non_selr_baseline",
            "model_variant": "baseline",
            "ablation_type": "baseline",
            "max_steps": None,
            "bottleneck_enabled": False,
        },
        {
            "configuration_id": "matched_selr",
            "model_variant": "selr",
            "ablation_type": "selr",
            "max_steps": standard_steps,
            "bottleneck_enabled": True,
        },
    ]
    for steps in normalized_steps:
        if steps == standard_steps:
            continue
        configurations.append(
            {
                "configuration_id": f"selr_latent_steps_{steps}",
                "model_variant": "selr",
                "ablation_type": "latent_steps",
                "max_steps": steps,
                "bottleneck_enabled": True,
            }
        )
    configurations.append(
        {
            "configuration_id": "selr_bottleneck_disabled",
            "model_variant": "selr",
            "ablation_type": "bottleneck",
            "max_steps": standard_steps,
            "bottleneck_enabled": False,
        }
    )
    return configurations


def run_task65_pilot(
    *,
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    output_dir: str | Path,
    seed: int = 61,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    """Train each pilot configuration and select settings using validation only."""
    _validate_train_validation(train_records, validation_records)
    if not facts:
        raise ValueError("facts must be a non-empty sequence")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_retrieval_hops(
        [*train_records, *validation_records],
        retrieval_config,
    )

    configurations = build_training_configurations(
        model_config,
        latent_steps=latent_steps,
    )
    qa_records = [*train_records, *validation_records]
    design = build_pilot_report(qa_records, latent_steps=latent_steps)
    destination = Path(output_dir)
    result_path = destination / "pilot_results.json"
    if result_path.exists():
        raise FileExistsError(
            f"{result_path} already exists; choose a fresh output directory"
        )
    destination.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        **design,
        "status": "training",
        "pilot_seed": seed,
        "test_evaluation_performed": False,
        "split_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "validation_difficulty_slices": difficulty_slice_counts(validation_records),
        "training_configurations": configurations,
        "training_config": dict(training_config),
        "model_config": copy.deepcopy(dict(model_config)),
        "retrieval_config": dict(retrieval_config),
        "runs": [],
        "input_sha256": _input_hashes(facts, train_records, validation_records),
        "selection_metric": "best_validation_exact_match",
    }
    _atomic_write(result_path, summary)

    for configuration in configurations:
        run_training = dict(training_config)
        run_training["seed"] = seed
        run_training["model_variant"] = configuration["model_variant"]
        run_training["checkpoint_dir"] = str(
            destination / configuration["configuration_id"] / "checkpoints"
        )
        run_training["log_dir"] = str(
            destination / configuration["configuration_id"] / "logs"
        )
        run_model = _model_config_for_configuration(model_config, configuration)
        started_at = time.monotonic()
        result = _train_configuration(
            facts,
            train_records,
            validation_records,
            training_config=run_training,
            model_config=run_model,
            retrieval_config=retrieval_config,
            runtime_started_at=started_at,
        )
        run_record = _training_record(
            configuration,
            seed,
            result,
            elapsed_seconds=time.monotonic() - started_at,
        )
        summary["runs"].append(run_record)
        _atomic_write(result_path, summary)
        if run_record["stop_reason"] == "max_runtime":
            summary["status"] = "incomplete"
            _atomic_write(result_path, summary)
            return summary

    by_id = {run["configuration_id"]: run for run in summary["runs"]}
    latent_candidates = [
        run for run in summary["runs"] if run["ablation_type"] == "latent_steps"
    ]
    if not latent_candidates:
        raise ValueError("pilot requires at least one non-default latent-step ablation")
    selected_latent = max(
        latent_candidates,
        key=lambda run: (
            run["best_validation_exact_match"],
            -int(run["max_steps"]),
        ),
    )
    selected = [
        by_id["matched_non_selr_baseline"],
        by_id["matched_selr"],
        selected_latent,
        by_id["selr_bottleneck_disabled"],
    ]
    summary["selected_configurations"] = [
        _selected_configuration(run) for run in selected
    ]
    summary["selection"] = {
        "split": "validation",
        "metric": "exact_match_accuracy",
        "latent_step_tie_break": "smaller max_steps",
        "bottleneck_ablation": (
            "masked mean pooling replaces learned latent-slot cross-attention"
        ),
    }
    summary["status"] = "complete"
    _atomic_write(result_path, summary)
    return summary


def run_repeated_training(
    *,
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    selected_configurations: Sequence[Mapping[str, Any]],
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    output_dir: str | Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Train selected configurations over paired seeds without reading test data."""
    _validate_train_validation(train_records, validation_records)
    if not facts:
        raise ValueError("facts must be a non-empty sequence")
    _validate_retrieval_hops(
        [*train_records, *validation_records],
        retrieval_config,
    )
    if not isinstance(selected_configurations, (list, tuple)) or not selected_configurations:
        raise ValueError("selected_configurations must be a non-empty sequence")
    if not isinstance(seeds, (list, tuple)) or len(seeds) < 2:
        raise ValueError("at least two paired seeds are required")
    normalized_seeds = [_non_negative_int(seed, "seed") for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")

    configurations = [dict(config) for config in selected_configurations]
    identifiers = [config.get("configuration_id") for config in configurations]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("every configuration requires a non-empty configuration_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("configuration_id values must be unique")
    if not any(config.get("model_variant") == "baseline" for config in configurations):
        raise ValueError("selected configurations must include the baseline")
    if not any(config.get("model_variant") == "selr" for config in configurations):
        raise ValueError("selected configurations must include SELR")

    destination = Path(output_dir)
    result_path = destination / "repeated_training.json"
    if result_path.exists():
        raise FileExistsError(
            f"{result_path} already exists; choose a fresh output directory"
        )
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": "task66_repeated_training",
        "status": "training",
        "seed_policy": "paired",
        "seeds": normalized_seeds,
        "selected_configurations": configurations,
        "training_config": dict(training_config),
        "model_config": copy.deepcopy(dict(model_config)),
        "retrieval_config": dict(retrieval_config),
        "split_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
        },
        "input_sha256": _input_hashes(facts, train_records, validation_records),
        "code_revision": detect_code_revision(),
        "test_evaluation_performed": False,
        "checkpoint_selection": {
            "split": "validation",
            "metric": "exact_match_accuracy",
        },
        "runs": [],
    }
    _atomic_write(result_path, summary)

    for seed in normalized_seeds:
        for configuration in configurations:
            run_training = dict(training_config)
            run_training["seed"] = seed
            run_training["model_variant"] = configuration["model_variant"]
            run_name = f"{configuration['configuration_id']}_seed_{seed}"
            run_training["checkpoint_dir"] = str(
                destination / run_name / "checkpoints"
            )
            run_training["log_dir"] = str(destination / run_name / "logs")
            run_model = _model_config_for_configuration(
                model_config,
                configuration,
            )
            started_at = time.monotonic()
            result = _train_configuration(
                facts,
                train_records,
                validation_records,
                training_config=run_training,
                model_config=run_model,
                retrieval_config=retrieval_config,
                runtime_started_at=started_at,
            )
            run_record = _training_record(
                configuration,
                seed,
                result,
                elapsed_seconds=time.monotonic() - started_at,
            )
            summary["runs"].append(run_record)
            _atomic_write(result_path, summary)
            if (
                run_record["stop_reason"] == "max_runtime"
                or run_record["elapsed_seconds"]
                > run_training["max_runtime_seconds"]
            ):
                summary["status"] = "incomplete"
                summary["incomplete_run"] = {
                    "configuration_id": configuration["configuration_id"],
                    "seed": seed,
                }
                _atomic_write(result_path, summary)
                return summary

    summary["status"] = "complete"
    _atomic_write(result_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Task 65 validation-only pilot and Task 66 paired training."
    )
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("configs/retrieval_config.yaml"))
    parser.add_argument("--pilot-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--max-hops",
        type=int,
        help="Override the retrieval config's max_hops for this benchmark.",
    )
    args = parser.parse_args(argv)

    facts = _read_json_array(args.facts, "facts")
    train_records = _read_json_array(args.train_split, "training split")
    validation_records = _read_json_array(args.validation_split, "validation split")
    training_config = _read_yaml_mapping(args.training_config, "training config")
    model_config = _read_yaml_mapping(args.model_config, "model config")
    retrieval_config = _read_yaml_mapping(args.retrieval_config, "retrieval config")
    if args.max_hops is not None:
        retrieval_config["max_hops"] = args.max_hops
    if args.pilot_results is None:
        result = run_task65_pilot(
            facts=facts,
            train_records=train_records,
            validation_records=validation_records,
            training_config=training_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        pilot = _read_json_object(args.pilot_results, "pilot results")
        selected = pilot.get("selected_configurations")
        result = run_repeated_training(
            facts=facts,
            train_records=train_records,
            validation_records=validation_records,
            selected_configurations=selected,
            training_config=training_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            output_dir=args.output_dir,
            seeds=args.seeds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _train_configuration(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    *,
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    retrieval_config: Mapping[str, Any],
    runtime_started_at: float,
) -> TrainingResult:
    if training_config["model_variant"] == "baseline":
        return run_baseline_training(
            facts,
            train_records,
            validation_records,
            training_config=training_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            runtime_started_at=runtime_started_at,
        )
    return run_selr_training(
        facts,
        train_records,
        validation_records,
        training_config=training_config,
        model_config=model_config,
        retrieval_config=retrieval_config,
        runtime_started_at=runtime_started_at,
    )


def _model_config_for_configuration(
    model_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(model_config))
    variant = configuration.get("model_variant")
    if variant not in {"baseline", "selr"}:
        raise ValueError("model_variant must be baseline or selr")
    result["variant"] = variant
    selr = result.get("selr")
    if not isinstance(selr, dict):
        raise ValueError("model_config.selr must be a mapping")
    selr["enabled"] = variant == "selr"
    if variant == "selr":
        selr["max_steps"] = _positive_int(
            configuration.get("max_steps"),
            "configuration.max_steps",
        )
        bottleneck_enabled = configuration.get("bottleneck_enabled", True)
        if not isinstance(bottleneck_enabled, bool):
            raise ValueError("configuration.bottleneck_enabled must be a boolean")
        selr["bottleneck_enabled"] = bottleneck_enabled
    return result


def _training_record(
    configuration: Mapping[str, Any],
    seed: int,
    result: TrainingResult,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "configuration_id": configuration["configuration_id"],
        "model_variant": configuration["model_variant"],
        "ablation_type": configuration.get("ablation_type"),
        "max_steps": configuration.get("max_steps"),
        "bottleneck_enabled": configuration.get("bottleneck_enabled"),
        "seed": seed,
        "best_validation_exact_match": result.best_validation_exact_match,
        "global_step": result.global_step,
        "epochs_completed": result.epochs_completed,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": elapsed_seconds,
        "best_checkpoint": result.best_checkpoint,
        "latest_checkpoint": result.latest_checkpoint,
        "history": list(result.history),
    }


def _selected_configuration(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: run[key]
        for key in (
            "configuration_id",
            "model_variant",
            "ablation_type",
            "max_steps",
            "bottleneck_enabled",
        )
    }


def _validate_train_validation(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    for split_name, records in (
        ("train", train_records),
        ("validation", validation_records),
    ):
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError(f"{split_name}_records must be a non-empty sequence")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"{split_name} record {index} must be a mapping")
            if record.get("split") != split_name:
                raise ValueError(f"{split_name} record {index} must have split={split_name!r}")


def _validate_retrieval_hops(
    records: Sequence[Mapping[str, Any]],
    retrieval_config: Mapping[str, Any],
) -> None:
    max_hops = retrieval_config.get("max_hops")
    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise ValueError("retrieval_config.max_hops must be a positive integer")
    required_hops = max(
        (len(record.get("relation_sequence", ())) for record in records),
        default=0,
    )
    if max_hops < required_hops:
        raise ValueError(
            "retrieval_config.max_hops is smaller than the training/validation "
            f"query hop count ({max_hops} < {required_hops})"
        )


def _input_hashes(
    facts: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        name: hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest()
        for name, records in (
            ("facts", facts),
            ("train", train_records),
            ("validation", validation_records),
        )
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_array(path: Path, field: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} at {path} must be a non-empty array of objects")
    return value


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a JSON object")
    return value


def _read_yaml_mapping(path: Path, field: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a YAML object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
