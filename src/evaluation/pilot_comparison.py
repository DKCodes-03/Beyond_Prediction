"""Training/validation-only pilot for harder benchmarks and SELR ablations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.synthetic_benchmark import DIFFICULTY_SLICES


DEFAULT_LATENT_STEPS = (1, 2, 3, 4)


def _require_records(records: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(f"{name} must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} record {index} must be a mapping")
        normalized.append(dict(record))
    return normalized


def difficulty_slice_counts(
    qa_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Partition QA records into the frozen difficulty slice definitions."""
    records = _require_records(qa_records, "qa_records")
    counts = {slice_id: 0 for slice_id in DIFFICULTY_SLICES}
    memberships = {slice_id: [] for slice_id in DIFFICULTY_SLICES}

    cutoff_values = sorted(
        {
            int(str(cutoff).replace("-", ""))
            for record in records
            if (cutoff := _record_cutoff(record)) is not None
        }
    )
    cutoff_bounds: tuple[int, int] = (0, 99999999)
    if cutoff_values:
        low_index = (len(cutoff_values) + 2) // 3 - 1
        high_index = (2 * len(cutoff_values) + 2) // 3 - 1
        low = cutoff_values[low_index]
        high = cutoff_values[high_index]
        cutoff_bounds = (low, high)

    for record in records:
        relations = record.get("relation_sequence") or ()
        hop_count = len(relations)
        answer_ids = record.get("accepted_answer_ids") or ()
        try:
            record_cutoff = _record_cutoff(record)
            cutoff = (
                int(str(record_cutoff).replace("-", ""))
                if record_cutoff is not None
                else 0
            )
        except (TypeError, ValueError):
            cutoff = 0

        membership: list[str] = []
        if hop_count == 1:
            membership.append("one_hop")
        elif hop_count == 2:
            membership.append("two_hop")
        elif hop_count >= 3:
            membership.append("three_or_more_hops")

        if len(answer_ids) <= 2:
            membership.append("sparse_candidates")
        else:
            membership.append("dense_candidates")

        if cutoff_values:
            low, high = cutoff_bounds
            if cutoff <= low:
                membership.append("early_cutoff")
            elif cutoff <= high:
                membership.append("middle_cutoff")
            else:
                membership.append("late_cutoff")

        for slice_id in membership:
            counts[slice_id] += 1
            qa_id = record.get("qa_id")
            if isinstance(qa_id, (str, int)):
                memberships[slice_id].append(str(qa_id))

    return {
        slice_id: {
            "record_count": counts[slice_id],
            "qa_ids": memberships[slice_id],
        }
        for slice_id in DIFFICULTY_SLICES
    }


def _record_cutoff(record: Mapping[str, Any]) -> Any:
    if record.get("cutoff") is not None:
        return record["cutoff"]
    constraint = record.get("temporal_constraint")
    if isinstance(constraint, Mapping):
        return constraint.get("cutoff_date")
    return None


def pilot_difficulty_settings(
    qa_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return descriptive pilot slices plus matched retrieval settings."""
    records = _require_records(qa_records, "qa_records")
    slices = difficulty_slice_counts(records)
    return {
        "status": "pilot",
        "difficulty_slices": slices,
        "difficulty_slice_ids": list(DIFFICULTY_SLICES),
        "matched_retrieval_conditions": [
            {
                "condition_id": "unlimited",
                "max_hops": 4,
                "max_nodes": 128,
                "max_edges": 512,
            },
            {
                "condition_id": "node_limited",
                "max_hops": 4,
                "max_nodes": 32,
                "max_edges": 512,
            },
            {
                "condition_id": "edge_limited",
                "max_hops": 4,
                "max_nodes": 128,
                "max_edges": 128,
            },
        ],
    }


def define_latent_step_ablations(
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> list[dict[str, Any]]:
    """Define separate SELR latent-step budget variants."""
    if not isinstance(latent_steps, (list, tuple)) or not latent_steps:
        raise ValueError("latent_steps must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for steps in latent_steps:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("every latent-step budget must be a positive integer")
        if steps in seen:
            raise ValueError("latent-step budgets must be unique")
        seen.add(steps)
        normalized.append(
            {
                "ablation_id": f"selr_latent_steps_{steps}",
                "ablation_type": "latent_steps",
                "bottleneck_enabled": True,
                "max_steps": steps,
            }
        )
    return normalized


def define_bottleneck_ablations() -> list[dict[str, Any]]:
    """Define the two isolated bottleneck-state ablations."""
    return [
        {
            "ablation_id": "selr_bottleneck_enabled",
            "ablation_type": "bottleneck",
            "bottleneck_enabled": True,
        },
        {
            "ablation_id": "selr_bottleneck_disabled",
            "ablation_type": "bottleneck",
            "bottleneck_enabled": False,
        },
    ]


def build_pilot_configurations(
    *,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    """Build matched baseline, difficulty, and ablation pilot configurations."""
    latent_ablations = define_latent_step_ablations(latent_steps)
    bottleneck_ablations = define_bottleneck_ablations()
    return {
        "baseline": {
            "configuration_id": "matched_non_selr_baseline",
            "model_variant": "baseline",
            "bottleneck_enabled": False,
        },
        "selr": {
            "configuration_id": "matched_selr",
            "model_variant": "selr",
            "bottleneck_enabled": True,
        },
        "latent_step_ablations": latent_ablations,
        "bottleneck_ablations": bottleneck_ablations,
        "selection_protocol": {
            "allowed_splits": ["train", "validation"],
            "selection_split": "validation",
            "test_evaluation_permitted": False,
        },
    }


def pilot_learned_comparison(
    qa_records: Sequence[Mapping[str, Any]],
    *,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    """Descriptive alias for building a Task 65 pilot comparison."""
    return build_pilot_report(
        qa_records,
        latent_steps=latent_steps,
    )


def build_pilot_report(
    qa_records: Sequence[Mapping[str, Any]],
    *,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    """Build a report without any test-split data or test evaluation."""
    records = _require_records(qa_records, "qa_records")
    splits = {str(record.get("split")) for record in records if record.get("split")}
    if "test" in splits:
        raise ValueError("pilot comparison must not receive test-split records")
    if splits and not splits.issubset({"train", "validation"}):
        raise ValueError("pilot records must use only train or validation splits")

    configurations = build_pilot_configurations(latent_steps=latent_steps)
    difficulty = pilot_difficulty_settings(records)
    return {
        "benchmark": "task65_pilot",
        "status": "designed",
        "test_evaluation_performed": False,
        "allowed_splits": ["train", "validation"],
        "validation_only_selection": True,
        "difficulty": difficulty,
        "difficulty_settings": difficulty,
        "configurations": configurations,
        "ablation_settings": {
            "latent_step_ablations": configurations["latent_step_ablations"],
            "bottleneck_ablations": configurations["bottleneck_ablations"],
        },
        "matched_comparison": {
            "baseline": configurations["baseline"],
            "selr": configurations["selr"],
            "comparison_basis": "same training data, validation split, retrieval policy, and candidate policy",
        },
    }


def run_pilot_comparison(
    *,
    output_dir: str | Path,
    qa_records: Sequence[Mapping[str, Any]],
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    """Validate and export a Task 65 train/validation-only pilot report."""
    report = build_pilot_report(qa_records, latent_steps=latent_steps)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "pilot_comparison.json"
    temporary_path = destination / ".pilot_comparison.json.tmp"
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return {
        "report_path": str(report_path),
        "benchmark": report["benchmark"],
        "status": report["status"],
        "test_evaluation_performed": False,
        "difficulty_slice_count": len(report["difficulty"]["difficulty_slice_ids"]),
    }


# Compatibility aliases for scripts and external callers.
def define_pilot_ablations(
    *,
    latent_steps: Sequence[int] = DEFAULT_LATENT_STEPS,
) -> dict[str, Any]:
    return build_pilot_configurations(latent_steps=latent_steps)


design_pilot_comparison = pilot_learned_comparison
run_pilot_difficulty_comparison = run_pilot_comparison


def main(argv: Sequence[str] | None = None) -> int:
    """Export a Task 65 pilot report from a JSON QA-record array."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Design the train/validation-only Task 65 pilot."
    )
    parser.add_argument("--qa-records", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/task65_pilot"),
    )
    args = parser.parse_args(argv)
    qa_records = json.loads(args.qa_records.read_text(encoding="utf-8"))
    result = run_pilot_comparison(
        output_dir=args.output_dir,
        qa_records=qa_records,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
