"""Hard-synthetic benchmark design and reproducible artifact export."""

from __future__ import annotations

import json
import random
import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.graph.graph_utils import validate_fact
from src.ingestion.preprocess import preprocess_facts
from src.ingestion.qa_preparation import prepare_qa
from src.ingestion.split_generation import (
    create_disjoint_benchmark_splits,
    run_benchmark_split_preparation,
)


DIFFICULTY_SLICES = (
    "one_hop",
    "two_hop",
    "three_or_more_hops",
    "sparse_candidates",
    "dense_candidates",
    "early_cutoff",
    "middle_cutoff",
    "late_cutoff",
)

FACT_SCALE_COUNTS = (5000, 10000, 20000)
DEFAULT_FACT_SAMPLING_SEED = 17
DEFAULT_RESOURCE_LIMITS = {
    "max_resident_memory_bytes": 3221225472,
    "qa_max_hops": 2,
    "stop_on_resource_limit": True,
}


def generate_harder_synthetic_benchmark(
    *,
    seed: int = 61,
    seed_count: int = 4,
    branches: int = 3,
    max_hops: int = 4,
) -> dict[str, Any]:
    """Generate deterministic dense, multi-hop, temporally varied synthetic facts and QA."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value, minimum in (
        ("seed_count", seed_count, 1),
        ("branches", branches, 2),
        ("max_hops", max_hops, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = random.Random(seed)
    facts: list[dict[str, str]] = []
    base_date = 20200101
    fact_number = 0

    for seed_index in range(seed_count):
        seed_id = f"s{seed_index}"
        current_nodes = [seed_id]
        for hop in range(1, max_hops + 1):
            relation_id = f"r{hop}"
            next_nodes: list[str] = []
            for parent_index, parent_id in enumerate(current_nodes):
                for branch in range(branches):
                    if hop == max_hops:
                        node_id = f"a{seed_index}_{parent_index}_{branch}"
                    else:
                        node_id = f"n{seed_index}_{hop}_{parent_index}_{branch}"
                    fact_number += 1
                    date_value = base_date + (hop * 3) + rng.randint(0, 2)
                    timestamp = (
                        f"{date_value // 10000}-{(date_value // 100) % 100:02d}-{date_value % 100:02d}"
                    )
                    facts.append(
                        {
                            "fact_id": f"f{fact_number}",
                            "subject_id": parent_id,
                            "relation_id": relation_id,
                            "object_id": node_id,
                            "timestamp": timestamp,
                            "source_id": f"synthetic:{seed}",
                            "subject_label": parent_id,
                            "relation_label": f"relation_{hop}",
                            "object_label": node_id,
                        }
                    )
                    next_nodes.append(node_id)
            current_nodes = next_nodes

    qa_records = prepare_qa(facts, max_hops=max_hops)
    for index, record in enumerate(qa_records):
        remainder = index % 10
        record["split"] = (
            "test" if remainder in (8, 9)
            else "validation" if remainder == 7
            else "train"
        )
        record["provenance"] = {
            **record["provenance"],
            "benchmark": "harder_synthetic",
            "generator_seed": seed,
        }
    return {
        "benchmark": "hard_synthetic",
        "seed": seed,
        "facts": facts,
        "qa_records": qa_records,
    }


def design_synthetic_benchmark(
    facts: Sequence[Mapping[str, Any]] | None = None,
    qa_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    seed: int = 61,
    seed_count: int = 4,
    branches: int = 3,
    max_hops: int = 4,
    retrieval_settings: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Return a frozen-style benchmark definition with slices and matched conditions."""
    if facts is None or qa_records is None:
        generated = generate_harder_synthetic_benchmark(
            seed=seed,
            seed_count=seed_count,
            branches=branches,
            max_hops=max_hops,
        )
        facts = generated["facts"]
        qa_records = generated["qa_records"]
    if not isinstance(facts, (list, tuple)):
        raise ValueError("facts must be a sequence")
    if not isinstance(qa_records, (list, tuple)):
        raise ValueError("qa_records must be a sequence")
    for index, fact in enumerate(facts):
        validate_fact(fact, index)

    normalized_qa: list[dict[str, Any]] = []
    for index, record in enumerate(qa_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"QA record {index} must be a mapping")
        normalized_qa.append(dict(record))

    slices = _difficulty_slices(normalized_qa)
    settings = _retrieval_settings(retrieval_settings)
    matched_conditions = [
        {
            "condition_id": "matched_unlimited",
            "settings": settings["unlimited"],
            "comparison": "baseline_for_each_difficulty_slice",
        },
        {
            "condition_id": "matched_node_limited",
            "settings": settings["node_limited"],
            "comparison": "same_QA_and_seed_same_as_unlimited",
        },
        {
            "condition_id": "matched_edge_limited",
            "settings": settings["edge_limited"],
            "comparison": "same_QA_and_seed_same_as_unlimited",
        },
    ]
    return {
        "benchmark": "hard_synthetic",
        "seed": seed,
        "status": "designed",
        "fact_count": len(facts),
        "qa_count": len(normalized_qa),
        "difficulty_slices": slices,
        "difficulty_slice_ids": DIFFICULTY_SLICES,
        "matched_comparison_conditions": matched_conditions,
        "retrieval_limitations": settings,
        "facts": [dict(fact) for fact in facts],
        "qa_records": normalized_qa,
        "limitations": [
            "Synthetic graph paths are controlled and may not represent natural-language QA.",
            "Matched comparisons differ only by retrieval limits.",
            "No learned model or checkpoint is evaluated by benchmark design.",
        ],
    }


def benchmark_synthetic_difficulty(
    facts: Sequence[Mapping[str, Any]] | None = None,
    qa_records: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Descriptive alias for :func:`design_synthetic_benchmark`."""
    return design_synthetic_benchmark(facts, qa_records, **kwargs)


def run_synthetic_benchmark(
    *,
    output_dir: str | Path,
    facts: Sequence[Mapping[str, Any]] | None = None,
    qa_records: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 61,
    **kwargs: Any,
) -> dict[str, Any]:
    report = design_synthetic_benchmark(
        facts,
        qa_records,
        seed=seed,
        **kwargs,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "synthetic_benchmark.json"
    temporary_path = destination / ".synthetic_benchmark.json.tmp"
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return {
        "report_path": str(report_path),
        "benchmark": report["benchmark"],
        "fact_count": report["fact_count"],
        "qa_count": report["qa_count"],
        "difficulty_slice_count": len(report["difficulty_slices"]),
    }


def prepare_fact_scale_samples(
    raw_facts: Sequence[Mapping[str, Any]],
    *,
    sizes: Sequence[int] = FACT_SCALE_COUNTS,
    fact_sampling_seed: int = DEFAULT_FACT_SAMPLING_SEED,
    qa_seed: int = 61,
    max_hops: int = 2,
    resource_limits: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create deterministic 5K/10K/20K ICEWS-style fact samples and graph QA."""
    if not isinstance(raw_facts, (list, tuple)) or not raw_facts:
        raise ValueError("raw_facts must be a non-empty sequence")
    if isinstance(fact_sampling_seed, bool) or not isinstance(fact_sampling_seed, int):
        raise ValueError("fact_sampling_seed must be a non-negative integer")
    if fact_sampling_seed < 0:
        raise ValueError("fact_sampling_seed must be a non-negative integer")
    if isinstance(qa_seed, bool) or not isinstance(qa_seed, int) or qa_seed < 0:
        raise ValueError("qa_seed must be a non-negative integer")
    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise ValueError("max_hops must be an integer >= 1")
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError("sizes must be a non-empty sequence")

    normalized_sizes: list[int] = []
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("every fact sample size must be an integer >= 1")
        if size in normalized_sizes:
            raise ValueError("fact sample sizes must be unique")
        normalized_sizes.append(size)

    records = [dict(record) for record in raw_facts]
    if len(records) < max(normalized_sizes):
        raise ValueError(
            "raw_facts does not contain enough records for the largest requested "
            f"sample: need {max(normalized_sizes)}, got {len(records)}"
        )

    limits = dict(DEFAULT_RESOURCE_LIMITS)
    limits["qa_max_hops"] = max_hops
    if resource_limits is not None:
        if not isinstance(resource_limits, Mapping):
            raise ValueError("resource_limits must be a mapping")
        limits.update(dict(resource_limits))
        limits["qa_max_hops"] = max_hops

    provenance = dict(dataset or {})
    rng = random.Random(fact_sampling_seed)
    # One seeded shuffle makes every requested sample a prefix of the next size.
    sampled_indexes = list(range(len(records)))
    rng.shuffle(sampled_indexes)

    samples: dict[str, dict[str, Any]] = {}
    for size in sorted(normalized_sizes):
        sampled_rows = [records[index] for index in sampled_indexes[:size]]
        facts = preprocess_facts(sampled_rows)
        qa_records = prepare_qa(facts, max_hops=max_hops)
        for record in qa_records:
            record["provenance"] = {
                **record["provenance"],
                "benchmark": "scale_fact_sample",
                "fact_sampling_seed": fact_sampling_seed,
                "qa_generation_seed": qa_seed,
                "requested_fact_count": size,
                "canonical_fact_count": len(facts),
            }

        fact_text = _scale_json_text(facts)
        qa_text = _scale_json_text(qa_records)
        name = f"{size // 1000}k" if size % 1000 == 0 else f"{size}"
        samples[name] = {
            "sample_name": name,
            "requested_fact_count": size,
            "canonical_fact_count": len(facts),
            "qa_count": len(qa_records),
            "facts": facts,
            "qa_records": qa_records,
            "facts_sha256": hashlib.sha256(fact_text.encode("utf-8")).hexdigest(),
            "qa_sha256": hashlib.sha256(qa_text.encode("utf-8")).hexdigest(),
        }

    return {
        "benchmark": "fact_scale_samples",
        "status": "prepared",
        "seed": fact_sampling_seed,
        "fact_sampling_seed": fact_sampling_seed,
        "qa_generation_seed": qa_seed,
        "dataset": provenance,
        "resource_limits": limits,
        "sample_names": [f"{size // 1000}k" if size % 1000 == 0 else str(size)
                         for size in sorted(normalized_sizes)],
        "samples": samples,
    }


def run_fact_scale_sample_preparation(
    *,
    raw_facts: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    sizes: Sequence[int] = FACT_SCALE_COUNTS,
    fact_sampling_seed: int = DEFAULT_FACT_SAMPLING_SEED,
    qa_seed: int = 61,
    max_hops: int = 2,
    resource_limits: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export each scale sample and a single reproducibility manifest."""
    report = prepare_fact_scale_samples(
        raw_facts,
        sizes=sizes,
        fact_sampling_seed=fact_sampling_seed,
        qa_seed=qa_seed,
        max_hops=max_hops,
        resource_limits=resource_limits,
        dataset=dataset,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest_samples: dict[str, dict[str, Any]] = {}
    for name, sample in report["samples"].items():
        sample_dir = destination / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        _atomic_scale_write(
            sample_dir / "facts.json",
            _scale_json_text(sample["facts"]),
        )
        _atomic_scale_write(
            sample_dir / "qa_records.json",
            _scale_json_text(sample["qa_records"]),
        )
        sample_manifest = {
            "benchmark": "fact_scale_samples",
            "sample_name": name,
            "dataset": report["dataset"],
            "sampling": {
                "strategy": "uniform_without_replacement",
                "seed": fact_sampling_seed,
                "requested_fact_count": sample["requested_fact_count"],
                "canonical_fact_count": sample["canonical_fact_count"],
            },
            "qa": {
                "generation_seed": qa_seed,
                "max_hops": max_hops,
                "record_count": sample["qa_count"],
                "sha256": sample["qa_sha256"],
            },
            "resource_limits": report["resource_limits"],
            "artifacts": {
                "facts": {"path": "facts.json", "sha256": sample["facts_sha256"],
                          "record_count": sample["canonical_fact_count"]},
                "qa_records": {"path": "qa_records.json", "sha256": sample["qa_sha256"],
                               "record_count": sample["qa_count"]},
            },
        }
        sample_manifest_path = sample_dir / "manifest.json"
        _atomic_scale_write(
            sample_manifest_path,
            _scale_json_text(sample_manifest),
        )
        manifest_samples[name] = {
            key: value
            for key, value in sample.items()
            if key not in ("facts", "qa_records")
        }

    summary = {
        key: value
        for key, value in report.items()
        if key != "samples"
    }
    summary["samples"] = manifest_samples
    summary_path = destination / "scale_samples_manifest.json"
    _atomic_scale_write(summary_path, _scale_json_text(summary))
    return {
        "manifest_path": str(summary_path),
        "benchmark": report["benchmark"],
        "sample_names": report["sample_names"],
        "sample_count": len(report["samples"]),
    }


prepare_reproducible_fact_samples = prepare_fact_scale_samples
prepare_5k_10k_20k_samples = prepare_fact_scale_samples
prepare_fact_scale_sample = prepare_fact_scale_samples
run_scale_fact_sample_preparation = run_fact_scale_sample_preparation
run_fact_scale_benchmark = run_fact_scale_sample_preparation


def _scale_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_scale_write(path: Path, text: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(path)


def _retrieval_settings(
    supplied: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int | str]]:
    defaults: dict[str, dict[str, int | str]] = {
        "unlimited": {
            "strategy": "relation_constrained_bfs",
            "max_hops": 8,
            "max_nodes": 128,
            "max_edges": 256,
        },
        "node_limited": {
            "strategy": "relation_constrained_bfs",
            "max_hops": 8,
            "max_nodes": 16,
            "max_edges": 256,
        },
        "edge_limited": {
            "strategy": "relation_constrained_bfs",
            "max_hops": 8,
            "max_nodes": 128,
            "max_edges": 32,
        },
    }
    if supplied is None:
        return defaults
    if not isinstance(supplied, Mapping):
        raise ValueError("retrieval_settings must be a mapping")
    resolved = dict(defaults)
    for name, values in supplied.items():
        if name not in resolved:
            raise ValueError(f"unknown retrieval limitation {name!r}")
        if not isinstance(values, Mapping):
            raise ValueError(f"retrieval settings for {name} must be a mapping")
        resolved[name] = dict(resolved[name] | dict(values))
    return resolved


def _difficulty_slices(
    qa_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hop_groups: Counter[str] = Counter()
    cutoff_dates: list[str] = []
    answer_counts: list[int] = []
    qa_ids_by_slice: dict[str, list[str]] = defaultdict(list)

    for record in qa_records:
        relation_sequence = record.get("relation_sequence")
        accepted = record.get("accepted_answer_ids")
        temporal = record.get("temporal_constraint")
        if not isinstance(relation_sequence, (list, tuple)):
            continue
        hop_count = len(relation_sequence)
        hop_slice = "one_hop" if hop_count == 1 else (
            "two_hop" if hop_count == 2 else "three_or_more_hops"
        )
        hop_groups[hop_slice] += 1
        qa_id = str(record.get("qa_id", ""))
        qa_ids_by_slice[hop_slice].append(qa_id)
        if isinstance(accepted, (list, tuple)):
            answer_count = len(set(accepted))
            answer_counts.append(answer_count)
            candidate_slice = "sparse_candidates" if answer_count <= 3 else "dense_candidates"
            qa_ids_by_slice[candidate_slice].append(qa_id)
        if isinstance(temporal, Mapping):
            cutoff = temporal.get("cutoff_date")
            if isinstance(cutoff, str):
                cutoff_dates.append(cutoff)

    cutoff_slice_counts: Counter[str] = Counter()
    if cutoff_dates:
        ordered = sorted(set(cutoff_dates))
        if len(ordered) >= 3:
            split = len(ordered) // 3
            early = set(ordered[:split])
            late_start = split * 2
            middle = set(ordered[split:late_start])
            late = set(ordered[late_start:])
        else:
            early = {ordered[0]}
            middle = set(ordered[1:-1])
            late = {ordered[-1]} if len(ordered) > 1 else set()
        for record in qa_records:
            temporal = record.get("temporal_constraint")
            cutoff = temporal.get("cutoff_date") if isinstance(temporal, Mapping) else None
            qa_id = str(record.get("qa_id", ""))
            if cutoff in early:
                cutoff_slice_counts["early_cutoff"] += 1
                qa_ids_by_slice["early_cutoff"].append(qa_id)
            elif cutoff in middle:
                cutoff_slice_counts["middle_cutoff"] += 1
                qa_ids_by_slice["middle_cutoff"].append(qa_id)
            elif cutoff in late:
                cutoff_slice_counts["late_cutoff"] += 1
                qa_ids_by_slice["late_cutoff"].append(qa_id)

    return {
        slice_id: {
            "record_count": (
                hop_groups[slice_id] if slice_id in hop_groups
                else cutoff_slice_counts[slice_id]
                if slice_id in cutoff_slice_counts
                else sum(
                    1 for qa_id in qa_ids_by_slice[slice_id]
                )
            ),
            "qa_ids": sorted(set(qa_ids_by_slice[slice_id])),
        }
        for slice_id in DIFFICULTY_SLICES
    }


# Compatibility names for callers that describe the benchmark by generation
# or matched-comparison rather than "design".
build_harder_synthetic_benchmark = generate_harder_synthetic_benchmark
create_harder_synthetic_benchmark = generate_harder_synthetic_benchmark
design_harder_synthetic_benchmark = design_synthetic_benchmark
build_synthetic_benchmark = design_synthetic_benchmark
prepare_benchmark_splits = create_disjoint_benchmark_splits
create_benchmark_splits = create_disjoint_benchmark_splits
split_new_benchmark = create_disjoint_benchmark_splits
create_disjoint_splits = create_disjoint_benchmark_splits

