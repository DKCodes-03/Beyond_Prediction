"""CLI and helpers for evaluating the graph-path sanity baseline."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_answers import write_evaluation_results
from src.evaluation.metrics import compute_answer_metrics
from src.evaluation.path_baseline import graph_path_sanity_baseline
from src.graph.graph_builder import build_temporal_graph

# Re-export the isolated optional-integration benchmark so existing benchmark
# entry points can discover it without making core evaluation depend on it.
from src.evaluation.integration_benchmark import (  # noqa: E402
    benchmark_adopted_integrations,
    benchmark_icews18_transfer,
    benchmark_optional_integrations,
    run_integration_benchmark,
)
from src.evaluation.benchmark_verification import (  # noqa: E402
    independently_verify_benchmark,
    run_benchmark_verification,
)
from src.ingestion.split_generation import (  # noqa: E402
    create_disjoint_benchmark_splits,
    prepare_disjoint_benchmark_splits,
    run_benchmark_split_preparation,
)


_SPLITS = frozenset({"train", "validation", "test"})


def run_path_baseline_benchmark(
    facts: Sequence[Mapping[str, Any]],
    qa_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    split: str,
    seed: int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and export the deterministic graph-path baseline for one split."""
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(qa_records, (list, tuple)) or not qa_records:
        raise ValueError("qa_records must be a non-empty list")
    for index, record in enumerate(qa_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"QA record {index} must be a mapping")
        if record.get("split") != split:
            raise ValueError(
                f"QA record {index} split must match requested split {split!r}"
            )

    graph = build_temporal_graph(facts)
    prediction_records: list[dict[str, Any]] = []
    for record in qa_records:
        result = graph_path_sanity_baseline(graph, record)
        prediction: dict[str, Any] = {
            "qa_id": result["qa_id"],
            "model_variant": "graph_path_sanity",
            "status": result["status"],
            "predicted_answer_ids": result["predicted_answer_ids"],
            "answer_scores": [],
        }
        if result["status"] != "ok":
            prediction["failure_reason"] = (
                "graph traversal returned no answers under the query constraints"
            )
        prediction_records.append(prediction)

    metrics = compute_answer_metrics(qa_records, prediction_records)
    run_info = {
        "benchmark": "graph_path_sanity",
        "seed": seed,
        "split": split,
        "config": dict(config or {}),
        "seed_usage": "recorded; graph traversal is deterministic and uses no RNG",
    }
    predictions_path, metrics_path, run_path = write_evaluation_results(
        output_dir, prediction_records, metrics, run_info
    )
    return {
        "metrics": metrics,
        "prediction_count": len(prediction_records),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "run_path": str(run_path),
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic graph-path sanity baseline."
    )
    parser.add_argument("--facts", type=Path, default=Path("data/processed/facts.json"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument(
        "--split", choices=sorted(_SPLITS), default="test"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/graph_path_sanity"),
    )
    args = parser.parse_args(argv)
    split_file = args.split_file or Path("data/splits") / {
        "train": "train.json",
        "validation": "val.json",
        "test": "test.json",
    }[args.split]

    facts = _read_json_array(args.facts, "facts")
    qa_records = _read_json_array(split_file, "QA split")
    result = run_path_baseline_benchmark(
        facts,
        qa_records,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
        config={
            "facts": str(args.facts),
            "split_file": str(split_file),
            "model_variant": "graph_path_sanity",
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
