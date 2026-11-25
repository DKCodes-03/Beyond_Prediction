"""Verify prepared fact-scale sample QA and export bounded-memory summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_verification import run_benchmark_verification


def verify_scale_samples(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    sample_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify selected samples using aggregate-only reports and input hashes."""
    manifest_file = Path(manifest_path)
    manifest = _read_json_object(manifest_file, "scale sample manifest")
    all_names = manifest.get("sample_names")
    sample_manifest = manifest.get("samples")
    if not isinstance(all_names, list) or not all_names:
        raise ValueError("scale sample manifest sample_names must be a non-empty array")
    if not isinstance(sample_manifest, Mapping):
        raise ValueError("scale sample manifest samples must be an object")

    requested_names = list(all_names if sample_names is None else sample_names)
    if not requested_names or len(set(requested_names)) != len(requested_names):
        raise ValueError("sample_names must be non-empty and unique")
    unknown_names = sorted(set(requested_names) - set(all_names))
    if unknown_names:
        raise ValueError(f"unknown scale sample(s): {unknown_names!r}")

    source_dir = manifest_file.parent
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "scale_verification_summary.json"
    summary: dict[str, Any] = {
        "status": "running",
        "verification_mode": "complete_aggregate_no_per_record_payload",
        "samples": [],
    }
    if summary_path.is_file():
        existing = _read_json_object(summary_path, "existing verification summary")
        existing_samples = existing.get("samples")
        if isinstance(existing_samples, list):
            for value in existing_samples:
                if not isinstance(value, Mapping):
                    raise ValueError("existing verification sample entries must be objects")
                name = value.get("sample")
                if name not in all_names or name in requested_names:
                    continue
                sample = sample_manifest.get(name)
                if not isinstance(sample, Mapping):
                    raise ValueError(f"sample manifest entry {name!r} must be an object")
                if (
                    value.get("facts") != sample.get("canonical_fact_count")
                    or value.get("qa") != sample.get("qa_count")
                ):
                    raise ValueError(
                        f"existing {name} verification counts do not match its manifest"
                    )
                report_path = Path(str(value.get("report_path", "")))
                if not report_path.is_file():
                    raise ValueError(f"existing {name} verification report is missing")
                report = _read_json_object(report_path, f"{name} verification report")
                if (
                    report.get("status") != "verified"
                    or report.get("qa_count") != sample.get("qa_count")
                    or report.get("record_details_included") is not False
                ):
                    raise ValueError(
                        f"existing {name} verification report is incomplete or invalid"
                    )
                record = dict(value)
                record["facts_sha256"] = sample.get("facts_sha256")
                record["qa_sha256"] = sample.get("qa_sha256")
                summary["samples"].append(record)

    for name in requested_names:
        sample = sample_manifest.get(name)
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample manifest entry {name!r} must be an object")
        sample_dir = source_dir / name
        facts_path = sample_dir / "facts.json"
        qa_path = sample_dir / "qa_records.json"
        facts_sha256 = _file_sha256(facts_path)
        qa_sha256 = _file_sha256(qa_path)
        if facts_sha256 != sample.get("facts_sha256"):
            raise ValueError(f"{name} facts checksum does not match its manifest")
        if qa_sha256 != sample.get("qa_sha256"):
            raise ValueError(f"{name} QA checksum does not match its manifest")
        facts = _read_json_array(facts_path, f"{name} facts")
        if len(facts) != sample.get("canonical_fact_count"):
            raise ValueError(f"{name} fact count does not match its manifest")

        report_result = run_benchmark_verification(
            facts=facts,
            qa_records=_iter_json_array(qa_path, f"{name} QA records"),
            output_dir=destination / name,
            retrieval_settings={
                "strategy": "relation_constrained_bfs",
                "max_hops": 2,
                "max_nodes": 128,
                "max_edges": 256,
            },
            include_records=False,
        )
        report = _read_json_object(
            Path(report_result["report_path"]),
            f"{name} benchmark verification",
        )
        if report["qa_count"] != sample.get("qa_count"):
            raise ValueError(f"{name} QA count does not match its manifest")
        record = {
            "sample": name,
            "facts": len(facts),
            "qa": report["qa_count"],
            "facts_sha256": facts_sha256,
            "qa_sha256": qa_sha256,
            "status": report["status"],
            "answer_set_coverage": report["answer_set_coverage"],
            "temporal_boundary_coverage": report["temporal_boundary_coverage"],
            "candidate_coverage": report["candidate_coverage"],
            "supporting_evidence_coverage": report["supporting_evidence_coverage"],
            "graph_path_sanity": report["graph_path_sanity"],
            "report_path": report_result["report_path"],
        }
        summary["samples"].append(record)
        _atomic_write(summary_path, summary)

    summary["samples"].sort(key=lambda record: all_names.index(record["sample"]))
    summary["status"] = (
        "complete"
        if len(summary["samples"]) == len(all_names)
        and all(record["status"] == "verified" for record in summary["samples"])
        else "partial"
    )
    _atomic_write(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify prepared 5K/10K/20K fact samples."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "outputs/experiments/task62_fact_scale_samples/scale_samples_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/task63_benchmark_verification"),
    )
    parser.add_argument("--samples", nargs="+")
    args = parser.parse_args(argv)
    summary = verify_scale_samples(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        sample_names=args.samples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


def _read_json_array(path: Path, field: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} at {path} must be a non-empty JSON array")
    if any(not isinstance(record, dict) for record in value):
        raise ValueError(f"{field} at {path} must contain only JSON objects")
    return value


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} at {path} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_json_array(path: Path, field: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8") as file:
        buffer = file.read(chunk_size)
        position = 0
        at_end = False

        def refill() -> None:
            nonlocal buffer, position, at_end
            buffer = buffer[position:]
            position = 0
            chunk = file.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                at_end = True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or at_end:
                    return
                refill()

        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"{field} at {path} must be a JSON array")
        position += 1
        first = True
        while True:
            skip_whitespace()
            if position >= len(buffer):
                raise ValueError(f"{field} at {path} has no closing array bracket")
            if buffer[position] == "]":
                position += 1
                break
            if not first:
                if buffer[position] != ",":
                    raise ValueError(f"{field} at {path} must be a JSON array")
                position += 1
                skip_whitespace()
                if position >= len(buffer) or buffer[position] == "]":
                    raise ValueError(f"{field} at {path} has a trailing comma")
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if at_end:
                        raise ValueError(
                            f"{field} at {path} contains invalid JSON"
                        ) from exc
                    refill()
            if not isinstance(record, dict):
                raise ValueError(f"{field} at {path} must contain JSON objects")
            yield record
            position = end
            first = False

        while True:
            skip_whitespace()
            if position < len(buffer):
                raise ValueError(f"{field} at {path} has trailing content")
            if at_end:
                return
            refill()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
