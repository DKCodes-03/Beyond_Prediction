"""Isolated benchmarks for optional project integrations.

These benchmarks keep optional integration evidence separate from the core
answer metrics and from each other. They must not require Neo4j or the optional
Qwen/QLoRA dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.graph.graph_builder import build_temporal_graph
from src.ingestion.loader import load_fact_records
from src.ingestion.text_loader import (
    evaluate_extraction_quality,
    extract_temporal_facts_from_text,
)


def benchmark_ingestion(
    files: Mapping[str, str | Path],
    *,
    extracted_text: str | None = None,
    references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Benchmark format dispatch and text extraction independently."""
    if not isinstance(files, Mapping):
        raise ValueError("files must be a mapping")

    formats: dict[str, Any] = {}
    for name, path_value in files.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every ingestion benchmark name must be non-empty")
        path = Path(path_value)
        started = time.perf_counter()
        records = load_fact_records(path)
        elapsed = time.perf_counter() - started
        formats[name] = {
            "path": str(path),
            "record_count": len(records),
            "elapsed_seconds": elapsed,
        }

    report: dict[str, Any] = {
        "benchmark": "optional_ingestion",
        "formats": formats,
    }
    if extracted_text is not None:
        started = time.perf_counter()
        extracted = extract_temporal_facts_from_text(
            extracted_text,
            strict=True,
        )
        elapsed = time.perf_counter() - started
        report["text_extraction"] = {
            "elapsed_seconds": elapsed,
            **evaluate_extraction_quality(extracted, references),
        }
    return report


def benchmark_neo4j(
    adapter: Any,
    facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Benchmark persistence/query behavior against an injected Neo4j adapter."""
    if not isinstance(facts, (list, tuple)) or not facts:
        raise ValueError("facts must be a non-empty list")
    if any(not isinstance(fact, Mapping) for fact in facts):
        raise ValueError("every fact must be a mapping")

    graph = build_temporal_graph(facts)
    started = time.perf_counter()
    persisted_count = adapter.persist_temporal_graph(graph)
    loaded_facts = adapter.load_facts()
    elapsed = time.perf_counter() - started
    if persisted_count != graph.fact_count:
        raise RuntimeError(
            "Neo4j adapter persisted a different fact count than the in-memory graph"
        )
    def fact_key(record: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(record["fact_id"]),
            str(record["subject_id"]),
            str(record["relation_id"]),
            str(record["object_id"]),
            str(record["timestamp"]),
        )

    loaded_keys = {fact_key(record) for record in loaded_facts}
    in_memory_keys = {fact_key(dict(fact)) for fact in graph.facts}
    return {
        "benchmark": "optional_neo4j",
        "persisted_count": persisted_count,
        "loaded_count": len(loaded_facts),
        "in_memory_count": graph.fact_count,
        "round_trip_match": loaded_keys == in_memory_keys,
        "elapsed_seconds": elapsed,
    }


def benchmark_qwen(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    answer_ids: Sequence[str],
) -> dict[str, Any]:
    """Benchmark bounded prompt construction without loading a model."""
    from src.llm.prompt_templates import build_answer_prompt

    started = time.perf_counter()
    prompt = build_answer_prompt(question, evidence, answer_ids=answer_ids)
    elapsed = time.perf_counter() - started
    return {
        "benchmark": "optional_qwen",
        "model_loaded": False,
        "inference_performed": False,
        "prompt_chars": len(prompt),
        "prompt_evidence_count": len(evidence),
        "optional_packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
        "elapsed_seconds": elapsed,
    }


def benchmark_icews18_transfer(
    source_facts: Sequence[Mapping[str, Any]] | None = None,
    target_facts: Sequence[Mapping[str, Any]] | None = None,
    *,
    icews14_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_qa_records: Sequence[Mapping[str, Any]] | None = None,
    entity_mapping: Mapping[str, str] | None = None,
    relation_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assess ICEWS18 compatibility without adding it to the core pipeline.

    This is an isolated feasibility assessment.  It performs no learned-model
    transfer and makes no result claim unless target facts and QA are supplied.
    """
    from src.evaluation.transfer_assessment import assess_icews18_transfer

    return assess_icews18_transfer(
        source_facts=source_facts,
        target_facts=target_facts,
        icews14_facts=icews14_facts,
        icews18_facts=icews18_facts,
        icews18_qa_records=icews18_qa_records,
        entity_mapping=entity_mapping,
        relation_mapping=relation_mapping,
    )


def assess_icews18_transfer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the isolated ICEWS18 assessment."""
    return benchmark_icews18_transfer(*args, **kwargs)


def benchmark_adopted_integrations(
    *,
    ingestion_files: Mapping[str, str | Path] | None = None,
    extracted_text: str | None = None,
    references: Sequence[Mapping[str, Any]] | None = None,
    neo4j_adapter: Any | None = None,
    facts: Sequence[Mapping[str, Any]] | None = None,
    question: str | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    answer_ids: Sequence[str] | None = None,
    icews14_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_facts: Sequence[Mapping[str, Any]] | None = None,
    icews18_qa_records: Sequence[Mapping[str, Any]] | None = None,
    entity_mapping: Mapping[str, str] | None = None,
    relation_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run selected optional benchmarks and return a combined report."""
    components: dict[str, Any] = {}
    if ingestion_files is not None or extracted_text is not None:
        components["ingestion"] = benchmark_ingestion(
            ingestion_files or {},
            extracted_text=extracted_text,
            references=references,
        )

    if neo4j_adapter is not None:
        if facts is None:
            raise ValueError("facts are required for the Neo4j benchmark")
        components["neo4j"] = benchmark_neo4j(neo4j_adapter, facts)

    if question is not None:
        components["qwen"] = benchmark_qwen(
            question,
            evidence or [],
            answer_ids or [],
        )

    icews18_assessment_requested = (
        icews14_facts is not None
        or icews18_facts is not None
        or icews18_qa_records is not None
        or entity_mapping is not None
        or relation_mapping is not None
    )
    if icews18_assessment_requested:
        components["icews18"] = benchmark_icews18_transfer(
            icews14_facts=icews14_facts,
            icews18_facts=icews18_facts,
            icews18_qa_records=icews18_qa_records,
            entity_mapping=entity_mapping,
            relation_mapping=relation_mapping,
        )

    if not components:
        raise ValueError("at least one integration benchmark must be selected")

    return {
        "schema_version": 1,
        "components": components,
    }


def run_integration_benchmark(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an integration benchmark report atomically."""
    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "integration_benchmark.json"
    temporary_path = destination / ".integration_benchmark.json.tmp"
    temporary_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return {
        "report_path": str(report_path),
        "component_count": len(report.get("components", {})),
    }


# Short aliases make this benchmark usable from scripts and focused tests.
benchmark_optional_integrations = benchmark_adopted_integrations
run_optional_integration_benchmark = run_integration_benchmark
