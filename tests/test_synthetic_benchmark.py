import hashlib
import json

import pytest

from src.evaluation.synthetic_benchmark import (
    design_synthetic_benchmark,
    generate_harder_synthetic_benchmark,
    prepare_fact_scale_samples,
    run_synthetic_benchmark,
    run_fact_scale_sample_preparation,
)


def test_harder_synthetic_benchmark_is_deterministic_and_multi_hop():
    first = generate_harder_synthetic_benchmark(
        seed=61,
        seed_count=2,
        branches=3,
        max_hops=4,
    )
    second = generate_harder_synthetic_benchmark(
        seed=61,
        seed_count=2,
        branches=3,
        max_hops=4,
    )

    assert first == second
    assert first["facts"]
    assert any(
        len(record["relation_sequence"]) >= 3
        for record in first["qa_records"]
    )
    assert any(
        len(record["accepted_answer_ids"]) >= 3
        for record in first["qa_records"]
    )


def test_synthetic_benchmark_defines_slices_and_matched_limits(tmp_path):
    report = design_synthetic_benchmark(
        seed=61,
        seed_count=2,
        branches=3,
        max_hops=4,
    )

    assert report["status"] == "designed"
    assert report["difficulty_slice_ids"]
    assert report["difficulty_slices"]["three_or_more_hops"]["record_count"] > 0
    assert report["difficulty_slices"]["dense_candidates"]["record_count"] > 0
    conditions = report["matched_comparison_conditions"]
    assert [item["condition_id"] for item in conditions] == [
        "matched_unlimited",
        "matched_node_limited",
        "matched_edge_limited",
    ]
    assert conditions[1]["settings"]["max_nodes"] < conditions[0]["settings"]["max_nodes"]
    assert conditions[2]["settings"]["max_edges"] < conditions[0]["settings"]["max_edges"]

    result = run_synthetic_benchmark(output_dir=tmp_path, seed=61)
    saved = json.loads(
        (tmp_path / "synthetic_benchmark.json").read_text(encoding="utf-8")
    )
    assert saved["benchmark"] == "hard_synthetic"
    assert result["qa_count"] == saved["qa_count"]


@pytest.mark.parametrize("value", [-1, True])
def test_harder_synthetic_benchmark_rejects_invalid_seed(value):
    with pytest.raises(ValueError, match="seed"):
        generate_harder_synthetic_benchmark(seed=value)


def test_fact_scale_samples_are_deterministic_and_exported(tmp_path):
    raw_facts = [
        {
            "subject_id": f"s{index}",
            "relation_id": "r1",
            "object_id": f"o{index}",
            "timestamp": f"2014-01-{(index % 28) + 1:02d}",
            "source_id": f"source:{index}",
        }
        for index in range(30)
    ]

    first = prepare_fact_scale_samples(
        raw_facts,
        sizes=(10, 20),
        fact_sampling_seed=17,
        qa_seed=61,
        max_hops=2,
        dataset={"id": "ICEWS14", "source": "test_fixture"},
    )
    second = prepare_fact_scale_samples(
        raw_facts,
        sizes=(10, 20),
        fact_sampling_seed=17,
        qa_seed=61,
        max_hops=2,
        dataset={"id": "ICEWS14", "source": "test_fixture"},
    )

    assert first == second
    assert first["sample_names"] == ["10", "20"]
    assert first["resource_limits"]["max_resident_memory_bytes"] == 3221225472
    assert first["samples"]["10"]["facts"]
    assert first["samples"]["10"]["qa_records"]

    result = run_fact_scale_sample_preparation(
        raw_facts=raw_facts,
        output_dir=tmp_path,
        sizes=(10, 20),
        fact_sampling_seed=17,
        qa_seed=61,
        max_hops=2,
        dataset={"id": "ICEWS14", "source": "test_fixture"},
    )
    assert (tmp_path / "scale_samples_manifest.json").is_file()
    assert (tmp_path / "10" / "facts.json").is_file()
    assert (tmp_path / "20" / "qa_records.json").is_file()
    assert result["sample_count"] == 2
    manifest = json.loads(
        (tmp_path / "scale_samples_manifest.json").read_text(encoding="utf-8")
    )
    for name in manifest["sample_names"]:
        sample_dir = tmp_path / name
        sample = manifest["samples"][name]
        assert hashlib.sha256(
            (sample_dir / "facts.json").read_bytes()
        ).hexdigest() == sample["facts_sha256"]
        assert hashlib.sha256(
            (sample_dir / "qa_records.json").read_bytes()
        ).hexdigest() == sample["qa_sha256"]
