import json

import pytest

from src.ingestion.loader import load_fact_records
from src.ingestion.text_loader import (
    evaluate_extraction_quality,
    extract_temporal_facts_from_text,
    load_txt_facts,
)


def test_txt_loader_supports_delimited_arrow_and_json_records(tmp_path):
    path = tmp_path / "facts.txt"
    path.write_text(
        "subject|relation|object|timestamp\n"
        "Alpha|met|Beta|2014-01-01\n"
        "Gamma -[visited]-> Delta (2014-01-02)\n"
        + json.dumps(
            {
                "subject_id": "Epsilon",
                "predicate": "met",
                "object_id": "Zeta",
                "date": "2014-01-03",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    facts = load_txt_facts(path)

    assert [(fact["subject_id"], fact["timestamp"]) for fact in facts] == [
        ("Alpha", "2014-01-01"),
        ("Gamma", "2014-01-02"),
        ("Epsilon", "2014-01-03"),
    ]


def test_txt_extraction_deduplicates_and_rejects_invalid_dates_tolerantly():
    text = (
        "Alpha|met|Beta|2014-01-01\n"
        "Alpha|met|Beta|2014-01-01\n"
        "Alpha|met|Beta|2014-02-30\n"
        "not a valid record\n"
    )

    facts = extract_temporal_facts_from_text(text)
    report = evaluate_extraction_quality(
        facts,
        [
            {
                "subject": "Alpha",
                "relation": "met",
                "object": "Beta",
                "timestamp": "2014-01-01",
            }
        ],
    )

    assert len(facts) == 1
    assert report["extracted_count"] == 1
    assert report["unique_extracted_count"] == 1
    assert report["duplicate_count"] == 0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0


def test_txt_loader_strict_mode_rejects_malformed_records(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("Alpha met Beta at 2014-01-01\nbad record\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_txt_facts(path)


def test_format_dispatch_loads_jsonl_and_json(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        '{"subject":"Alpha","relation":"met","object":"Beta","timestamp":"2014-01-01"}\n',
        encoding="utf-8",
    )
    json_path = tmp_path / "records.json"
    json_path.write_text(
        json.dumps(
            {"records": [{"subject": "Alpha", "relation": "met", "object": "Beta"}]}
        ),
        encoding="utf-8",
    )

    assert load_fact_records(jsonl_path)[0]["subject"] == "Alpha"
    assert load_fact_records(json_path)[0]["subject"] == "Alpha"
