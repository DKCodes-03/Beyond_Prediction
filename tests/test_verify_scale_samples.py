import pytest

from src.evaluation.verify_scale_samples import _iter_json_array


def test_streaming_json_array_reader_handles_chunk_boundaries(tmp_path):
    source = tmp_path / "records.json"
    source.write_text(
        '[{"payload":"' + ("x" * (1024 * 1024 + 17)) + '"},{"payload":"last"}]',
        encoding="utf-8",
    )

    records = list(_iter_json_array(source, "records"))

    assert len(records) == 2
    assert records[0]["payload"].endswith("x")
    assert records[1] == {"payload": "last"}


@pytest.mark.parametrize(
    "contents",
    ["{}", '[{"value":1}', '[{"value":1},]'],
)
def test_streaming_json_array_reader_rejects_malformed_arrays(tmp_path, contents):
    source = tmp_path / "invalid.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        list(_iter_json_array(source, "records"))
