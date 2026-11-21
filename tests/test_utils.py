import json
import logging

import numpy as np
import torch

from src.utils.logging_utils import (
    create_run_metadata,
    setup_logger,
    write_run_metadata,
)
from src.utils.seed import set_seed


def test_set_seed_reproduces_python_numpy_and_torch_values():
    set_seed(17, deterministic=True)
    python_value = __import__("random").random()
    numpy_value = np.random.rand()
    torch_value = torch.rand(1).item()

    set_seed(17, deterministic=True)

    assert __import__("random").random() == python_value
    assert np.random.rand() == numpy_value
    assert torch.rand(1).item() == torch_value


def test_set_seed_rejects_invalid_values():
    for invalid in (-1, 1.5, True, "17"):
        try:
            set_seed(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("set_seed accepted an invalid seed")


def test_run_metadata_and_json_output(tmp_path):
    metadata = create_run_metadata(
        config={"seed": 17},
        seed=17,
        split="validation",
        model_variant="baseline",
        run_id="phase0-test",
        code_revision="test",
    )
    destination = write_run_metadata(tmp_path / "run.json", metadata)

    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8"))["run_id"] == "phase0-test"
    assert metadata["seed"] == 17


def test_setup_logger_writes_explicit_log(tmp_path):
    logger = setup_logger(
        name="phase0_test",
        log_dir=tmp_path,
        level=logging.INFO,
        console=False,
    )
    logger.info("retrieval status=%s", "empty")

    log_file = tmp_path / "phase0_test.log"
    assert "retrieval status=empty" in log_file.read_text(encoding="utf-8")
