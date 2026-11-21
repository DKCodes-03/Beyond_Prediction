"""Explicit run logging and experiment-metadata helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def setup_logger(
    name: str = "beyond_prediction",
    log_dir: str | os.PathLike[str] | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a named logger without duplicating handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            directory / f"{name}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "beyond_prediction") -> logging.Logger:
    """Return a logger, configuring a console-only logger if it is new."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name=name)
    logger.propagate = False
    return logger


configure_logging = setup_logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log an explicit event and stable key/value fields."""
    details = " ".join(
        f"{key}={json.dumps(value, sort_keys=True, default=str)}"
        for key, value in sorted(fields.items())
    )
    message = f"{event} {details}".rstrip()
    logger.info(message)


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp with a timezone."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def detect_code_revision(repository: str | os.PathLike[str] | None = None) -> str:
    """Return the HEAD revision, or an explicit unknown marker outside Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def create_run_metadata(
    *,
    config: Mapping[str, Any],
    seed: int,
    split: str,
    model_variant: str,
    data_manifest: Mapping[str, Any] | None = None,
    code_revision: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    measurement_conditions: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    checkpoint_location: str | None = None,
) -> dict[str, Any]:
    """Build a run-metadata record matching the Phase 0 data contract."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not model_variant:
        raise ValueError("model_variant must be non-empty")

    manifest = dict(
        data_manifest
        or {
            "dataset_id": "unknown",
            "version": "unknown",
            "checksum_sha256": "0" * 64,
        }
    )
    required_manifest_fields = {"dataset_id", "version", "checksum_sha256"}
    missing = required_manifest_fields.difference(manifest)
    if missing:
        raise ValueError(
            "data_manifest is missing required fields: " + ", ".join(sorted(missing))
        )
    for field in ("dataset_id", "version"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(f"data_manifest.{field} must be non-empty")
    checksum = manifest["checksum_sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise ValueError("data_manifest.checksum_sha256 must contain 64 hexadecimal characters")

    metadata: dict[str, Any] = {
        "run_id": run_id or str(uuid.uuid4()),
        "created_at": utc_now(),
        "config": dict(config),
        "seed": seed,
        "code_revision": code_revision or detect_code_revision(),
        "data_manifest": manifest,
        "split": split,
        "model_variant": model_variant,
        "metrics": dict(metrics or {}),
        "measurement_conditions": dict(measurement_conditions or {}),
    }
    if checkpoint_location is not None:
        metadata["checkpoint_location"] = checkpoint_location
    return metadata


def write_run_metadata(
    path: str | os.PathLike[str],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write one run-metadata JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


save_run_metadata = write_run_metadata
