"""Cryptographic content fingerprint for dataset files, configs, and fold partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_content_hash(path: str | Path, *, chunk_size: int = 65536) -> str:
    """Return lowercase hex SHA-256 of file bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash missing dataset file: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def try_compute_file_content_hash(path: str | Path | None) -> str | None:
    """Best-effort hash; returns None when path is missing or unreadable."""
    if path is None:
        return None
    try:
        return compute_file_content_hash(path)
    except (OSError, ValueError):
        return None


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Return lowercase hex SHA-256 fingerprint for trial configuration."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_fold_assignment_hash(assignments: Any) -> str:
    """Return lowercase hex SHA-256 fingerprint for cross-validation fold partition assignments."""
    if hasattr(assignments, "tolist"):
        items = assignments.tolist()
    else:
        items = list(assignments)
    raw = json.dumps([int(x) for x in items])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

