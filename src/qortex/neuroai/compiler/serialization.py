"""Deterministic JSON serialization for NeuroAI compiler artifacts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    """Convert Qortex/Pydantic objects into JSON-compatible primitives."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return to_plain(value.__dict__)
    return str(value)


def canonical_json(value: Any) -> str:
    """Return canonical JSON used for stable plan hashing."""

    return json.dumps(to_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def pretty_json(value: Any) -> str:
    """Return human-readable JSON with stable key ordering."""

    return json.dumps(to_plain(value), sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def sha256_json(value: Any) -> str:
    """Hash canonical JSON for deterministic plan identity."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "pretty_json", "sha256_json", "to_plain"]
