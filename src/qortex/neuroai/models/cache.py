"""Model weight cache / provenance layer.

This is a manifest on top of each backend's own download cache (HF hub
cache, MONAI bundle directory, torch hub cache) — NOT a downloader. Qortex
records what it knows was downloaded and its checksum; it never fetches
weights itself. See docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 15.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    model_id: str
    provider: str
    local_path: str
    size_bytes: int
    sha256: str | None
    downloaded_at: str
    source_url: str | None = None


def _default_cache_dir() -> Path:
    override = os.environ.get("QORTEX_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".qortex" / "model_cache"


class ModelCache:
    """JSON-manifest-backed provenance record of downloaded model weights."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.manifest_path = self.cache_dir / "manifest.json"

    def _load(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {e["model_id"]: e for e in data.get("entries", [])}

    def _save(self, entries: dict[str, dict]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": self.SCHEMA_VERSION, "entries": list(entries.values())}
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_cached(self, model_id: str) -> bool:
        return model_id in self._load()

    def lookup(self, model_id: str) -> CacheEntry | None:
        raw = self._load().get(model_id)
        return CacheEntry(**raw) if raw else None

    def record(self, entry: CacheEntry) -> None:
        entries = self._load()
        entries[entry.model_id] = asdict(entry)
        self._save(entries)

    def remove(self, model_id: str) -> None:
        entries = self._load()
        entries.pop(model_id, None)
        self._save(entries)

    def list_cached(self) -> list[CacheEntry]:
        return [CacheEntry(**raw) for raw in self._load().values()]

    def disk_usage(self) -> int:
        return sum(e.size_bytes for e in self.list_cached())


__all__ = ["CacheEntry", "ModelCache"]
