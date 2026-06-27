"""Thread-safe LRU byte-range cache for streamed neuroimaging data.

Stores (url, byte_range) → bytes in a bounded LRU structure backed by an
``OrderedDict``.  Each entry carries a timestamp for optional TTL-based
eviction.  The cache is process-local (in-memory); for cross-process reuse
see ``DiskCache`` below, which spills entries to a ``~/.qortex/stream_cache``
directory.

Both classes share the same ``get(key) / put(key, data)`` interface so callers
are agnostic to the backing store.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MAXSIZE = 256          # entries
_DEFAULT_TTL = 3600.0           # seconds (1 hour)
_DEFAULT_DISK_MAX_GB = 2.0


class MemoryCache:
    """In-process LRU cache with optional TTL.

    Parameters
    ----------
    maxsize:
        Maximum number of cached entries.  Oldest-used entry evicted on overflow.
    ttl:
        Seconds before an entry is considered stale and evicted.  0 disables TTL.
    """

    def __init__(self, maxsize: int = _DEFAULT_MAXSIZE, ttl: float = _DEFAULT_TTL) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            data, ts = entry
            if self._ttl > 0 and (time.monotonic() - ts) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (data, time.monotonic())
            while len(self._store) > self._maxsize:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("Cache evicted: %s", evicted_key[:60])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class DiskCache:
    """Persistent file-backed cache that survives process restarts.

    Each entry is stored as a flat binary file named by the SHA-256 of its key.
    Eviction is size-based: when the cache directory exceeds ``max_gb``, the
    oldest-modified entries are removed until the limit is met.

    Parameters
    ----------
    cache_dir:
        Directory for cached files.  Created if absent.
    max_gb:
        Maximum total size of cached files in gigabytes.
    ttl:
        Seconds before a cached file is considered stale.  0 disables TTL.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_gb: float = _DEFAULT_DISK_MAX_GB,
        ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._dir = (
            cache_dir or Path.home() / ".qortex" / "stream_cache"
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_gb * 1e9)
        self._ttl = ttl
        self._lock = threading.Lock()

    def _key_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self._dir / h[:2] / h

    def get(self, key: str) -> bytes | None:
        path = self._key_path(key)
        try:
            if not path.exists():
                return None
            if self._ttl > 0:
                age = time.time() - path.stat().st_mtime
                if age > self._ttl:
                    path.unlink(missing_ok=True)
                    return None
            # Touch file to update LRU ordering
            os.utime(path, None)
            return path.read_bytes()
        except OSError:
            return None

    def put(self, key: str, data: bytes) -> None:
        path = self._key_path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as exc:
            log.warning("DiskCache write failed for key %s: %s", key[:40], exc)
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """Evict oldest files if total size exceeds the limit."""
        try:
            entries = sorted(
                (p for p in self._dir.rglob("*") if p.is_file() and not p.suffix == ".tmp"),
                key=lambda p: p.stat().st_mtime,
            )
            total = sum(p.stat().st_size for p in entries)
            while total > self._max_bytes and entries:
                oldest = entries.pop(0)
                size = oldest.stat().st_size
                oldest.unlink(missing_ok=True)
                total -= size
        except OSError:
            pass

    def clear(self) -> None:
        import shutil
        with self._lock:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)


def make_cache(
    backend: str = "memory",
    cache_dir: Path | None = None,
    maxsize: int = _DEFAULT_MAXSIZE,
    max_gb: float = _DEFAULT_DISK_MAX_GB,
    ttl: float = _DEFAULT_TTL,
) -> MemoryCache | DiskCache:
    """Factory for stream caches.

    Parameters
    ----------
    backend:
        ``"memory"`` (default) or ``"disk"``.
    """
    if backend == "disk":
        return DiskCache(cache_dir=cache_dir, max_gb=max_gb, ttl=ttl)
    return MemoryCache(maxsize=maxsize, ttl=ttl)
