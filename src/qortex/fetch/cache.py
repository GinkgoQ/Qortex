"""Content-addressed local file cache.

The cache maps file checksums to local paths.  When a file has already been
downloaded (possibly as part of a different snapshot), it is hard-linked or
symlinked rather than re-fetched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from qortex._internal.hashing import md5_file
from qortex.core.config import QortexConfig, get_config


class FileCache:
    """Manages a content-addressed cache under ``{cache_dir}/objects/``."""

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()
        self._root = self._cfg.cache_dir / "objects"
        self._root.mkdir(parents=True, exist_ok=True)

    def _object_path(self, checksum: str) -> Path:
        """Return the cache path for a given MD5 checksum (two-level sharding)."""
        return self._root / checksum[:2] / checksum[2:]

    def contains(self, checksum: str) -> bool:
        return self._object_path(checksum).exists()

    def put(self, source: Path, checksum: str) -> Path:
        """Register *source* in the cache and return the object path."""
        obj = self._object_path(checksum)
        obj.parent.mkdir(parents=True, exist_ok=True)
        if not obj.exists():
            shutil.copy2(source, obj)
        return obj

    def get(self, checksum: str, destination: Path, *, link: bool = True) -> bool:
        """Materialise a cached file at *destination*.

        Parameters
        ----------
        link:
            If True, attempt a hard link first (saves disk space).
            Falls back to copy if the hard link crosses filesystem boundaries.
        """
        obj = self._object_path(checksum)
        if not obj.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link:
            try:
                os.link(obj, destination)
                return True
            except OSError:
                pass
        shutil.copy2(obj, destination)
        return True

    def compute_and_put(self, source: Path) -> str:
        """Hash *source* and store it in the cache.  Returns the checksum."""
        checksum = md5_file(source)
        self.put(source, checksum)
        return checksum

    def prune(self, keep_checksums: set[str]) -> int:
        """Remove cached objects whose checksums are not in *keep_checksums*.

        Returns the number of objects removed.
        """
        removed = 0
        for obj in self._root.rglob("*"):
            if obj.is_file():
                checksum = obj.parent.name + obj.name
                if checksum not in keep_checksums:
                    obj.unlink()
                    removed += 1
        return removed
