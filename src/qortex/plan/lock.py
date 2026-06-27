"""Download lockfile — tracks which files have been fetched for a snapshot.

The lockfile is written atomically after each successful file download and
read on resume to skip already-complete files.

Format: YAML (human-readable, diffable).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ruamel.yaml import YAML

from qortex._version import __version__
from qortex.core.entities import DownloadPlan, FileRecord

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.allow_unicode = True

FileStatus = Literal["present", "missing", "failed", "pending"]


class LockFile:
    """Read/write a YAML lockfile for one dataset snapshot."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict = {}

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_plan(cls, plan: DownloadPlan) -> "LockFile":
        """Initialise a new lock from a DownloadPlan (all files → pending)."""
        lf = cls(plan.target_dir / ".qortex" / "download.lock.yaml")
        lf._data = {
            "qortex_version": __version__,
            "dataset_id": plan.dataset_id,
            "snapshot": plan.snapshot,
            "doi": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": plan.selection.model_dump(),
            "files": {
                f.path: {
                    "size": f.size,
                    "checksum": f.checksum,
                    "status": "pending",
                    "downloaded_at": None,
                }
                for f in plan.files
            },
        }
        return lf

    @classmethod
    def load(cls, path: Path) -> "LockFile":
        """Load an existing lockfile."""
        lf = cls(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                lf._data = _yaml.load(f) or {}
        return lf

    # ── Queries ───────────────────────────────────────────────────────────

    def is_present(self, path: str) -> bool:
        return self._data.get("files", {}).get(path, {}).get("status") == "present"

    def is_failed(self, path: str) -> bool:
        return self._data.get("files", {}).get(path, {}).get("status") == "failed"

    def pending_files(self, all_files: list[FileRecord]) -> list[FileRecord]:
        """Return files that are not yet marked present."""
        return [f for f in all_files if not self.is_present(f.path)]

    # ── Mutations ─────────────────────────────────────────────────────────

    def mark_present(self, path: str, checksum: str | None = None) -> None:
        if "files" not in self._data:
            self._data["files"] = {}
        entry = self._data["files"].setdefault(path, {})
        entry["status"] = "present"
        entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        if checksum:
            entry["checksum"] = checksum
        self._save()

    def mark_failed(self, path: str, error: str) -> None:
        if "files" not in self._data:
            self._data["files"] = {}
        entry = self._data["files"].setdefault(path, {})
        entry["status"] = "failed"
        entry["error"] = error
        self._save()

    def set_doi(self, doi: str) -> None:
        self._data["doi"] = doi
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Include PID and a short random token so parallel Qortex processes
        # operating on the same target directory never write to the same temp
        # file.  On POSIX, rename(2) is atomic; on Windows it is best-effort
        # but still isolated per-process.
        token = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        tmp = self._path.with_name(f"{self._path.name}.{token}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                _yaml.dump(self._data, f)
            tmp.replace(self._path)  # atomic rename on POSIX
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def save(self) -> None:
        self._save()

    @property
    def path(self) -> Path:
        return self._path

    # ── Summary ───────────────────────────────────────────────────────────

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"present": 0, "missing": 0, "failed": 0, "pending": 0}
        for entry in self._data.get("files", {}).values():
            status = entry.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        return counts
