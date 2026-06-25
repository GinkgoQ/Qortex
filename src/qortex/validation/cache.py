"""Validation result cache with local dataset fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import ValidationReport


class ValidationCache:
    """Persistent cache for validation reports.

    The key is based on validator options plus a metadata fingerprint of the
    local dataset tree. This avoids content hashing large neurodata files while
    still invalidating cache entries when files are added, removed, resized, or
    rewritten.
    """

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()
        self.root = self._cfg.cache_dir / "validation"
        self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        dataset_path: str | Path,
        *,
        executable: str,
        config_path: str | Path | None,
        ignore_warnings: bool,
        ignore_nifti_headers: bool,
    ) -> str:
        root = Path(dataset_path).expanduser().resolve()
        payload = {
            "dataset_path": str(root),
            "fingerprint": fingerprint_dataset(root),
            "executable": executable,
            "config_path": str(Path(config_path).expanduser().resolve()) if config_path else None,
            "config_fingerprint": fingerprint_file(Path(config_path).expanduser().resolve()) if config_path else None,
            "ignore_warnings": ignore_warnings,
            "ignore_nifti_headers": ignore_nifti_headers,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> ValidationReport | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ValidationReport(**data)
        except Exception:
            return None

    def put(self, key: str, report: ValidationReport) -> Path:
        path = self._path_for(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"


def fingerprint_dataset(root: Path) -> dict[str, Any]:
    files: list[tuple[str, int, int]] = []
    dirs = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel == ".qortex" or rel.startswith(".qortex/"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_dir():
            dirs += 1
            continue
        files.append((rel, int(stat.st_size), int(stat.st_mtime_ns)))
    digest = hashlib.sha256()
    for rel, size, mtime_ns in files:
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return {
        "digest": digest.hexdigest(),
        "n_files": len(files),
        "n_dirs": dirs,
        "total_bytes": sum(size for _rel, size, _mtime in files),
    }


def fingerprint_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(str(int(stat.st_size)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return {
        "digest": digest.hexdigest(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
