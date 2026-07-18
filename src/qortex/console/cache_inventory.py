"""Read-only inventory of Qortex's persistent storage surfaces.

The console has several independent caches with different ownership and
eviction semantics.  This module reports the directories that the installed
runtime actually uses.  It deliberately does not infer a quota where the
owning component has none and does not follow symlinks while measuring disk
usage.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from qortex.core.config import QortexConfig, get_config
from qortex.neuroai.models.cache import ModelCache


@dataclass(frozen=True)
class CacheSurface:
    id: str
    label: str
    path: str
    exists: bool
    file_count: int
    size_bytes: int
    last_modified: str | None
    max_bytes: int | None
    ttl_seconds: int | None
    policy_evidence: str
    removable: bool = False


def _measure(path: Path) -> tuple[int, int, float | None]:
    """Return file count, apparent bytes, and newest mtime without following links."""
    if not path.exists():
        return 0, 0, None

    count = 0
    size = 0
    newest: float | None = None
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    count += 1
                    size += stat.st_size
                    newest = stat.st_mtime if newest is None else max(newest, stat.st_mtime)
        except (NotADirectoryError, OSError):
            try:
                stat = current.stat()
            except OSError:
                continue
            if current.is_file():
                count += 1
                size += stat.st_size
                newest = stat.st_mtime if newest is None else max(newest, stat.st_mtime)
    return count, size, newest


def _surface(
    cache_id: str,
    label: str,
    path: Path,
    *,
    max_bytes: int | None = None,
    ttl_seconds: int | None = None,
    policy_evidence: str,
) -> CacheSurface:
    count, size, newest = _measure(path)
    return CacheSurface(
        id=cache_id,
        label=label,
        path=str(path),
        exists=path.exists(),
        file_count=count,
        size_bytes=size,
        last_modified=(
            datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
            if newest is not None
            else None
        ),
        max_bytes=max_bytes,
        ttl_seconds=ttl_seconds,
        policy_evidence=policy_evidence,
    )


def cache_inventory(
    config: QortexConfig | None = None,
    *,
    home: Path | None = None,
    model_cache: ModelCache | None = None,
) -> dict[str, object]:
    """Measure every persistent storage surface known to the Qortex runtime."""
    cfg = config or get_config()
    user_home = home or Path.home()
    provenance = model_cache or ModelCache()
    qortex_home = user_home / ".qortex"
    core = cfg.cache_dir

    surfaces = [
        _surface(
            "catalog",
            "Dataset catalog",
            core / "catalog",
            policy_evidence="No configured quota or TTL; refreshed explicitly from OpenNeuro.",
        ),
        _surface(
            "downloaded_datasets",
            "Downloaded OpenNeuro data",
            core / "datasets",
            policy_evidence="Managed by download plans and the local registry; no global quota configured.",
        ),
        _surface(
            "shared_models",
            "Backend model assets",
            core / "models",
            policy_evidence="Backend-owned model files; no global quota or TTL configured.",
        ),
        _surface(
            "objects",
            "Content-addressed objects",
            core / "objects",
            policy_evidence="Pruned against registered checksums; no automatic size quota configured.",
        ),
        _surface(
            "validation",
            "Validation results",
            core / "validation",
            policy_evidence="Fingerprint-keyed validation results; no configured TTL.",
        ),
        _surface(
            "stream",
            "Remote byte-range cache",
            qortex_home / "stream_cache",
            max_bytes=2_000_000_000,
            ttl_seconds=3600,
            policy_evidence="DiskCache defaults: 2 GB LRU ceiling and one-hour TTL.",
        ),
        _surface(
            "model_provenance",
            "Recorded model artifacts",
            provenance.cache_dir,
            policy_evidence="Manifest-backed provenance; artifact lifetime is explicit, with no automatic TTL.",
        ),
        _surface(
            "public_validation_inputs",
            "Public validation inputs",
            qortex_home / "datasets",
            policy_evidence="Pinned public datasets used for reproducible validation; no automatic eviction.",
        ),
        _surface(
            "public_validation_runs",
            "Public validation runs",
            qortex_home / "runs",
            policy_evidence="Immutable run artifacts and provenance; no automatic eviction.",
        ),
    ]
    # A cache-root override can make the model-provenance root identical to,
    # or an ancestor of, core surfaces. Keep per-surface measurements useful
    # while making the aggregate a union rather than double-counting roots.
    aggregate: list[CacheSurface] = []
    for item in sorted(surfaces, key=lambda value: len(Path(value.path).parts)):
        item_path = Path(item.path)
        if any(
            item_path == Path(root.path)
            or item_path.is_relative_to(Path(root.path))
            for root in aggregate
        ):
            continue
        aggregate.append(item)
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "configured_cache_root": str(core),
        "surfaces": [asdict(item) for item in surfaces],
        "total_bytes": sum(item.size_bytes for item in aggregate),
        "total_file_count": sum(item.file_count for item in aggregate),
        "measurement": "apparent file size; symbolic links excluded; aggregate de-duplicates nested roots",
    }


__all__ = ["CacheSurface", "cache_inventory"]
