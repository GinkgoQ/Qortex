"""Owner-aware, recoverable controls for model artifacts recorded by Qortex."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models.cache import CacheEntry, ModelCache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torchvision_checkpoint_root() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    return (Path(torch.hub.get_dir()) / "checkpoints").resolve()


def _allowed_roots(cache: ModelCache, entry: CacheEntry) -> list[tuple[str, Path]]:
    roots = [("qortex", cache.cache_dir.resolve())]
    if entry.provider == "torchvision":
        root = _torchvision_checkpoint_root()
        if root is not None:
            roots.append(("torchvision", root))
    return roots


def _ownership(cache: ModelCache, entry: CacheEntry) -> tuple[str | None, Path | None]:
    target = Path(entry.local_path).expanduser().resolve()
    for owner, root in _allowed_roots(cache, entry):
        if target != root and root in target.parents:
            return owner, root
    return None, None


def _shared_entries(cache: ModelCache, entry: CacheEntry) -> list[str]:
    target = Path(entry.local_path).expanduser().resolve()
    return [
        candidate.model_id for candidate in cache.list_cached()
        if candidate.model_id != entry.model_id
        and Path(candidate.local_path).expanduser().resolve() == target
    ]


def model_cache_inventory(*, cache: ModelCache | None = None) -> dict[str, Any]:
    provenance = cache or ModelCache()
    entries = []
    for entry in provenance.list_cached():
        target = Path(entry.local_path).expanduser().resolve()
        owner, owner_root = _ownership(provenance, entry)
        shared = _shared_entries(provenance, entry)
        exists = target.exists()
        entries.append({
            "model_id": entry.model_id,
            "provider": entry.provider,
            "path": str(target),
            "exists": exists,
            "target_type": "directory" if target.is_dir() else "file" if target.is_file() else "missing",
            "recorded_size_bytes": entry.size_bytes,
            "sha256": entry.sha256,
            "downloaded_at": entry.downloaded_at,
            "source_url": entry.source_url,
            "storage_owner": owner or "unmanaged_external",
            "owner_root": str(owner_root) if owner_root is not None else None,
            "shared_with": shared,
            "removable": bool(exists and not target.is_symlink() and owner and not shared and entry.sha256),
            "removal_mode": "move_to_qortex_trash",
            "integrity_check": "The recorded SHA-256 must match the file or a file inside the recorded bundle before removal.",
        })
    return {
        "entries": entries,
        "trash_root": str((provenance.cache_dir / "trash").resolve()),
        "policy": (
            "Only exclusive, non-symlink artifacts inside a known owner root are removable. "
            "Removal requires the recorded SHA-256 and moves the target to Qortex trash with a receipt."
        ),
    }


def _verify_recorded_hash(target: Path, expected: str) -> str:
    if target.is_file():
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(f"Recorded artifact hash mismatch: expected {expected}, got {actual}")
        return str(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Recorded model artifact is missing: {target}")
    inspected = 0
    for path in sorted(target.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        inspected += 1
        if inspected > 10_000:
            raise ValueError("Model bundle integrity scan exceeded 10,000 files")
        if _sha256(path) == expected:
            return str(path)
    raise ValueError(f"No regular file in the recorded model bundle matches SHA-256 {expected}")


def move_model_artifact_to_trash(
    model_id: str,
    *,
    confirmation_sha256: str,
    cache: ModelCache | None = None,
) -> dict[str, Any]:
    provenance = cache or ModelCache()
    entry = provenance.lookup(model_id)
    if entry is None:
        raise KeyError(f"No cached model artifact {model_id!r}")
    inventory = model_cache_inventory(cache=provenance)
    state = next(item for item in inventory["entries"] if item["model_id"] == model_id)
    if not state["removable"]:
        raise ValueError(
            f"Model artifact is not safely removable: owner={state['storage_owner']}, "
            f"exists={state['exists']}, shared_with={state['shared_with']}"
        )
    if not confirmation_sha256 or confirmation_sha256 != entry.sha256:
        raise ValueError("confirmation_sha256 must exactly match the recorded model artifact SHA-256")

    target = Path(state["path"])
    matched_path = _verify_recorded_hash(target, confirmation_sha256)
    trash_root = provenance.cache_dir / "trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    trash_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}-{target.name}"
    trashed = trash_root / trash_name
    shutil.move(str(target), str(trashed))
    receipt = {
        "model_id": model_id,
        "provider": entry.provider,
        "moved_at": datetime.now(timezone.utc).isoformat(),
        "original_path": str(target),
        "trash_path": str(trashed.resolve()),
        "verified_file": matched_path,
        "verified_sha256": confirmation_sha256,
        "recorded_size_bytes": entry.size_bytes,
        "cache_entry": asdict(entry),
        "recoverable": True,
        "recovery": "Move trash_path back to original_path, then re-record the original cache entry.",
    }
    receipt_path = trash_root / f"{trash_name}.receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        temporary.replace(receipt_path)
        provenance.remove(model_id)
    except Exception:
        temporary.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        shutil.move(str(trashed), str(target))
        raise
    return {**receipt, "receipt_path": str(receipt_path.resolve())}


__all__ = ["model_cache_inventory", "move_model_artifact_to_trash"]
