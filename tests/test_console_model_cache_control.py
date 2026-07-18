from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qortex.console.model_cache_control import model_cache_inventory, move_model_artifact_to_trash
from qortex.neuroai.models.cache import CacheEntry, ModelCache


def _record(cache: ModelCache, model_id: str, target: Path, content: bytes = b"real weights") -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    cache.record(CacheEntry(
        model_id=model_id,
        provider="monai",
        local_path=str(target.parent),
        size_bytes=len(content),
        sha256=digest,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        source_url="https://example.invalid/weights",
    ))
    return digest


def test_inventory_marks_exclusive_qortex_owned_bundle_removable(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "model-cache")
    digest = _record(cache, "model-a", cache.cache_dir / "bundles" / "model-a" / "model.pt")

    report = model_cache_inventory(cache=cache)

    assert report["entries"] == [{
        "model_id": "model-a",
        "provider": "monai",
        "path": str((cache.cache_dir / "bundles" / "model-a").resolve()),
        "exists": True,
        "target_type": "directory",
        "recorded_size_bytes": 12,
        "sha256": digest,
        "downloaded_at": cache.lookup("model-a").downloaded_at,
        "source_url": "https://example.invalid/weights",
        "storage_owner": "qortex",
        "owner_root": str(cache.cache_dir.resolve()),
        "shared_with": [],
        "removable": True,
        "removal_mode": "move_to_qortex_trash",
        "integrity_check": "The recorded SHA-256 must match the file or a file inside the recorded bundle before removal.",
    }]


def test_removal_moves_bundle_to_trash_and_writes_recovery_receipt(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "model-cache")
    bundle = cache.cache_dir / "bundles" / "model-a"
    digest = _record(cache, "model-a", bundle / "model.pt")

    receipt = move_model_artifact_to_trash("model-a", confirmation_sha256=digest, cache=cache)

    assert not bundle.exists()
    assert Path(receipt["trash_path"]).is_dir()
    assert Path(receipt["trash_path"], "model.pt").read_bytes() == b"real weights"
    assert cache.lookup("model-a") is None
    persisted = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))
    assert persisted["cache_entry"]["model_id"] == "model-a"
    assert persisted["verified_sha256"] == digest
    assert persisted["recoverable"] is True


def test_removal_rejects_wrong_confirmation_without_mutating(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "model-cache")
    bundle = cache.cache_dir / "bundles" / "model-a"
    _record(cache, "model-a", bundle / "model.pt")

    with pytest.raises(ValueError, match="confirmation_sha256"):
        move_model_artifact_to_trash("model-a", confirmation_sha256="wrong", cache=cache)

    assert bundle.is_dir()
    assert cache.lookup("model-a") is not None


def test_inventory_rejects_unmanaged_or_shared_targets(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path / "model-cache")
    external = tmp_path / "external" / "model.pt"
    digest = _record(cache, "external", external)
    cache.record(CacheEntry(
        model_id="external-copy",
        provider="monai",
        local_path=str(external.parent),
        size_bytes=external.stat().st_size,
        sha256=digest,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    ))

    report = model_cache_inventory(cache=cache)

    assert {entry["storage_owner"] for entry in report["entries"]} == {"unmanaged_external"}
    assert all(entry["removable"] is False for entry in report["entries"])
    assert {tuple(entry["shared_with"]) for entry in report["entries"]} == {("external-copy",), ("external",)}
