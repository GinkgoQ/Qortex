from __future__ import annotations

from pathlib import Path

import pytest

from qortex.neuroai.models.cache import CacheEntry, ModelCache


def _entry(model_id: str = "monai.brats_mri_segmentation") -> CacheEntry:
    return CacheEntry(
        model_id=model_id,
        provider="monai",
        local_path="/tmp/fake/bundle",
        size_bytes=1024,
        sha256="deadbeef",
        downloaded_at="2026-07-09T00:00:00Z",
        source_url="https://huggingface.co/MONAI/brats_mri_segmentation",
    )


def test_record_and_is_cached(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.is_cached("monai.brats_mri_segmentation") is False

    cache.record(_entry())

    assert cache.is_cached("monai.brats_mri_segmentation") is True


def test_lookup_returns_recorded_entry(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())

    found = cache.lookup("monai.brats_mri_segmentation")

    assert found is not None
    assert found.sha256 == "deadbeef"
    assert found.size_bytes == 1024


def test_lookup_missing_returns_none(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.lookup("nonexistent") is None


def test_manifest_persists_across_instances(tmp_path: Path):
    ModelCache(cache_dir=tmp_path).record(_entry())

    reopened = ModelCache(cache_dir=tmp_path)

    assert reopened.is_cached("monai.brats_mri_segmentation") is True
    assert (tmp_path / "manifest.json").exists()


def test_list_cached_and_disk_usage(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry("monai.brats_mri_segmentation"))
    cache.record(_entry("braindecode.EEGNet"))

    listed = cache.list_cached()

    assert {e.model_id for e in listed} == {"monai.brats_mri_segmentation", "braindecode.EEGNet"}
    assert cache.disk_usage() == 2048


def test_record_overwrites_existing_entry_for_same_id(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())
    updated = _entry()
    updated.size_bytes = 9999
    cache.record(updated)

    assert cache.disk_usage() == 9999
    assert len(cache.list_cached()) == 1


def test_remove_drops_entry_from_manifest(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())

    cache.remove("monai.brats_mri_segmentation")

    assert cache.is_cached("monai.brats_mri_segmentation") is False


def test_default_cache_dir_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QORTEX_CACHE_DIR", str(tmp_path))
    cache = ModelCache()

    cache.record(_entry())

    assert (tmp_path / "manifest.json").exists()
