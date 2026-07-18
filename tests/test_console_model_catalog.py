from __future__ import annotations

from pathlib import Path

from qortex.console.atlas_models import (
    compatibility_catalog,
    list_models,
    runtime_summary,
)
from qortex.neuroai.models.cache import CacheEntry, ModelCache


def test_atlas_catalog_is_the_registered_public_zoo(tmp_path: Path):
    rows = list_models(cache=ModelCache(tmp_path))

    ids = {row["id"] for row in rows}
    assert "monai.brats_mri_segmentation" in ids
    assert "external.totalsegmentator" in ids
    assert "custom/self-supervised-meg" not in ids
    assert all(row["source_url"].startswith("https://") for row in rows)
    assert all("license" in row and "evidence_status" in row for row in rows)
    brats = next(row for row in rows if row["id"] == "monai.brats_mri_segmentation")
    assert brats["license"]["name"] == "Apache-2.0"
    assert brats["license"]["evidence_status"] == "confirmed"
    assert brats["compatibility_available"] is True


def test_compatibility_catalog_is_derived_from_zoo_contracts():
    catalog = compatibility_catalog()

    assert "monai.brats_mri_segmentation" in catalog
    assert "braindecode.EEGNet" in catalog
    assert "braindecode/eegnet" not in catalog
    assert all(profile.input_contract is not None for profile in catalog.values())
    assert all(profile.output_contract is not None for profile in catalog.values())


def test_atlas_catalog_reports_only_persisted_cache_evidence(tmp_path: Path):
    cache = ModelCache(tmp_path)
    cache.record(CacheEntry(
        model_id="monai.brats_mri_segmentation",
        provider="monai",
        local_path=str(tmp_path / "bundle"),
        size_bytes=4096,
        sha256="abc123",
        downloaded_at="2026-07-18T00:00:00Z",
        source_url="https://huggingface.co/MONAI/brats_mri_segmentation",
    ))

    rows = {row["id"]: row for row in list_models(cache=cache)}

    assert rows["monai.brats_mri_segmentation"]["cached"] is True
    assert rows["monai.brats_mri_segmentation"]["cache"]["sha256"] == "abc123"
    assert rows["external.totalsegmentator"]["cached"] is False


def test_runtime_summary_uses_real_cache_manifest(tmp_path: Path):
    cache = ModelCache(tmp_path)
    summary = runtime_summary(cache=cache)

    assert summary["cache"]["entries"] == 0
    assert summary["cache"]["size_bytes"] == 0
    assert summary["offline_available_models"] == []
    assert set(summary["backends"]) >= {"torch", "monai", "braindecode"}
    assert all("available" in result and "module" in result for result in summary["backends"].values())
