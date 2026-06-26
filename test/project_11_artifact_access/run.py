"""project_11_artifact_access

Verifies that Artifact.open() can read an artifact_manifest.json and expose
the correct summary, format, and split information.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, real_manifest,
    require, require_equal, passed,
)

from qortex.artifact import Artifact


def _write_fake_artifact(path: Path, dataset_id: str, snapshot: str) -> None:
    """Write a minimal artifact_manifest.json for testing Artifact.open()."""
    manifest = {
        "artifact_id": "test0000deadbeef",
        "dataset_id": dataset_id,
        "snapshot": snapshot,
        "doi": None,
        "output_format": "parquet",
        "output_path": str(path),
        "n_samples": 120,
        "n_subjects": 6,
        "splits": {"train": 80, "val": 20, "test": 20},
        "source_files": [f"sub-0{i}/eeg/sub-0{i}_task-rest_eeg.set" for i in range(1, 7)],
        "window_config": {"duration_s": 2.0, "stride_s": 1.0},
        "split_config": {"strategy": "subject"},
        "data_schema": {
            "sample": "qortex.core.entities.SampleRecord",
            "data": "array",
            "label": "int",
        },
        "n_failed_files": 0,
        "n_skipped_files": 0,
        "failed_files": [],
    }
    (path / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    banner("project_11: artifact open and summary")

    ds, manifest = real_manifest()

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = Path(tmp) / "test_artifact"
        artifact_path.mkdir()
        _write_fake_artifact(artifact_path, manifest.dataset_id, manifest.snapshot)

        # ── open via Artifact ─────────────────────────────────────────────────
        artifact = Artifact.open(artifact_path)
        summary = artifact.summary()

        print_kv("artifact summary", summary)

        require_equal(summary["dataset_id"], manifest.dataset_id, "dataset_id")
        require_equal(summary["format"], "parquet", "format")
        require_equal(summary["n_samples"], 120, "n_samples")
        require_equal(summary["n_subjects"], 6, "n_subjects")
        require(isinstance(summary["splits"], dict), "splits is not a dict")
        require(summary["splits"].get("train") == 80, "splits.train != 80")
        require(summary["splits"].get("val") == 20, "splits.val != 20")
        require(summary["splits"].get("test") == 20, "splits.test != 20")

        # ── manifest fields ───────────────────────────────────────────────────
        am = artifact.manifest
        require(am.artifact_id == "test0000deadbeef", "artifact_id mismatch")
        require(am.output_format == "parquet", "output_format mismatch")
        require(len(am.source_files) == 6, f"expected 6 source_files, got {len(am.source_files)}")

        print_kv("artifact_id", am.artifact_id)
        print_kv("source_files", am.source_files[:3])

        # ── missing manifest → raises FileNotFoundError ────────────────────────
        missing_path = Path(tmp) / "nonexistent"
        missing_path.mkdir()
        try:
            Artifact.open(missing_path)
            raise RuntimeError("Expected FileNotFoundError for missing artifact_manifest.json")
        except FileNotFoundError:
            pass

    passed("project_11_artifact_access")


if __name__ == "__main__":
    main()
