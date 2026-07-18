from __future__ import annotations

from pathlib import Path

import nibabel as nib

from qortex.neuroclassic.public_roi_connectivity import (
    load_public_roi_connectivity_run,
    run_public_roi_connectivity,
)


def test_public_mni_roi_connectivity_persists_real_artifacts(tmp_path: Path) -> None:
    result = run_public_roi_connectivity(
        max_frames=40,
        fd_threshold_mm=0.5,
        connectivity_threshold=0.3,
        run_root=tmp_path / "runs",
        data_root=Path.home() / ".cache" / "qortex" / "public" / "roi-connectivity",
    )

    assert result["dataset"]["spatial_reference"] == "MNI152NLin2009cAsym"
    assert result["atlas"]["n_regions"] == 100
    assert result["connectivity"]["matrix_shape"] == [100, 100]
    assert result["scrubbing"]["retained_count"] >= 20
    assert len(result["roi_statistics"]) == 100
    assert all(row["voxel_count"] > 0 for row in result["roi_statistics"])
    assert result["artifact_inventory"]["montage"]["size_bytes"] > 0

    run_dir = tmp_path / "runs" / result["run_id"]
    mean_img = nib.load(run_dir / result["artifacts"]["mean_volume"])
    atlas_img = nib.load(run_dir / result["artifacts"]["atlas_labels"])
    assert mean_img.shape == atlas_img.shape == (50, 59, 50)
    assert load_public_roi_connectivity_run(result["run_id"], run_root=tmp_path / "runs")["status"] == "completed"
