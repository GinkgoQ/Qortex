from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qortex.console.fmri_qc_runs import load_fmri_qc_run, run_persistent_fmri_qc


def test_persistent_fmri_qc_writes_mean_frames_scrub_and_hashes(tmp_path: Path) -> None:
    grid = np.indices((6, 6, 4)).sum(axis=0).astype(np.float32)
    data = np.stack([grid + frame for frame in range(6)], axis=3)
    source = tmp_path / "source.nii.gz"
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((2.0, 2.0, 2.5, 1.5))
    nib.save(image, source)

    result = run_persistent_fmri_qc(
        dataset_id="dataset-real-path",
        snapshot="1.0.0",
        source_path="sub-01/func/sub-01_task-test_bold.nii.gz",
        local_file=source,
        max_frames=6,
        dvars_threshold=2.0,
        run_root=tmp_path / "runs",
    )

    run_dir = tmp_path / "runs" / result["run_id"]
    mean = np.asarray(nib.load(run_dir / "mean-bold.nii.gz").dataobj)
    assert np.allclose(mean, np.mean(data, axis=3))
    assert result["source"]["sha256"]
    assert result["mean_volume"]["source_volumes"] == 6
    assert set(result["artifact_inventory"]) == {"mean_volume", "framewise_table", "scrub_plan"}
    assert all(item["sha256"] for item in result["artifact_inventory"].values())
    scrub = json.loads((run_dir / "scrub-plan.json").read_text(encoding="utf-8"))
    assert scrub["immutable_source"] is True
    assert sorted(scrub["flagged_volumes"] + scrub["retained_volumes"]) == list(range(6))
    with (run_dir / "framewise-qc.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["volume"] == "0"
    assert rows[-1]["time_seconds"] == "7.5"
    assert load_fmri_qc_run(result["run_id"], run_root=tmp_path / "runs")["run_id"] == result["run_id"]
