"""Artifact-backed ROI connectivity validation on public normalized fMRI."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import time
import uuid
import warnings
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from qortex.neuroclassic.connectivity import compute_graph_metrics, compute_pearson_connectivity


_ATLAS_SOURCE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
    "v0.14.3-Update_Yeo2011_Schaefer2018_labelname/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)
_DATASET_SOURCE = "https://osf.io/5hju4/files/"
_RAW_DATASET_SOURCE = "https://openneuro.org/datasets/ds000228/versions/1.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _model_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _environment() -> dict[str, Any]:
    import nibabel
    import nilearn
    import pandas
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nibabel.__version__,
            "nilearn": nilearn.__version__,
            "pandas": pandas.__version__,
        },
    }


def _roi_geometry(labels_img: Any, labels: list[str]) -> list[dict[str, Any]]:
    from scipy.ndimage import center_of_mass

    data = np.asarray(labels_img.dataobj)
    affine = labels_img.affine
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        mask = data == index
        voxel_count = int(mask.sum())
        if voxel_count:
            ijk = np.asarray(center_of_mass(mask), dtype=float)
            xyz = affine @ np.r_[ijk, 1.0]
            centroid = [float(value) for value in xyz[:3]]
        else:
            centroid = None
        rows.append({
            "index": index,
            "label": label,
            "voxel_count": voxel_count,
            "centroid_mni_mm": centroid,
        })
    return rows


def _write_montage(mean_img: Any, labels_img: Any, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
        import matplotlib.pyplot as plt

    mean_data = np.asarray(mean_img.dataobj, dtype=np.float32)
    label_data = np.asarray(labels_img.dataobj)
    nonzero = np.where(np.any(label_data > 0, axis=(0, 1)))[0]
    if nonzero.size == 0:
        raise ValueError("The resampled atlas has no labeled voxels in the BOLD field of view")
    slices = np.unique(np.linspace(nonzero.min(), nonzero.max(), 12).astype(int))
    finite = mean_data[np.isfinite(mean_data)]
    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor="#0c1117")
    for axis, z_index in zip(axes.flat, slices, strict=False):
        image = np.rot90(mean_data[:, :, z_index])
        atlas_slice = np.rot90(label_data[:, :, z_index])
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        boundaries = np.zeros_like(atlas_slice, dtype=bool)
        boundaries[1:, :] |= atlas_slice[1:, :] != atlas_slice[:-1, :]
        boundaries[:, 1:] |= atlas_slice[:, 1:] != atlas_slice[:, :-1]
        boundaries &= atlas_slice > 0
        overlay = np.zeros((*boundaries.shape, 4), dtype=float)
        overlay[boundaries] = (0.278, 0.843, 0.675, 0.7)
        axis.imshow(overlay, interpolation="nearest")
        world_z = float((mean_img.affine @ np.array([0, 0, z_index, 1]))[2])
        axis.set_title(f"z={world_z:.1f} mm", color="#dce7f0", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(slices):]:
        axis.axis("off")
    fig.suptitle("Public MNI BOLD mean with Schaefer-100 parcel boundaries", color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_public_roi_connectivity(
    *,
    max_frames: int = 168,
    fd_threshold_mm: float = 0.5,
    std_dvars_threshold: float | None = None,
    connectivity_threshold: float = 0.3,
    run_root: Path | str | None = None,
    data_root: Path | str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Validate atlas ROI extraction and connectivity on public MNI-space BOLD."""
    if not 20 <= max_frames <= 168:
        raise ValueError("max_frames must be in [20, 168]")
    if fd_threshold_mm < 0 or (std_dvars_threshold is not None and std_dvars_threshold < 0):
        raise ValueError("scrubbing thresholds must be non-negative")
    if not 0 < connectivity_threshold < 1:
        raise ValueError("connectivity_threshold must be in (0, 1)")

    import nibabel as nib
    import pandas as pd
    from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_development_fmri
    from nilearn.image import mean_img, resample_to_img
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.signal import clean

    started = time.perf_counter()
    cache_root = Path(data_root) if data_root else Path.home() / ".cache" / "qortex" / "public" / "roi-connectivity"
    atlas = fetch_atlas_schaefer_2018(
        n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=cache_root
    )
    dataset = fetch_development_fmri(n_subjects=1, data_dir=cache_root)
    bold_path = Path(dataset.func[0]).resolve()
    confounds_path = Path(dataset.confounds[0]).resolve()
    atlas_path = Path(atlas.maps).resolve()
    labels = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in atlas.labels]
    if len(labels) != 100:
        raise ValueError(f"Schaefer atlas returned {len(labels)} labels; expected 100")
    if on_progress:
        on_progress(1, 6)

    bold_img = nib.load(str(bold_path))
    if len(bold_img.shape) != 4 or "space-MNI152NLin2009cAsym" not in bold_path.name:
        raise ValueError("Public validation input is not the expected preprocessed MNI-space 4-D BOLD image")
    frame_count = min(max_frames, bold_img.shape[3])
    confounds = pd.read_csv(confounds_path, sep="\t").iloc[:frame_count].copy()
    fd = confounds["framewise_displacement"].fillna(0.0).to_numpy(dtype=float)
    flagged_mask = fd > fd_threshold_mm
    std_dvars_available = "std_dvars" in confounds.columns
    if std_dvars_threshold is not None:
        if not std_dvars_available:
            raise ValueError("The fetched reduced-confounds contract has no std_dvars column")
        std_dvars = confounds["std_dvars"].fillna(0.0).to_numpy(dtype=float)
        flagged_mask |= std_dvars > std_dvars_threshold
    retained = np.flatnonzero(~flagged_mask)
    flagged = np.flatnonzero(flagged_mask)
    if retained.size < 20:
        raise ValueError(f"Scrubbing retained only {retained.size} frames; at least 20 are required")
    confound_columns = [
        "csf", "white_matter", "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"
    ]
    confound_matrix = confounds[confound_columns].fillna(0.0).to_numpy(dtype=float)

    clipped_img = bold_img.slicer[:, :, :, :frame_count]
    labels_img = resample_to_img(
        str(atlas_path), clipped_img, interpolation="nearest", force_resample=True
    )
    label_values = np.unique(np.asarray(labels_img.dataobj))
    if not np.array_equal(label_values, np.arange(101)):
        raise ValueError(
            f"Resampled atlas contains labels {label_values.tolist()}; expected background plus 1..100"
        )
    raw_masker = NiftiLabelsMasker(
        labels_img=labels_img, resampling_target=None,
        standardize=False, detrend=False, keep_masked_labels=False, reports=False,
    )
    raw_all_signals = raw_masker.fit_transform(clipped_img)
    raw_signals = raw_all_signals[retained]
    clean_signals = clean(
        raw_all_signals,
        confounds=confound_matrix,
        sample_mask=retained,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=float(bold_img.header.get_zooms()[3]),
        ensure_finite=True,
        extrapolate=False,
    )
    if clean_signals.shape != (retained.size, 100):
        raise ValueError(f"ROI extraction returned shape {clean_signals.shape}; expected {(retained.size, 100)}")
    if not np.isfinite(clean_signals).all():
        raise ValueError("ROI time series contain non-finite values after preprocessing")
    if on_progress:
        on_progress(2, 6)

    tr = float(bold_img.header.get_zooms()[3])
    connectivity = compute_pearson_connectivity(
        clean_signals.T,
        channel_names=labels,
        time_window_s=clean_signals.shape[0] * tr,
        sampling_hz=1.0 / tr,
        threshold=connectivity_threshold,
        scope=f"public-development-fmri:{bold_path.name}",
        input_signal_type="BOLD ROI",
        node_definition="Schaefer2018 100 parcels, 7 Yeo networks, MNI 2 mm",
    )
    graph = compute_graph_metrics(connectivity, scope=connectivity.spec.scope if hasattr(connectivity.spec, "scope") else bold_path.name)
    if on_progress:
        on_progress(3, 6)

    geometry = _roi_geometry(labels_img, labels)
    means = np.mean(raw_signals, axis=0)
    stds = np.std(raw_signals, axis=0, ddof=1)
    for index, row in enumerate(geometry):
        row["mean_signal"] = float(means[index])
        row["signal_std"] = float(stds[index])
        row["temporal_snr"] = float(means[index] / stds[index]) if stds[index] > 0 else None

    root = Path(run_root) if run_root else Path.home() / ".qortex" / "runs" / "roi-connectivity"
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"roiconn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    final_dir = root / run_id
    tmp_dir = root / f".{run_id}.tmp"
    tmp_dir.mkdir(parents=False, exist_ok=False)
    try:
        mean = mean_img(clipped_img)
        nib.save(mean, tmp_dir / "mean-bold.nii.gz")
        nib.save(labels_img, tmp_dir / "schaefer100-resampled.nii.gz")
        _write_montage(mean, labels_img, tmp_dir / "montage.png")
        with (tmp_dir / "roi-statistics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(geometry[0]))
            writer.writeheader()
            writer.writerows(geometry)
        np.savetxt(tmp_dir / "connectivity.csv", connectivity.matrix, delimiter=",", fmt="%.8g")
        if on_progress:
            on_progress(4, 6)

        artifacts = {
            "mean_volume": "mean-bold.nii.gz",
            "atlas_labels": "schaefer100-resampled.nii.gz",
            "montage": "montage.png",
            "roi_statistics": "roi-statistics.csv",
            "connectivity": "connectivity.csv",
            "provenance": "result.json",
        }
        inventory = {
            name: {"path": filename, "size_bytes": (tmp_dir / filename).stat().st_size, "sha256": _sha256(tmp_dir / filename)}
            for name, filename in artifacts.items() if name != "provenance"
        }
        matrix = np.asarray(connectivity.matrix)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        result = {
            "run_id": run_id,
            "kind": "public_roi_connectivity_validation",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "id": "development_fmri",
                "subject": str(dataset.phenotypic[0][0]),
                "source": _DATASET_SOURCE,
                "raw_openneuro_source": _RAW_DATASET_SOURCE,
                "license": "unrestricted for non-commercial research purposes",
                "spatial_reference": "MNI152NLin2009cAsym",
                "bold_path": str(bold_path),
                "bold_sha256": _sha256(bold_path),
                "confounds_path": str(confounds_path),
                "confounds_sha256": _sha256(confounds_path),
            },
            "atlas": {
                "id": "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm",
                "source": _ATLAS_SOURCE,
                "license": "not stated by the Nilearn fetcher; source and reference are recorded",
                "path": str(atlas_path),
                "sha256": _sha256(atlas_path),
                "n_regions": 100,
                "labels": labels,
            },
            "configuration": {
                "max_frames": max_frames,
                "frames_analyzed": frame_count,
                "fd_threshold_mm": fd_threshold_mm,
                "std_dvars_threshold": std_dvars_threshold,
                "std_dvars_available": std_dvars_available,
                "connectivity_threshold": connectivity_threshold,
                "confound_columns": confound_columns,
                "filter_hz": [0.01, 0.1],
                "detrend": True,
                "standardize": True,
            },
            "scrubbing": {
                "retained_count": int(retained.size),
                "flagged_count": int(flagged.size),
                "retained_frames": retained.tolist(),
                "flagged_frames": flagged.tolist(),
            },
            "roi_statistics": geometry,
            "connectivity": {
                "construction": _model_dict(connectivity.spec),
                "n_nonzero_edges": int(np.count_nonzero(upper)),
                "mean_absolute_retained_r": float(np.mean(np.abs(upper[upper != 0]))) if np.any(upper != 0) else None,
                "matrix_shape": list(matrix.shape),
            },
            "graph": _model_dict(graph),
            "runtime": {"elapsed_seconds": time.perf_counter() - started, "environment": _environment()},
            "artifacts": artifacts,
            "artifact_inventory": inventory,
        }
        _write_json(tmp_dir / "result.json", result)
        if on_progress:
            on_progress(5, 6)
        os.replace(tmp_dir, final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if on_progress:
        on_progress(6, 6)
    return result


def load_public_roi_connectivity_run(run_id: str, *, run_root: Path | str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else Path.home() / ".qortex" / "runs" / "roi-connectivity").resolve()
    result_path = (root / run_id / "result.json").resolve()
    if root not in result_path.parents or not result_path.is_file():
        raise FileNotFoundError(f"No ROI-connectivity run {run_id!r}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run result {result_path} is not a JSON object")
    return payload


def public_roi_connectivity_artifact_path(run_id: str, artifact: str) -> Path:
    result = load_public_roi_connectivity_run(run_id)
    filename = result.get("artifacts", {}).get(artifact)
    if not isinstance(filename, str):
        raise KeyError(f"Run {run_id!r} has no artifact {artifact!r}")
    root = (Path.home() / ".qortex" / "runs" / "roi-connectivity").resolve()
    path = (root / run_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Artifact {artifact!r} is missing for run {run_id!r}")
    return path


__all__ = [
    "load_public_roi_connectivity_run",
    "public_roi_connectivity_artifact_path",
    "run_public_roi_connectivity",
]
