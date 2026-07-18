"""Persistent, immutable fMRI QC reports and scrub plans."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _default_run_root() -> Path:
    return Path.home() / ".qortex" / "runs" / "fmri-qc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mean_volume(source: Path, destination: Path, *, memory_budget_bytes: int = 256_000_000) -> dict[str, Any]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("Persistent fMRI QC requires nibabel; install qortex with the mri extra") from None
    image = nib.load(str(source))
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError(f"fMRI QC requires a 4-D NIfTI with at least two volumes, got {image.shape}")
    voxels = int(np.prod(image.shape[:3]))
    bytes_per_frame = max(1, voxels * np.dtype(np.float32).itemsize)
    block_frames = max(1, min(image.shape[3], memory_budget_bytes // bytes_per_frame))
    accumulator = np.zeros(image.shape[:3], dtype=np.float64)
    count = 0
    for start in range(0, image.shape[3], block_frames):
        stop = min(image.shape[3], start + block_frames)
        block = np.asarray(image.dataobj[..., start:stop], dtype=np.float32)
        accumulator += np.sum(block, axis=3, dtype=np.float64)
        count += stop - start
    mean = np.asarray(accumulator / count, dtype=np.float32)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(mean, image.affine, header), str(destination))
    return {
        "method": "sequential float64 accumulation of every analyzed source volume; float32 NIfTI output",
        "source_volumes": count,
        "block_frames": block_frames,
        "memory_budget_bytes": memory_budget_bytes,
        "shape": list(mean.shape),
    }


def _write_frame_table(path: Path, report: dict[str, Any]) -> None:
    n_volumes = int(report["n_volumes_analyzed"])
    tr = report.get("tr_seconds")
    global_by_time = dict(zip(report["global_signal"]["time"], report["global_signal"]["values"]))
    dvars_by_time = dict(zip(report["dvars"]["time"], report["dvars"]["values"]))
    fd_report = report["framewise_displacement"]
    fd_by_time = dict(zip(fd_report.get("time", []), fd_report.get("values", [])))
    dvars_flagged = set(report["dvars"].get("flagged_volumes", []))
    fd_flagged = set(fd_report.get("flagged_volumes", []))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "volume", "time_seconds", "global_signal", "dvars", "framewise_displacement_mm",
            "flagged_by_dvars", "flagged_by_fd", "flagged",
        ])
        writer.writeheader()
        for volume in range(n_volumes):
            time_seconds = float(volume * tr) if tr is not None else None
            writer.writerow({
                "volume": volume,
                "time_seconds": time_seconds,
                "global_signal": global_by_time.get(time_seconds),
                "dvars": dvars_by_time.get(time_seconds),
                "framewise_displacement_mm": fd_by_time.get(time_seconds),
                "flagged_by_dvars": volume in dvars_flagged,
                "flagged_by_fd": volume in fd_flagged,
                "flagged": volume in dvars_flagged or volume in fd_flagged,
            })


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("qortex", "numpy", "nibabel"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def run_persistent_fmri_qc(
    *,
    dataset_id: str,
    snapshot: str,
    source_path: str,
    local_file: Path | str,
    max_frames: int = 500,
    fd_threshold_mm: float = 0.5,
    dvars_threshold: float | None = None,
    run_root: Path | str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute and persist a non-mutating QC report for one validated local BOLD source."""
    from qortex.visualize.volume import VolumeViewer

    source = Path(local_file).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Local BOLD source is unavailable or symbolic: {source}")
    started = time.perf_counter()
    source_hash = _sha256(source)
    if on_progress:
        on_progress(1, 5)
    report = VolumeViewer(source, modality="fmri").fmri_qc_report(
        max_frames=max_frames,
        fd_threshold_mm=fd_threshold_mm,
        dvars_threshold=dvars_threshold,
    )
    if on_progress:
        on_progress(2, 5)

    run_id = f"fmriqc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    root = Path(run_root) if run_root else _default_run_root()
    final_dir = root / run_id
    run_dir = root / f".{run_id}.tmp"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        mean_evidence = _mean_volume(source, run_dir / "mean-bold.nii.gz")
        if on_progress:
            on_progress(3, 5)
        _write_frame_table(run_dir / "framewise-qc.csv", report)
        flagged_volumes = set(report["scrubbing"]["flagged_volumes"])
        scrub = {
            "source": {
                "dataset_id": dataset_id,
                "snapshot": snapshot,
                "path": source_path,
                "sha256": source_hash,
            },
            "immutable_source": True,
            "thresholds": {
                "dvars": report["dvars"].get("threshold"),
                "framewise_displacement_mm": report["framewise_displacement"].get("threshold_mm"),
            },
            "flagged_volumes": report["scrubbing"]["flagged_volumes"],
            "retained_volumes": [
                index for index in range(report["n_volumes_analyzed"])
                if index not in flagged_volumes
            ],
            "note": report["scrubbing"]["note"],
        }
        _write_json(run_dir / "scrub-plan.json", scrub)
        if on_progress:
            on_progress(4, 5)
        result = {
            "run_id": run_id,
            "kind": "fmri_qc",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"id": dataset_id, "snapshot": snapshot},
            "source": {
                "path": source_path,
                "local_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
            },
            "configuration": {
                "max_frames": max_frames,
                "fd_threshold_mm": fd_threshold_mm,
                "dvars_threshold": dvars_threshold,
            },
            "runtime": {"elapsed_seconds": time.perf_counter() - started, "environment": _environment()},
            "report": report,
            "mean_volume": mean_evidence,
            "scrub_plan": scrub,
            "artifacts": {
                "mean_volume": "mean-bold.nii.gz",
                "framewise_table": "framewise-qc.csv",
                "scrub_plan": "scrub-plan.json",
                "provenance": "result.json",
            },
        }
        result["artifact_inventory"] = {
            name: {
                "path": filename,
                "size_bytes": (run_dir / filename).stat().st_size,
                "sha256": _sha256(run_dir / filename),
            }
            for name, filename in result["artifacts"].items()
            if name != "provenance"
        }
        _write_json(run_dir / "result.json", result)
        run_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    if on_progress:
        on_progress(5, 5)
    return result


def load_fmri_qc_run(run_id: str, *, run_root: Path | str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else _default_run_root()).resolve()
    result_path = (root / run_id / "result.json").resolve()
    if root not in result_path.parents or not result_path.is_file():
        raise FileNotFoundError(f"No persistent fMRI QC run {run_id!r}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run result {result_path} is not a JSON object")
    return payload


def fmri_qc_artifact_path(run_id: str, artifact: str) -> Path:
    result = load_fmri_qc_run(run_id)
    filename = result.get("artifacts", {}).get(artifact)
    if not isinstance(filename, str):
        raise KeyError(f"Run {run_id!r} has no artifact {artifact!r}")
    root = _default_run_root().resolve()
    path = (root / run_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Artifact {artifact!r} is missing for run {run_id!r}")
    return path


__all__ = ["fmri_qc_artifact_path", "load_fmri_qc_run", "run_persistent_fmri_qc"]
