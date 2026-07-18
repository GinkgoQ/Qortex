"""Real local conversion jobs and artifact inventory for Qortex Atlas."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

from qortex.convert import ConversionPipeline
from qortex.convert.formats import get_writer
from qortex.convert.pipeline import _is_non_sample_metadata_file
from qortex.core.entities import Manifest
from qortex.parse._registry import LoaderRegistry

_FORMAT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "parquet": ("numpy",),
    "zarr": ("numpy", "zarr", "polars"),
    "hdf5": ("numpy", "h5py"),
    "webdataset": ("numpy",),
    "huggingface": ("numpy", "datasets"),
    "tfrecord": ("numpy", "tensorflow"),
}


def conversion_capabilities() -> dict[str, Any]:
    formats = []
    for name, requirements in _FORMAT_DEPENDENCIES.items():
        missing = [package for package in requirements if importlib.util.find_spec(package) is None]
        formats.append(
            {
                "name": name,
                "available": not missing,
                "availability_evidence": "Required Python packages are discoverable in the active Qortex environment.",
                "required_packages": list(requirements),
                "missing_packages": missing,
                "output_contract": {
                    "parquet": "Parquet shards plus artifact/provenance reports",
                    "zarr": "Zarr store plus artifact/provenance reports",
                    "hdf5": "HDF5 file plus artifact/provenance reports",
                    "webdataset": "Tar shards plus index and artifact/provenance reports",
                    "huggingface": "Dataset save_to_disk directory plus artifact/provenance reports",
                    "tfrecord": "TFRecord shards plus artifact/provenance reports",
                }[name],
            }
        )
    return {
        "formats": formats,
        "unsupported_proposal_outputs": {
            "nifti": "NIfTI is an input/source representation in the current conversion pipeline, not a sample-container writer.",
            "numpy": "No production NumPy artifact writer is registered; the API does not advertise one.",
        },
    }


def conversion_options(manifest: Manifest, data_dir: Path, *, limit: int = 500) -> dict[str, Any]:
    registry = LoaderRegistry()
    registry.discover()
    candidates = []
    total_local = 0
    total_convertible = 0
    unsafe_manifest_paths = 0
    for record in manifest.files:
        if record.is_dir or _is_non_sample_metadata_file(record):
            continue
        try:
            local_file = _local_source_path(data_dir, record.path)
        except ValueError:
            unsafe_manifest_paths += 1
            continue
        if not local_file.is_file():
            continue
        total_local += 1
        loader = registry.resolve(record)
        if loader is None:
            continue
        total_convertible += 1
        if len(candidates) < limit:
            candidates.append(
                {
                    "path": record.path,
                    "size_bytes": local_file.stat().st_size,
                    "subject": record.subject,
                    "session": record.session,
                    "task": record.task,
                    "modality": record.modality,
                    "suffix": record.suffix,
                    "extension": record.extension,
                    "loader": type(loader).__name__,
                    "parse_validated": False,
                }
            )
    return {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "local_file_count": total_local,
        "convertible_candidate_count": total_convertible,
        "unsafe_manifest_path_count": unsafe_manifest_paths,
        "candidates_truncated": total_convertible > len(candidates),
        "candidates": candidates,
        "candidate_evidence": "A registered loader and local file exist. Parsing is validated only when the conversion job runs.",
        **conversion_capabilities(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_source_path(data_dir: Path, relative_path: str) -> Path:
    root = data_dir.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"manifest path resolves outside the dataset root: {relative_path}")
    return candidate


def run_conversion(
    manifest: Manifest,
    data_dir: Path,
    output_dir: Path,
    *,
    paths: list[str],
    output_format: str,
    shard_size: int,
) -> dict[str, Any]:
    """Run a strict conversion over explicit manifest paths and inventory outputs."""
    if not paths or len(paths) > 100:
        raise ValueError("paths must contain between 1 and 100 explicit manifest paths")
    if len(paths) != len(set(paths)):
        raise ValueError("paths must not contain duplicates")
    if output_format not in _FORMAT_DEPENDENCIES:
        raise ValueError(f"unsupported output format: {output_format}")
    missing = next(
        item["missing_packages"]
        for item in conversion_capabilities()["formats"]
        if item["name"] == output_format
    )
    if missing:
        raise RuntimeError(f"{output_format} requires missing packages: {', '.join(missing)}")
    get_writer(output_format)  # validate registry availability before creating output

    by_path = {record.path: record for record in manifest.files}
    unknown = [path for path in paths if path not in by_path]
    if unknown:
        raise ValueError(f"paths are not in the immutable snapshot manifest: {unknown[:5]}")
    local_sources = {path: _local_source_path(data_dir, path) for path in paths}
    missing_local = [path for path, local_path in local_sources.items() if not local_path.is_file()]
    if missing_local:
        raise FileNotFoundError(f"selected files are not downloaded locally: {missing_local[:5]}")
    if output_dir.exists():
        raise FileExistsError(f"conversion output already exists: {output_dir}")

    subset = manifest.model_copy(update={"files": [by_path[path] for path in paths]})
    subset.rebuild_index()
    try:
        result = ConversionPipeline(
            subset,
            data_dir,
            output_dir,
            output_format=output_format,
            shard_size=shard_size,
            skip_missing=False,
        ).run()
    except Exception:
        # This directory did not exist before this call and is owned solely by
        # the failed run. Never publish a partial conversion as an artifact.
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        raise

    artifacts = []
    total_bytes = 0
    for artifact in sorted(output_dir.rglob("*")):
        if artifact.is_symlink() or not artifact.is_file():
            continue
        size = artifact.stat().st_size
        total_bytes += size
        artifacts.append(
            {
                "path": artifact.relative_to(output_dir).as_posix(),
                "size_bytes": size,
                "sha256": _sha256(artifact),
            }
        )
    run_record = {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "output_format": result.output_format,
        "output_dir": str(output_dir),
        "selected_paths": paths,
        "n_samples": result.n_samples,
        "n_subjects": result.n_subjects,
        "splits": result.splits,
        "elapsed_seconds": result.elapsed,
        "warnings": result.warnings,
        "artifact_manifest": result.artifact_manifest.model_dump(mode="json"),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_output_bytes": total_bytes,
    }
    record_path = output_dir / "run_record.json"
    record_path.write_text(json.dumps(run_record, indent=2, default=str), encoding="utf-8")
    record_size = record_path.stat().st_size
    artifacts.append({
        "path": record_path.name,
        "size_bytes": record_size,
        "sha256": _sha256(record_path),
    })
    return {
        **run_record,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_output_bytes": total_bytes + record_size,
        "run_record_evidence": (
            "run_record.json persists the measured result and hashes for all artifacts "
            "that existed before the record itself was written."
        ),
    }


def resolve_conversion_artifact(output_dir: Path, artifact_path: str) -> Path:
    root = output_dir.resolve()
    candidate = (root / artifact_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(artifact_path)
    return candidate


__all__ = [
    "conversion_capabilities",
    "conversion_options",
    "resolve_conversion_artifact",
    "run_conversion",
]
