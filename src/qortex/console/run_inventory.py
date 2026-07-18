"""Persistent experiment and conversion run inventory for Qortex Atlas."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.core.config import get_config


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _timestamp_key(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _conversion_runs(cache_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    datasets_root = cache_root / "datasets"
    if not datasets_root.is_dir():
        return runs, errors
    for run_dir in datasets_root.glob("*/*/exports/cv-*"):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        try:
            dataset_id = run_dir.parents[2].name
            snapshot = run_dir.parents[1].name
            artifact = _read_object(run_dir / "artifact_manifest.json")
            provenance = _read_object(run_dir / "qortex_provenance.json")
            record_path = run_dir / "run_record.json"
            record = _read_object(record_path) if record_path.is_file() else {}
            artifact_files = [path for path in run_dir.rglob("*") if path.is_file() and not path.is_symlink()]
            recorded_artifacts = {
                item.get("path"): item
                for item in record.get("artifacts", [])
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            }
            runs.append({
                "run_id": run_dir.name,
                "kind": "conversion",
                "status": "completed" if (run_dir / "_SUCCESS").is_file() else "incomplete",
                "created_at": artifact.get("created_at") or provenance.get("created_at") or _mtime_iso(run_dir),
                "title": f"Convert {dataset_id}@{snapshot} to {artifact.get('output_format', 'unknown')}",
                "dataset": {"id": dataset_id, "snapshot": snapshot, "doi": artifact.get("doi")},
                "model": None,
                "operation": "convert",
                "metrics": {
                    "samples": artifact.get("n_samples"),
                    "subjects": artifact.get("n_subjects"),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                },
                "configuration": provenance.get("config", {}),
                "source_files": artifact.get("source_files", []),
                "artifacts": {
                    "count": len(artifact_files),
                    "total_bytes": sum(path.stat().st_size for path in artifact_files),
                    "hash_inventory_persisted": bool(record.get("artifacts")),
                    "files": [
                        {
                            "name": path.relative_to(run_dir).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": recorded_artifacts.get(path.relative_to(run_dir).as_posix(), {}).get("sha256"),
                        }
                        for path in sorted(artifact_files)
                    ],
                },
                "reproducibility": {
                    "qortex_version": provenance.get("qortex_version"),
                    "artifact_id": artifact.get("artifact_id"),
                    "source_files_recorded": bool(artifact.get("source_files")),
                    "configuration_recorded": bool(provenance.get("config")),
                    "environment_recorded": False,
                    "seed_recorded": False,
                    "limitations": [
                        "The conversion run does not capture a software/hardware environment snapshot.",
                        "A seed is not applicable unless a stochastic split or transform is configured.",
                    ],
                },
                "ranking": {"comparable": False, "reason": "Conversion and inference runs have different task metrics."},
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    return runs, errors


def _brats_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not run_root.is_dir():
        return runs, errors
    for result_path in run_root.glob("*/result.json"):
        run_dir = result_path.parent
        try:
            result = _read_object(result_path)
            runtime = result.get("runtime", {})
            reproducibility = result.get("reproducibility", {})
            environment = reproducibility.get("environment", {})
            artifact_files = [path for path in run_dir.iterdir() if path.is_file() and not path.is_symlink()]
            artifact_inventory = result.get("artifact_inventory", {})
            metrics = {
                f"dice_{name}": values.get("dice")
                for name, values in result.get("metrics", {}).items()
                if isinstance(values, dict)
            }
            metrics.update({
                "preprocess_seconds": runtime.get("preprocess_seconds"),
                "inference_seconds": runtime.get("inference_seconds"),
                "peak_allocated_bytes": runtime.get("peak_memory", {}).get("allocated_bytes"),
            })
            model = result.get("model", {})
            dataset = result.get("dataset", {})
            runs.append({
                "run_id": result.get("run_id") or run_dir.name,
                "kind": "pretrained_validation",
                "status": result.get("status", "unknown"),
                "created_at": _mtime_iso(result_path),
                "title": f"{model.get('id', 'model')} on {dataset.get('case_id', 'public case')}",
                "dataset": {
                    "id": dataset.get("repo_id"),
                    "snapshot": dataset.get("revision"),
                    "case_id": dataset.get("case_id"),
                    "license": dataset.get("license"),
                },
                "model": {
                    "id": model.get("id"),
                    "revision": model.get("revision"),
                    "checkpoint_sha256": model.get("checkpoint_sha256"),
                    "license": model.get("license"),
                },
                "operation": "pretrained_segmentation_validation",
                "metrics": metrics,
                "configuration": {
                    "device": runtime.get("device"),
                    "precision": reproducibility.get("precision"),
                    "roi_size": runtime.get("roi_size"),
                    "overlap": runtime.get("overlap"),
                    "threshold": runtime.get("threshold"),
                },
                "source_files": result.get("input", {}).get("files", []),
                "artifacts": {
                    "count": len(artifact_files),
                    "total_bytes": sum(path.stat().st_size for path in artifact_files),
                    "hash_inventory_persisted": bool(artifact_inventory),
                    "files": [
                        {"name": name, **item}
                        for name, item in artifact_inventory.items()
                        if isinstance(item, dict)
                    ],
                },
                "reproducibility": {
                    "model_revision_recorded": bool(model.get("revision")),
                    "model_hash_recorded": bool(model.get("checkpoint_sha256")),
                    "dataset_revision_recorded": bool(dataset.get("revision")),
                    "input_hashes_recorded": bool(result.get("input", {}).get("files")),
                    "configuration_hashes_recorded": bool(reproducibility.get("model_config_sha256")),
                    "environment_recorded": bool(environment),
                    "seed_recorded": "seed" in reproducibility,
                    "environment": environment,
                    "seed": reproducibility.get("seed"),
                    "seed_evidence": reproducibility.get("seed_evidence"),
                    "limitations": [] if environment else [
                        "This run predates environment capture and cannot prove its complete software/hardware state."
                    ],
                },
                "ranking": {
                    "comparable": True,
                    "scope": "Only runs with the same model revision, public case revision, preprocessing hashes, and metric definition.",
                },
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    return runs, errors


def _detection_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not run_root.is_dir():
        return runs, errors
    for result_path in run_root.glob("*/result.json"):
        run_dir = result_path.parent
        try:
            result = _read_object(result_path)
            runtime = result.get("runtime", {})
            reproducibility = result.get("reproducibility", {})
            environment = reproducibility.get("environment", {})
            model = result.get("model", {})
            dataset = result.get("dataset", {})
            metrics = result.get("metrics", {})
            artifact_inventory = result.get("artifact_inventory", {})
            artifact_files = [path for path in run_dir.iterdir() if path.is_file() and not path.is_symlink()]
            runs.append({
                "run_id": result.get("run_id") or run_dir.name,
                "kind": "pretrained_detection_validation",
                "status": result.get("status", "unknown"),
                "created_at": _mtime_iso(result_path),
                "title": f"{model.get('id', 'detector')} on COCO image {dataset.get('image_id', 'unknown')}",
                "dataset": {
                    "id": dataset.get("id"),
                    "snapshot": dataset.get("split"),
                    "case_id": dataset.get("image_id"),
                    "license": dataset.get("image_license"),
                },
                "model": {
                    "id": model.get("id"),
                    "revision": model.get("weights"),
                    "checkpoint_sha256": model.get("checkpoint_sha256"),
                    "license": model.get("license"),
                },
                "operation": "pretrained_object_detection_validation",
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "precision", "recall", "mean_matched_iou", "true_positives",
                        "false_positives", "false_negatives", "ground_truth_objects",
                    )
                } | {
                    "preprocess_seconds": runtime.get("preprocess_seconds"),
                    "inference_seconds": runtime.get("inference_seconds"),
                    "peak_allocated_bytes": runtime.get("peak_memory", {}).get("allocated_bytes"),
                },
                "configuration": {
                    "device": runtime.get("device"),
                    "precision": runtime.get("precision"),
                    "score_threshold": runtime.get("score_threshold"),
                    "iou_threshold": metrics.get("iou_threshold"),
                    "preprocessing": runtime.get("preprocessing"),
                },
                "source_files": [{
                    "path": result.get("input", {}).get("file_name"),
                    "sha256": result.get("input", {}).get("sha256"),
                }],
                "artifacts": {
                    "count": len(artifact_files),
                    "total_bytes": sum(path.stat().st_size for path in artifact_files),
                    "hash_inventory_persisted": bool(artifact_inventory),
                    "files": [
                        {"name": name, **item}
                        for name, item in artifact_inventory.items()
                        if isinstance(item, dict)
                    ],
                },
                "reproducibility": {
                    "model_revision_recorded": bool(model.get("weights")),
                    "model_hash_recorded": bool(model.get("checkpoint_sha256")),
                    "dataset_revision_recorded": bool(dataset.get("annotation_archive_sha256")),
                    "input_hashes_recorded": bool(result.get("input", {}).get("sha256")),
                    "configuration_hashes_recorded": False,
                    "environment_recorded": bool(environment),
                    "seed_recorded": "seed" in reproducibility,
                    "environment": environment,
                    "seed": reproducibility.get("seed"),
                    "seed_evidence": reproducibility.get("seed_evidence"),
                    "limitations": [metrics.get("metric_scope")],
                },
                "ranking": {
                    "comparable": True,
                    "scope": "Only runs with identical weights, image hash, thresholds, preprocessing, and metric definition.",
                },
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    return runs, errors


def _roi_connectivity_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not run_root.is_dir():
        return runs, errors
    for result_path in run_root.glob("*/result.json"):
        run_dir = result_path.parent
        try:
            result = _read_object(result_path)
            runtime = result.get("runtime", {})
            dataset = result.get("dataset", {})
            atlas = result.get("atlas", {})
            graph = result.get("graph", {})
            connectivity = result.get("connectivity", {})
            scrubbing = result.get("scrubbing", {})
            artifacts = result.get("artifact_inventory", {})
            artifact_files = [path for path in run_dir.iterdir() if path.is_file() and not path.is_symlink()]
            files = [
                {"name": name, **item}
                for name, item in artifacts.items()
                if isinstance(item, dict)
            ]
            files.append({
                "name": "provenance",
                "path": "result.json",
                "size_bytes": result_path.stat().st_size,
                "sha256": _sha256(result_path),
            })
            runs.append({
                "run_id": result.get("run_id") or run_dir.name,
                "kind": "public_roi_connectivity_validation",
                "status": result.get("status", "unknown"),
                "created_at": result.get("created_at") or _mtime_iso(result_path),
                "title": f"Schaefer-100 ROI connectivity on {dataset.get('subject', 'public BOLD')}",
                "dataset": {
                    "id": dataset.get("id"),
                    "snapshot": dataset.get("spatial_reference"),
                    "case_id": dataset.get("subject"),
                    "license": dataset.get("license"),
                },
                "model": {
                    "id": atlas.get("id"),
                    "revision": "Schaefer2018 100 parcels, 7 networks, 2 mm",
                    "checkpoint_sha256": atlas.get("sha256"),
                    "license": atlas.get("license"),
                },
                "operation": "public_mni_roi_connectivity_validation",
                "metrics": {
                    "retained_frames": scrubbing.get("retained_count"),
                    "flagged_frames": scrubbing.get("flagged_count"),
                    "nonzero_edges": connectivity.get("n_nonzero_edges"),
                    "density": graph.get("density"),
                    "modularity": graph.get("modularity"),
                    "elapsed_seconds": runtime.get("elapsed_seconds"),
                },
                "configuration": result.get("configuration", {}),
                "source_files": [
                    {"path": dataset.get("bold_path"), "sha256": dataset.get("bold_sha256")},
                    {"path": dataset.get("confounds_path"), "sha256": dataset.get("confounds_sha256")},
                ],
                "artifacts": {
                    "count": len(artifact_files),
                    "total_bytes": sum(path.stat().st_size for path in artifact_files),
                    "hash_inventory_persisted": bool(artifacts),
                    "provenance_hash_computed_on_read": True,
                    "files": files,
                },
                "reproducibility": {
                    "model_revision_recorded": True,
                    "model_hash_recorded": bool(atlas.get("sha256")),
                    "dataset_revision_recorded": bool(dataset.get("spatial_reference")),
                    "input_hashes_recorded": bool(dataset.get("bold_sha256") and dataset.get("confounds_sha256")),
                    "configuration_hashes_recorded": False,
                    "environment_recorded": bool(runtime.get("environment")),
                    "seed_recorded": False,
                    "environment": runtime.get("environment", {}),
                    "seed": None,
                    "seed_evidence": "ROI extraction, confound regression, Pearson correlation, and graph calculations are deterministic.",
                    "limitations": [
                        "This is one public subject used to validate the execution path; it is not a population estimate."
                    ],
                },
                "ranking": {
                    "comparable": True,
                    "scope": "Only runs with identical BOLD/confound/atlas hashes, thresholds, frame bounds, and preprocessing.",
                },
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    return runs, errors


def _fmri_qc_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not run_root.is_dir():
        return runs, errors
    for result_path in run_root.glob("*/result.json"):
        run_dir = result_path.parent
        try:
            result = _read_object(result_path)
            report = result.get("report", {})
            runtime = result.get("runtime", {})
            source = result.get("source", {})
            dataset = result.get("dataset", {})
            artifacts = result.get("artifact_inventory", {})
            artifact_files = [path for path in run_dir.iterdir() if path.is_file() and not path.is_symlink()]
            inventory_files = [
                {"name": name, **item}
                for name, item in artifacts.items()
                if isinstance(item, dict)
            ]
            provenance_name = result.get("artifacts", {}).get("provenance")
            if isinstance(provenance_name, str) and (run_dir / provenance_name).is_file():
                provenance_path = run_dir / provenance_name
                inventory_files.append({
                    "name": "provenance",
                    "path": provenance_name,
                    "size_bytes": provenance_path.stat().st_size,
                    "sha256": _sha256(provenance_path),
                })
            runs.append({
                "run_id": result.get("run_id") or run_dir.name,
                "kind": "fmri_qc",
                "status": result.get("status", "unknown"),
                "created_at": result.get("created_at") or _mtime_iso(result_path),
                "title": f"fMRI QC for {dataset.get('id', 'dataset')}:{source.get('path', 'BOLD')}",
                "dataset": {
                    "id": dataset.get("id"),
                    "snapshot": dataset.get("snapshot"),
                    "case_id": source.get("path"),
                    "license": None,
                },
                "model": None,
                "operation": "framewise_fmri_qc",
                "metrics": {
                    "median_tsnr": report.get("tsnr", {}).get("median"),
                    "flagged_volumes": report.get("scrubbing", {}).get("flagged_count"),
                    "retained_volumes": report.get("scrubbing", {}).get("retained_count"),
                    "elapsed_seconds": runtime.get("elapsed_seconds"),
                },
                "configuration": result.get("configuration", {}),
                "source_files": [source],
                "artifacts": {
                    "count": len(artifact_files),
                    "total_bytes": sum(path.stat().st_size for path in artifact_files),
                    "hash_inventory_persisted": bool(artifacts),
                    "provenance_hash_computed_on_read": bool(provenance_name),
                    "files": inventory_files,
                },
                "reproducibility": {
                    "model_revision_recorded": False,
                    "model_hash_recorded": False,
                    "dataset_revision_recorded": bool(dataset.get("snapshot")),
                    "input_hashes_recorded": bool(source.get("sha256")),
                    "configuration_hashes_recorded": False,
                    "environment_recorded": bool(runtime.get("environment")),
                    "seed_recorded": False,
                    "environment": runtime.get("environment", {}),
                    "seed": None,
                    "seed_evidence": "No stochastic operation or model training is part of framewise fMRI QC.",
                    "limitations": [reason] if (
                        not report.get("framewise_displacement", {}).get("available")
                        and (reason := report.get("framewise_displacement", {}).get("unavailable_reason"))
                    ) else [],
                },
                "ranking": {
                    "comparable": True,
                    "scope": "Only runs with the same source SHA-256, thresholds, frame window, and Qortex/nibabel versions.",
                },
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(run_dir), "error": f"{type(exc).__name__}: {exc}"})
    return runs, errors


def persistent_run_inventory(*, limit: int = 100) -> dict[str, Any]:
    """Return completed artifact-backed runs, independent of process lifetime."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be in [1, 500]")
    conversions, conversion_errors = _conversion_runs(get_config().cache_dir)
    validations, validation_errors = _brats_runs(Path.home() / ".qortex" / "runs" / "brats-validation")
    detections, detection_errors = _detection_runs(Path.home() / ".qortex" / "runs" / "detection-validation")
    roi_connectivity, roi_connectivity_errors = _roi_connectivity_runs(Path.home() / ".qortex" / "runs" / "roi-connectivity")
    fmri_qc, fmri_qc_errors = _fmri_qc_runs(Path.home() / ".qortex" / "runs" / "fmri-qc")
    runs = sorted(
        [*conversions, *validations, *detections, *roi_connectivity, *fmri_qc],
        key=lambda item: _timestamp_key(item.get("created_at")),
        reverse=True,
    )
    return {
        "runs": runs[:limit],
        "total_discovered": len(runs),
        "truncated": len(runs) > limit,
        "scan_errors": [
            *conversion_errors, *validation_errors, *detection_errors,
            *roi_connectivity_errors, *fmri_qc_errors,
        ],
        "evidence": "Only persisted run directories with readable result/provenance artifacts are listed.",
    }


__all__ = ["persistent_run_inventory"]
