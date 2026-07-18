"""Persistent experiment and conversion run inventory for Qortex Atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.core.config import get_config


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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


def persistent_run_inventory(*, limit: int = 100) -> dict[str, Any]:
    """Return completed artifact-backed runs, independent of process lifetime."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be in [1, 500]")
    conversions, conversion_errors = _conversion_runs(get_config().cache_dir)
    validations, validation_errors = _brats_runs(Path.home() / ".qortex" / "runs" / "brats-validation")
    runs = sorted(
        [*conversions, *validations],
        key=lambda item: _timestamp_key(item.get("created_at")),
        reverse=True,
    )
    return {
        "runs": runs[:limit],
        "total_discovered": len(runs),
        "truncated": len(runs) > limit,
        "scan_errors": [*conversion_errors, *validation_errors],
        "evidence": "Only persisted run directories with readable result/provenance artifacts are listed.",
    }


__all__ = ["persistent_run_inventory"]
