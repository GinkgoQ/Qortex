"""Artifact-backed official BIDS validation for locally downloaded content."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from qortex.core.entities import Manifest
from qortex.validation import validate_bids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_snapshot_scope(manifest: Manifest, data_dir: Path) -> dict[str, Any]:
    root = data_dir.resolve()
    manifest_files = [record for record in manifest.files if not record.is_dir]
    present = 0
    unsafe = 0
    local_bytes = 0
    for record in manifest_files:
        candidate = (root / record.path).resolve()
        if not candidate.is_relative_to(root):
            unsafe += 1
            continue
        if candidate.is_file():
            present += 1
            local_bytes += candidate.stat().st_size
    return {
        "manifest_file_count": len(manifest_files),
        "local_manifest_file_count": present,
        "local_manifest_bytes": local_bytes,
        "unsafe_manifest_path_count": unsafe,
        "snapshot_complete": present == len(manifest_files) and unsafe == 0,
        "scope_evidence": (
            "Completeness compares regular files under the local data root with every non-directory "
            "path in the immutable Qortex manifest. The validator result applies only to bytes present locally."
        ),
    }


def run_local_bids_validation(
    manifest: Manifest,
    data_dir: Path,
    output_dir: Path,
    *,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run the installed official validator and persist normalized/raw evidence."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"local dataset directory does not exist: {data_dir}")
    if output_dir.exists():
        raise FileExistsError(f"validation output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "validator_raw.json"
    normalized_path = output_dir / "validation_report.json"
    try:
        report = validate_bids(
            data_dir,
            output_json=raw_path,
            timeout_s=timeout_s,
            use_cache=False,
        )
        normalized_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        scope = _local_snapshot_scope(manifest, data_dir)
        artifacts = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (raw_path, normalized_path)
        ]
        issues = [
            issue.model_dump(mode="json", exclude={"raw"})
            for issue in report.issues[:500]
        ]
        return {
            "dataset_id": manifest.dataset_id,
            "snapshot": manifest.snapshot,
            "validator": "bids-validator",
            "validator_version": report.validator_version,
            "valid_local_content": report.valid,
            "return_code": report.return_code,
            "elapsed_seconds": report.elapsed,
            "counts": {
                "errors": report.n_errors,
                "warnings": report.n_warnings,
                "ignored": report.n_ignored,
                "passed_checks": None,
            },
            "passed_checks_evidence": (
                "bids-validator 1.x JSON does not report a total number of evaluated or passed rules."
            ),
            "issues": issues,
            "issues_truncated": len(report.issues) > len(issues),
            "scope": scope,
            "command": report.command,
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "total_artifact_bytes": sum(item["size_bytes"] for item in artifacts),
        }
    except Exception:
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        raise


def resolve_validation_artifact(output_dir: Path, artifact_name: str) -> Path:
    root = output_dir.resolve()
    candidate = (root / artifact_name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(artifact_name)
    return candidate


__all__ = ["resolve_validation_artifact", "run_local_bids_validation"]
