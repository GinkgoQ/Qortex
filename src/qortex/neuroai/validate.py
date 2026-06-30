"""Validation utilities for NeuroAI runtime artifacts.

The validator checks the artifact as a reproducible run product, not just as a
directory with JSON files. It verifies manifest hashes, required sidecars,
prediction output structure, marker records, and consistency with the runtime
report when enough information is available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REQUIRED_SIDECARS = (
    "pipeline.yaml",
    "compatibility_report.json",
    "preprocess_plan.json",
    "runtime_report.json",
    "latency_report.json",
    "artifact_contract.json",
    "provenance.json",
    "warnings.json",
    "artifact_manifest.json",
)


@dataclass
class ArtifactValidationIssue:
    """Single artifact validation finding."""

    code: str
    message: str
    severity: str = "warning"
    path: str | None = None
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.path is not None:
            data["path"] = self.path
        if self.expected is not None:
            data["expected"] = self.expected
        if self.observed is not None:
            data["observed"] = self.observed
        return data


@dataclass
class ArtifactValidationReport:
    """Structured validation result for a NeuroAI artifact directory."""

    artifact_dir: str
    status: str
    issues: list[ArtifactValidationIssue] = field(default_factory=list)
    n_manifest_files: int = 0
    n_verified_files: int = 0
    n_prediction_records: int = 0
    n_marker_records: int = 0
    output_files: list[dict[str, Any]] = field(default_factory=list)
    runtime_report: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "status": self.status,
            "n_manifest_files": self.n_manifest_files,
            "n_verified_files": self.n_verified_files,
            "n_prediction_records": self.n_prediction_records,
            "n_marker_records": self.n_marker_records,
            "output_files": self.output_files,
            "runtime_report": self.runtime_report,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# NeuroAI Artifact Validation: {self.status}",
            "",
            f"- Artifact: `{self.artifact_dir}`",
            f"- Manifest files: {self.n_manifest_files}",
            f"- Verified files: {self.n_verified_files}",
            f"- Prediction records: {self.n_prediction_records}",
            f"- Marker records: {self.n_marker_records}",
        ]
        if self.output_files:
            lines.extend(["", "## Outputs", "", "| Path | Type | Records | Markers |", "|---|---:|---:|---:|"])
            for item in self.output_files:
                lines.append(
                    f"| `{item.get('path', '')}` | {item.get('type', '')} | "
                    f"{item.get('prediction_records', item.get('records', 0))} | "
                    f"{item.get('marker_records', 0)} |"
                )
        if self.issues:
            lines.extend(["", "## Issues", "", "| Severity | Code | Path | Message |", "|---|---|---|---|"])
            for issue in self.issues:
                lines.append(
                    f"| {issue.severity} | {issue.code} | `{issue.path or ''}` | "
                    f"{issue.message} |"
                )
        return "\n".join(lines)

    def summary(self) -> str:
        n_errors = sum(1 for issue in self.issues if issue.severity == "error")
        n_warnings = sum(1 for issue in self.issues if issue.severity == "warning")
        return (
            f"ArtifactValidationReport: {self.status} "
            f"({n_errors} errors, {n_warnings} warnings, "
            f"{self.n_verified_files}/{self.n_manifest_files} files verified, "
            f"{self.n_prediction_records} predictions, {self.n_marker_records} markers)"
        )


def validate_artifact(
    artifact_dir: str | Path,
    *,
    strict: bool = False,
) -> ArtifactValidationReport:
    """Validate a NeuroAI artifact directory.

    Parameters
    ----------
    artifact_dir:
        Directory produced by ``Pipeline.run(artifact_dir=...)``.
    strict:
        When True, structural warnings such as missing optional output metadata
        are promoted to errors.
    """
    root = Path(artifact_dir).expanduser().resolve()
    issues: list[ArtifactValidationIssue] = []
    runtime_report: dict[str, Any] = {}
    output_files: list[dict[str, Any]] = []
    n_manifest_files = 0
    n_verified_files = 0
    n_prediction_records = 0
    n_marker_records = 0

    if not root.is_dir():
        return ArtifactValidationReport(
            artifact_dir=str(root),
            status="FAIL",
            issues=[ArtifactValidationIssue(
                code="ARTIFACT_DIR_MISSING",
                message="Artifact directory does not exist or is not a directory.",
                severity="error",
                path=str(root),
            )],
        )

    for rel in _REQUIRED_SIDECARS:
        if not (root / rel).is_file():
            issues.append(ArtifactValidationIssue(
                code="REQUIRED_SIDECAR_MISSING",
                message=f"Required artifact sidecar is missing: {rel}",
                severity="error",
                path=rel,
            ))

    manifest_path = root / "artifact_manifest.json"
    manifest = _read_json(manifest_path, issues, "artifact_manifest.json")
    manifest_files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    if not isinstance(manifest_files, dict):
        issues.append(ArtifactValidationIssue(
            code="MANIFEST_FILES_INVALID",
            message="artifact_manifest.json does not contain a files mapping.",
            severity="error",
            path="artifact_manifest.json",
        ))
        manifest_files = {}
    n_manifest_files = len(manifest_files)

    for rel, expected in manifest_files.items():
        rel_str = str(rel)
        file_path = _safe_child(root, rel_str)
        if file_path is None:
            issues.append(ArtifactValidationIssue(
                code="MANIFEST_PATH_ESCAPE",
                message="Manifest entry resolves outside the artifact directory.",
                severity="error",
                path=rel_str,
            ))
            continue
        if not file_path.is_file():
            issues.append(ArtifactValidationIssue(
                code="MANIFEST_FILE_MISSING",
                message="Manifested file is missing from disk.",
                severity="error",
                path=rel_str,
            ))
            continue
        n_verified_files += 1
        size_expected = expected.get("size_bytes") if isinstance(expected, dict) else None
        sha_expected = expected.get("sha256") if isinstance(expected, dict) else None
        size_observed = file_path.stat().st_size
        if size_expected is not None and int(size_expected) != size_observed:
            issues.append(ArtifactValidationIssue(
                code="MANIFEST_SIZE_MISMATCH",
                message="File size does not match artifact_manifest.json.",
                severity="error",
                path=rel_str,
                expected=size_expected,
                observed=size_observed,
            ))
        if sha_expected:
            sha_observed = _sha256_file(file_path)
            if sha_observed != sha_expected:
                issues.append(ArtifactValidationIssue(
                    code="MANIFEST_HASH_MISMATCH",
                    message="SHA-256 digest does not match artifact_manifest.json.",
                    severity="error",
                    path=rel_str,
                    expected=sha_expected,
                    observed=sha_observed,
                ))

    runtime_report = _read_json(root / "runtime_report.json", issues, "runtime_report.json")
    if isinstance(runtime_report, dict):
        n_outputs_reported = runtime_report.get("n_outputs_written")
        if n_outputs_reported is not None and int(n_outputs_reported) < 0:
            issues.append(ArtifactValidationIssue(
                code="RUNTIME_OUTPUT_COUNT_INVALID",
                message="runtime_report.json has a negative n_outputs_written value.",
                severity="error",
                path="runtime_report.json",
                observed=n_outputs_reported,
            ))
    else:
        runtime_report = {}

    output_root = root / "outputs"
    if output_root.is_dir():
        for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
            rel_path = path.relative_to(root).as_posix()
            if path.suffix.lower() == ".jsonl":
                stats = _inspect_jsonl(path, rel_path, issues, strict=strict)
            elif path.suffix.lower() == ".csv":
                stats = _inspect_csv(path, rel_path, issues)
            else:
                stats = {
                    "path": rel_path,
                    "type": path.suffix.lower().lstrip(".") or "file",
                    "records": None,
                    "marker_records": 0,
                }
            output_files.append(stats)
            n_prediction_records += int(stats.get("prediction_records") or stats.get("records") or 0)
            n_marker_records += int(stats.get("marker_records") or 0)
    else:
        issues.append(ArtifactValidationIssue(
            code="OUTPUTS_DIR_MISSING",
            message="Artifact has no outputs/ directory. This is valid only for stream-only outputs.",
            severity="warning",
            path="outputs",
        ))

    jsonl_outputs = [item for item in output_files if item.get("type") == "jsonl"]
    if len(jsonl_outputs) == 1 and runtime_report:
        observed = int(jsonl_outputs[0].get("prediction_records") or 0)
        expected = runtime_report.get("n_windows_processed")
        if expected is not None and int(expected) != observed:
            issues.append(ArtifactValidationIssue(
                code="RUNTIME_WINDOW_OUTPUT_MISMATCH",
                message="JSONL prediction count does not match runtime_report.n_windows_processed.",
                severity="warning",
                path=jsonl_outputs[0].get("path"),
                expected=expected,
                observed=observed,
            ))

    if strict:
        for issue in issues:
            if issue.severity == "warning":
                issue.severity = "error"

    status = _status_from_issues(issues)
    return ArtifactValidationReport(
        artifact_dir=str(root),
        status=status,
        issues=issues,
        n_manifest_files=n_manifest_files,
        n_verified_files=n_verified_files,
        n_prediction_records=n_prediction_records,
        n_marker_records=n_marker_records,
        output_files=output_files,
        runtime_report=runtime_report,
    )


def _read_json(path: Path, issues: list[ArtifactValidationIssue], rel_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(ArtifactValidationIssue(
            code="JSON_READ_FAILED",
            message=f"Cannot read JSON sidecar: {exc}",
            severity="error",
            path=rel_path,
        ))
        return {}
    if not isinstance(payload, dict):
        issues.append(ArtifactValidationIssue(
            code="JSON_ROOT_INVALID",
            message="JSON sidecar root must be an object.",
            severity="error",
            path=rel_path,
        ))
        return {}
    return payload


def _inspect_jsonl(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
    *,
    strict: bool,
) -> dict[str, Any]:
    prediction_records = 0
    marker_records = 0
    malformed_records = 0
    missing_metadata = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_records += 1
                issues.append(ArtifactValidationIssue(
                    code="JSONL_RECORD_INVALID",
                    message=f"Line {line_no} is not valid JSON: {exc}",
                    severity="error",
                    path=rel_path,
                ))
                continue
            if record.get("record_type") == "event_marker":
                marker_records += 1
                confidence = record.get("confidence")
                if confidence is not None:
                    try:
                        conf = float(confidence)
                        if not 0.0 <= conf <= 1.0:
                            raise ValueError
                    except (TypeError, ValueError):
                        issues.append(ArtifactValidationIssue(
                            code="MARKER_CONFIDENCE_INVALID",
                            message="Event marker confidence must be in [0, 1].",
                            severity="warning",
                            path=rel_path,
                            observed=confidence,
                        ))
                continue
            if not record.get("output_type"):
                malformed_records += 1
                issues.append(ArtifactValidationIssue(
                    code="PREDICTION_OUTPUT_TYPE_MISSING",
                    message=f"Line {line_no} has no output_type field.",
                    severity="error",
                    path=rel_path,
                ))
                continue
            prediction_records += 1
            missing = [
                key for key in ("window_index", "input_shape", "preprocessed_shape")
                if key not in record
            ]
            if missing:
                missing_metadata += 1
                severity = "error" if strict else "warning"
                issues.append(ArtifactValidationIssue(
                    code="PREDICTION_METADATA_INCOMPLETE",
                    message=f"Line {line_no} is missing runtime metadata: {', '.join(missing)}",
                    severity=severity,
                    path=rel_path,
                ))
    return {
        "path": rel_path,
        "type": "jsonl",
        "prediction_records": prediction_records,
        "marker_records": marker_records,
        "malformed_records": malformed_records,
        "records_with_missing_metadata": missing_metadata,
    }


def _inspect_csv(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            records = sum(1 for _ in reader)
            fieldnames = set(reader.fieldnames or [])
    except (OSError, csv.Error) as exc:
        issues.append(ArtifactValidationIssue(
            code="CSV_READ_FAILED",
            message=f"Cannot read CSV output: {exc}",
            severity="error",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "csv", "records": 0, "marker_records": 0}
    missing = {"timestamp", "output_type", "window_index"} - fieldnames
    if missing:
        issues.append(ArtifactValidationIssue(
            code="CSV_SCHEMA_INCOMPLETE",
            message=f"CSV output is missing columns: {', '.join(sorted(missing))}",
            severity="warning",
            path=rel_path,
        ))
    return {
        "path": rel_path,
        "type": "csv",
        "records": records,
        "marker_records": 0,
        "columns": sorted(fieldnames),
    }


def _safe_child(root: Path, rel: str) -> Path | None:
    try:
        path = (root / rel).resolve()
        path.relative_to(root)
        return path
    except (OSError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _status_from_issues(issues: list[ArtifactValidationIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "FAIL"
    if issues:
        return "WARN"
    return "PASS"
