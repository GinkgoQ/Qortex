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
    artifact_contract = _read_json(root / "artifact_contract.json", issues, "artifact_contract.json")
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
            elif path.suffix.lower() == ".parquet":
                stats = _inspect_parquet(path, rel_path, issues)
            elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
                stats = _inspect_nifti(path, rel_path, issues)
            elif path.suffix.lower() == ".json":
                stats = _inspect_json_output(path, rel_path, issues)
            elif path.suffix.lower() == ".txt":
                stats = _inspect_yolo_txt(path, rel_path, issues)
            elif path.suffix.lower() in {".dcm", ".dicom"}:
                stats = _inspect_dicom(path, rel_path, issues)
            else:
                stats = {
                    "path": rel_path,
                    "type": path.suffix.lower().lstrip(".") or "file",
                    "records": None,
                    "marker_records": 0,
                }
            _validate_output_against_contract(stats, artifact_contract, issues)
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

    _validate_runtime_output_counts(runtime_report, output_files, issues)

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
    output_types: set[str] = set()
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
            output_types.add(str(record.get("output_type")))
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
        "output_types": sorted(output_types),
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


def _inspect_parquet(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        schema = meta.schema.to_arrow_schema()
        columns = list(schema.names)
        records = int(meta.num_rows)
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            columns = list(df.columns)
            records = int(len(df))
        except ImportError:
            issues.append(ArtifactValidationIssue(
                code="PARQUET_VALIDATOR_DEPENDENCY_MISSING",
                message="Parquet validation requires pyarrow or pandas.",
                severity="warning",
                path=rel_path,
            ))
            return {"path": rel_path, "type": "parquet", "records": None, "marker_records": 0}
        except Exception as exc:
            issues.append(ArtifactValidationIssue(
                code="PARQUET_READ_FAILED",
                message=f"Cannot read Parquet output: {exc}",
                severity="error",
                path=rel_path,
            ))
            return {"path": rel_path, "type": "parquet", "records": 0, "marker_records": 0}
    except Exception as exc:
        issues.append(ArtifactValidationIssue(
            code="PARQUET_READ_FAILED",
            message=f"Cannot read Parquet output: {exc}",
            severity="error",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "parquet", "records": 0, "marker_records": 0}
    missing = {"output_type"} - set(columns)
    if missing:
        issues.append(ArtifactValidationIssue(
            code="PARQUET_SCHEMA_INCOMPLETE",
            message=f"Parquet output is missing columns: {', '.join(sorted(missing))}",
            severity="warning",
            path=rel_path,
        ))
    return {
        "path": rel_path,
        "type": "parquet",
        "records": records,
        "marker_records": 0,
        "columns": columns,
    }


def _inspect_nifti(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    try:
        import nibabel as nib
        img = nib.load(str(path))
        shape = tuple(int(v) for v in img.shape)
        dtype = str(img.get_data_dtype())
        affine = img.affine.tolist()
    except ImportError:
        issues.append(ArtifactValidationIssue(
            code="NIFTI_VALIDATOR_DEPENDENCY_MISSING",
            message="NIfTI validation requires nibabel.",
            severity="warning",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "nifti", "records": 1, "marker_records": 0}
    except Exception as exc:
        issues.append(ArtifactValidationIssue(
            code="NIFTI_READ_FAILED",
            message=f"Cannot read NIfTI output: {exc}",
            severity="error",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "nifti", "records": 0, "marker_records": 0}
    if len(shape) < 3:
        issues.append(ArtifactValidationIssue(
            code="NIFTI_SHAPE_INVALID",
            message=f"NIfTI output should be at least 3D, got shape {shape}.",
            severity="warning",
            path=rel_path,
        ))
    return {
        "path": rel_path,
        "type": "nifti",
        "records": 1,
        "marker_records": 0,
        "shape": list(shape),
        "dtype": dtype,
        "affine": affine,
    }


def _inspect_json_output(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    payload = _read_json(path, issues, rel_path)
    if rel_path.endswith("coco.json") or {"images", "annotations", "categories"} <= set(payload):
        for key in ("images", "annotations", "categories"):
            if not isinstance(payload.get(key), list):
                issues.append(ArtifactValidationIssue(
                    code="COCO_SCHEMA_INVALID",
                    message=f"COCO output requires list field {key!r}.",
                    severity="error",
                    path=rel_path,
                ))
        return {
            "path": rel_path,
            "type": "coco",
            "records": len(payload.get("annotations", [])) if isinstance(payload.get("annotations"), list) else 0,
            "marker_records": 0,
            "n_images": len(payload.get("images", [])) if isinstance(payload.get("images"), list) else 0,
            "n_categories": len(payload.get("categories", [])) if isinstance(payload.get("categories"), list) else 0,
        }
    return {"path": rel_path, "type": "json", "records": None, "marker_records": 0}


def _inspect_yolo_txt(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    records = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) not in {5, 6}:
            issues.append(ArtifactValidationIssue(
                code="YOLO_RECORD_INVALID",
                message=f"YOLO line {line_no} must have 5 or 6 fields.",
                severity="error",
                path=rel_path,
            ))
            continue
        try:
            values = [float(v) for v in parts[1:5]]
        except ValueError:
            issues.append(ArtifactValidationIssue(
                code="YOLO_RECORD_INVALID",
                message=f"YOLO line {line_no} has non-numeric box coordinates.",
                severity="error",
                path=rel_path,
            ))
            continue
        if any(v < 0.0 or v > 1.0 for v in values):
            issues.append(ArtifactValidationIssue(
                code="YOLO_COORDINATE_RANGE_INVALID",
                message=f"YOLO line {line_no} has coordinates outside [0, 1].",
                severity="error",
                path=rel_path,
            ))
        records += 1
    return {"path": rel_path, "type": "yolo", "records": records, "marker_records": 0}


def _inspect_dicom(
    path: Path,
    rel_path: str,
    issues: list[ArtifactValidationIssue],
) -> dict[str, Any]:
    try:
        import pydicom
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        sop_class = str(getattr(ds, "SOPClassUID", ""))
        modality = str(getattr(ds, "Modality", ""))
    except ImportError:
        issues.append(ArtifactValidationIssue(
            code="DICOM_VALIDATOR_DEPENDENCY_MISSING",
            message="DICOM validation requires pydicom.",
            severity="warning",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "dicom", "records": 1, "marker_records": 0}
    except Exception as exc:
        issues.append(ArtifactValidationIssue(
            code="DICOM_READ_FAILED",
            message=f"Cannot read DICOM output: {exc}",
            severity="error",
            path=rel_path,
        ))
        return {"path": rel_path, "type": "dicom", "records": 0, "marker_records": 0}
    if modality not in {"SEG", "SR"}:
        issues.append(ArtifactValidationIssue(
            code="DICOM_OUTPUT_MODALITY_UNEXPECTED",
            message=f"DICOM output modality is {modality!r}; expected SEG or SR for NeuroAI outputs.",
            severity="warning",
            path=rel_path,
        ))
    return {
        "path": rel_path,
        "type": "dicom",
        "records": 1,
        "marker_records": 0,
        "modality": modality,
        "sop_class_uid": sop_class,
    }


def _validate_output_against_contract(
    stats: dict[str, Any],
    artifact_contract: dict[str, Any],
    issues: list[ArtifactValidationIssue],
) -> None:
    if not isinstance(artifact_contract, dict) or not artifact_contract:
        return
    expected_type = artifact_contract.get("output_type")
    if expected_type:
        expected = str(expected_type)
        observed_types = stats.get("output_types")
        if observed_types:
            if expected not in {str(v) for v in observed_types}:
                issues.append(ArtifactValidationIssue(
                    code="OUTPUT_TYPE_CONTRACT_MISMATCH",
                    message="Prediction records do not match artifact_contract.output_type.",
                    severity="warning",
                    path=stats.get("path"),
                    expected=expected,
                    observed=observed_types,
                ))
        elif not _file_type_satisfies_output_contract(str(stats.get("type", "")), expected):
            issues.append(ArtifactValidationIssue(
                code="OUTPUT_FILE_TYPE_CONTRACT_MISMATCH",
                message="Output file type is unusual for the declared artifact output_type.",
                severity="warning",
                path=stats.get("path"),
                expected=expected,
                observed=stats.get("type"),
            ))

    schema = _artifact_output_schema(artifact_contract)
    output_shape = schema.get("output_shape") if isinstance(schema, dict) else None
    if stats.get("type") == "nifti" and output_shape and stats.get("shape"):
        expected_shape = _positive_int_list(output_shape)
        observed_shape = [int(v) for v in stats.get("shape", [])]
        if expected_shape and observed_shape[: len(expected_shape)] != expected_shape:
            issues.append(ArtifactValidationIssue(
                code="NIFTI_OUTPUT_SHAPE_CONTRACT_MISMATCH",
                message="NIfTI output shape does not match artifact output schema.",
                severity="warning",
                path=stats.get("path"),
                expected=expected_shape,
                observed=observed_shape,
            ))


def _artifact_output_schema(artifact_contract: dict[str, Any]) -> dict[str, Any]:
    raw = artifact_contract.get("output_schema")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _file_type_satisfies_output_contract(file_type: str, output_type: str) -> bool:
    out = output_type.lower()
    ftype = file_type.lower()
    if ftype in {"jsonl", "csv", "parquet", "json"}:
        return True
    if ftype == "nifti":
        return any(token in out for token in ("segmentation", "mask", "volume", "nifti"))
    if ftype in {"coco", "yolo"}:
        return any(token in out for token in ("detection", "bbox", "object"))
    if ftype == "dicom":
        return any(token in out for token in ("dicom", "seg", "sr", "structured_report"))
    return True


def _positive_int_list(value: Any) -> list[int]:
    result: list[int] = []
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0:
            result.append(number)
    return result


def _validate_runtime_output_counts(
    runtime_report: dict[str, Any],
    output_files: list[dict[str, Any]],
    issues: list[ArtifactValidationIssue],
) -> None:
    declared = runtime_report.get("outputs") if isinstance(runtime_report, dict) else None
    if not isinstance(declared, list) or not declared:
        return
    declared_predictions = sum(int(item.get("n_prediction_records", 0) or 0) for item in declared if isinstance(item, dict))
    declared_markers = sum(int(item.get("n_marker_records", 0) or 0) for item in declared if isinstance(item, dict))
    observed_predictions = sum(int(item.get("prediction_records") or item.get("records") or 0) for item in output_files)
    observed_markers = sum(int(item.get("marker_records") or 0) for item in output_files)
    if declared_predictions and declared_predictions != observed_predictions:
        issues.append(ArtifactValidationIssue(
            code="RUNTIME_OUTPUT_PREDICTION_COUNT_MISMATCH",
            message="Observed output prediction records do not match runtime_report.outputs.",
            severity="warning",
            expected=declared_predictions,
            observed=observed_predictions,
        ))
    if declared_markers != observed_markers:
        issues.append(ArtifactValidationIssue(
            code="RUNTIME_OUTPUT_MARKER_COUNT_MISMATCH",
            message="Observed output marker records do not match runtime_report.outputs.",
            severity="warning",
            expected=declared_markers,
            observed=observed_markers,
        ))


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
