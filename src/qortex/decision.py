"""Decision-first workflows for real-world Qortex use.

These functions are intentionally built on the stable manifest, graph,
planning, readiness, artifact, and indexing models. They avoid inventing a
parallel routing system and make uncertainty explicit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from qortex.artifact import Artifact
from qortex.check import compute_readiness
from qortex.convert.pipeline import ConversionPipeline
from qortex.convert.splits import SplitSpec
from qortex.core.entities import DownloadPlan, Manifest, ReadinessFinding, SelectionSpec
from qortex.indexing import index_local_bids
from qortex.manifest.graph import ManifestGraph, get_manifest_graph
from qortex.plan.planner import DownloadPlanner

DecisionStatus = Literal["possible", "uncertain", "not_possible"]
MinimumGoal = Literal["label-check", "first-batch", "validation", "metadata"]
MINIMUM_GOALS: frozenset[str] = frozenset({"label-check", "first-batch", "validation", "metadata"})


class DecisionFinding(BaseModel):
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    path: str | None = None
    recommendation: str | None = None


class DoctorReport(BaseModel):
    dataset_id: str
    snapshot: str
    status: DecisionStatus
    summary: str
    n_recordings: int
    n_event_complete: int
    n_label_ready: int
    estimated_bytes: int
    can_download: bool
    can_convert: bool
    next_actions: list[str] = Field(default_factory=list)
    findings: list[DecisionFinding] = Field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Dataset : {self.dataset_id} ({self.snapshot})",
            f"Status  : {self.status}",
            f"Summary : {self.summary}",
            f"Records : {self.n_recordings}",
            f"Events  : {self.n_event_complete}/{self.n_recordings}",
            f"Labels  : {self.n_label_ready}/{self.n_recordings}",
            f"Size    : {self.estimated_bytes / 1e9:.2f} GB",
            f"Download: {self.can_download}",
            f"Convert : {self.can_convert}",
        ]
        if self.findings:
            lines.append("Findings:")
            for finding in self.findings:
                location = f" [{finding.path}]" if finding.path else ""
                lines.append(f"  {finding.severity.upper()}: {finding.code}{location}: {finding.message}")
        if self.next_actions:
            lines.append("Next actions:")
            for action in self.next_actions:
                lines.append(f"  - {action}")
        return "\n".join(lines)


class MinimumPlanReport(BaseModel):
    dataset_id: str
    snapshot: str
    goal: MinimumGoal
    status: DecisionStatus
    reason: str
    plan: DownloadPlan
    selected_recording_id: str | None = None
    next_command: str | None = None

    def to_text(self) -> str:
        lines = [
            f"Dataset : {self.dataset_id} ({self.snapshot})",
            f"Goal    : {self.goal}",
            f"Status  : {self.status}",
            f"Reason  : {self.reason}",
            self.plan.summary(),
        ]
        if self.selected_recording_id:
            lines.append(f"Recording: {self.selected_recording_id}")
        if self.next_command:
            lines.append(f"Next    : {self.next_command}")
        lines.append("Files:")
        for file in self.plan.files[:30]:
            lines.append(f"  {file.path} ({file.size or 0} bytes)")
        remaining = self.plan.n_files - min(self.plan.n_files, 30)
        if remaining > 0:
            lines.append(f"  ... {remaining} more file(s)")
        return "\n".join(lines)


class CanTrainReport(BaseModel):
    dataset_id: str
    snapshot: str
    status: DecisionStatus
    modality: str | None = None
    target: str | None = None
    label_status: Literal["confirmed", "candidate", "missing"]
    n_subjects: int
    n_recordings: int
    n_label_ready: int
    required_download_bytes: int
    suggested_split: str
    leakage_risks: list[str] = Field(default_factory=list)
    next_command: str | None = None
    findings: list[DecisionFinding] = Field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Dataset  : {self.dataset_id} ({self.snapshot})",
            f"Status   : {self.status}",
            f"Modality : {self.modality or 'any'}",
            f"Target   : {self.target or 'unspecified'}",
            f"Labels   : {self.label_status}",
            f"Subjects : {self.n_subjects}",
            f"Records  : {self.n_recordings}",
            f"Ready    : {self.n_label_ready}",
            f"Required : {self.required_download_bytes / 1e6:.1f} MB",
            f"Split    : {self.suggested_split}",
        ]
        if self.leakage_risks:
            lines.append("Leakage risks:")
            lines.extend(f"  - {risk}" for risk in self.leakage_risks)
        if self.findings:
            lines.append("Findings:")
            lines.extend(f"  {f.severity.upper()}: {f.code}: {f.message}" for f in self.findings)
        if self.next_command:
            lines.append(f"Next     : {self.next_command}")
        return "\n".join(lines)


class FirstBatchReport(BaseModel):
    status: DecisionStatus
    source: str
    n_rows: int = 0
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None
    required_plan: MinimumPlanReport | None = None

    def to_text(self) -> str:
        lines = [
            f"Status : {self.status}",
            f"Source : {self.source}",
            f"Rows   : {self.n_rows}",
        ]
        if self.columns:
            lines.append("Columns: " + ", ".join(self.columns))
        if self.message:
            lines.append(f"Message: {self.message}")
        if self.required_plan is not None:
            lines.append("Required plan:")
            lines.append(self.required_plan.to_text())
        if self.rows:
            lines.append("Preview:")
            for row in self.rows:
                lines.append("  " + json.dumps(row, sort_keys=True, default=str))
        return "\n".join(lines)


class LeakageReport(BaseModel):
    artifact_path: str
    status: DecisionStatus
    n_rows: int
    n_subject_leaks: int = 0
    n_source_leaks: int = 0
    n_derivative_sources: int = 0
    findings: list[DecisionFinding] = Field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Artifact        : {self.artifact_path}",
            f"Status          : {self.status}",
            f"Rows            : {self.n_rows}",
            f"Subject leaks   : {self.n_subject_leaks}",
            f"Source leaks    : {self.n_source_leaks}",
            f"Derivative refs : {self.n_derivative_sources}",
        ]
        if self.findings:
            lines.append("Findings:")
            for finding in self.findings:
                lines.append(f"  {finding.severity.upper()}: {finding.code}: {finding.message}")
        return "\n".join(lines)


class ContentStatusReport(BaseModel):
    path: str
    status: DecisionStatus
    n_files: int
    n_zero_byte: int = 0
    n_annex_pointer_like: int = 0
    n_missing_remote: int = 0
    n_extra_local: int = 0
    n_size_mismatches: int = 0
    findings: list[DecisionFinding] = Field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Path            : {self.path}",
            f"Status          : {self.status}",
            f"Files           : {self.n_files}",
            f"Zero-byte       : {self.n_zero_byte}",
            f"Annex pointers  : {self.n_annex_pointer_like}",
            f"Missing remote  : {self.n_missing_remote}",
            f"Extra local     : {self.n_extra_local}",
            f"Size mismatches : {self.n_size_mismatches}",
        ]
        if self.findings:
            lines.append("Findings:")
            for finding in self.findings:
                location = f" [{finding.path}]" if finding.path else ""
                lines.append(f"  {finding.severity.upper()}: {finding.code}{location}: {finding.message}")
        return "\n".join(lines)


class Recipe(BaseModel):
    dataset_id: str
    snapshot: str | None = None
    modality: str | None = None
    target: str | None = None
    split: str = "subject"
    goal: MinimumGoal = "first-batch"
    output_dir: str | None = None
    metadata_only: bool = False


def doctor(manifest: Manifest, *, local_path: Path | None = None) -> DoctorReport:
    readiness = compute_readiness(manifest, local_path=local_path, conversion_target="sklearn")
    findings = [_decision_from_readiness(finding) for finding in readiness.findings]
    status: DecisionStatus
    if not readiness.can_download or not readiness.can_convert:
        status = "not_possible"
    elif readiness.n_label_ready > 0:
        status = "possible"
    elif readiness.n_event_complete > 0:
        status = "uncertain"
    else:
        status = "uncertain"

    next_actions = _next_actions(manifest, readiness.n_label_ready, local_path)
    summary = _doctor_summary(readiness.n_recordings, readiness.n_event_complete, readiness.n_label_ready)
    return DoctorReport(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        status=status,
        summary=summary,
        n_recordings=readiness.n_recordings,
        n_event_complete=readiness.n_event_complete,
        n_label_ready=readiness.n_label_ready,
        estimated_bytes=readiness.estimated_bytes,
        can_download=readiness.can_download,
        can_convert=readiness.can_convert,
        next_actions=next_actions,
        findings=findings,
    )


def minimum_plan(
    manifest: Manifest,
    *,
    goal: MinimumGoal,
    modality: str | None = None,
    target: str | None = None,
    output_dir: Path | None = None,
) -> MinimumPlanReport:
    if goal not in MINIMUM_GOALS:
        allowed = ", ".join(sorted(MINIMUM_GOALS))
        raise ValueError(f"Unknown minimum goal {goal!r}. Use one of: {allowed}")
    target_dir = output_dir or Path.cwd() / manifest.dataset_id
    graph = get_manifest_graph(manifest)
    recording = _choose_recording(graph, modality=modality, needs_events=goal in {"label-check", "first-batch"})
    if goal in {"metadata", "label-check"}:
        spec = SelectionSpec(metadata_only=True, modalities=[modality] if modality else None)
        reason = "Metadata, sidecars, and event tables are enough to inspect labels and dataset structure."
        status: DecisionStatus = "possible"
    elif goal == "validation":
        spec = SelectionSpec(modalities=[modality] if modality else None, include_derivatives=False)
        reason = "Official BIDS validation needs the local dataset content selected for validation."
        status = "possible"
    else:
        if recording is None:
            spec = SelectionSpec(metadata_only=True, modalities=[modality] if modality else None)
            reason = "No event-complete primary recording was found; metadata is the smallest useful next step."
            status = "uncertain"
        else:
            spec = SelectionSpec(include=[recording.primary.path], with_companions=True)
            reason = "A first batch needs one loadable primary recording plus required companions."
            status = "possible"
    plan = DownloadPlanner(check_disk_space=False).plan(manifest, spec, target_dir)
    cmd_goal = goal
    cmd = f"qortex minimum {manifest.dataset_id} --goal {cmd_goal}"
    if modality:
        cmd += f" --modality {modality}"
    if target:
        cmd += f" --target {target}"
    return MinimumPlanReport(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        goal=goal,
        status=status,
        reason=reason,
        plan=plan,
        selected_recording_id=recording.id if recording else None,
        next_command=cmd,
    )


def can_train(
    manifest: Manifest,
    *,
    modality: str | None = None,
    target: str | None = None,
    local_path: Path | None = None,
) -> CanTrainReport:
    readiness = compute_readiness(manifest, local_path=local_path, conversion_target="sklearn")
    min_report = minimum_plan(manifest, goal="first-batch", modality=modality, target=target)
    label_status: Literal["confirmed", "candidate", "missing"]
    if readiness.n_label_ready:
        label_status = "confirmed"
    elif readiness.n_event_complete:
        label_status = "candidate"
    else:
        label_status = "missing"
    risks = _training_leakage_risks(manifest)
    status: DecisionStatus
    if label_status == "confirmed" and len(manifest.summary.subjects) >= 2:
        status = "possible"
    elif label_status == "candidate":
        status = "uncertain"
    else:
        status = "not_possible"
    findings = [_decision_from_readiness(finding) for finding in readiness.findings]
    if label_status == "candidate":
        findings.append(DecisionFinding(
            severity="warning",
            code="labels.need_local_confirmation",
            message="Event files exist, but labels are not confirmed until local columns are inspected.",
            recommendation="Run qortex minimum with --goal label-check, download the plan, then rerun can-train with --local-path.",
        ))
    return CanTrainReport(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        status=status,
        modality=modality,
        target=target,
        label_status=label_status,
        n_subjects=manifest.summary.n_subjects,
        n_recordings=readiness.n_recordings,
        n_label_ready=readiness.n_label_ready,
        required_download_bytes=min_report.plan.estimated_bytes,
        suggested_split="subject" if manifest.summary.n_subjects >= 2 else "not meaningful: fewer than two subjects",
        leakage_risks=risks,
        next_command=f"qortex minimum {manifest.dataset_id} --goal first-batch" + (f" --modality {modality}" if modality else ""),
        findings=findings,
    )


def first_batch(
    *,
    artifact_path: Path | None = None,
    manifest: Manifest | None = None,
    local_path: Path | None = None,
    modality: str | None = None,
    target: str | None = None,
    limit: int = 8,
) -> FirstBatchReport:
    if artifact_path is not None:
        return _first_batch_from_artifact(artifact_path, limit=limit)
    if manifest is None:
        raise ValueError("first_batch requires either artifact_path or manifest")
    if local_path is None:
        return FirstBatchReport(
            status="uncertain",
            source=manifest.dataset_id,
            message="No local data path was provided; returning the smallest required first-batch plan.",
            required_plan=minimum_plan(manifest, goal="first-batch", modality=modality, target=target),
        )
    with tempfile.TemporaryDirectory(prefix="qortex-first-batch-") as tmp:
        out = Path(tmp) / "artifact"
        result = ConversionPipeline(
            manifest=manifest,
            data_dir=local_path,
            output_dir=out,
            output_format="parquet",
            split_spec=SplitSpec(strategy="subject"),
            shard_size=max(100, limit),
        ).run()
        report = _first_batch_from_artifact(out, limit=limit)
        report.message = f"Converted {result.n_samples} sample(s) before reading first rows."
        return report


def leakage_check(artifact_path: Path) -> LeakageReport:
    Artifact.open(artifact_path)
    rows = _read_artifact_rows(artifact_path)
    findings: list[DecisionFinding] = []
    subject_leaks = _count_value_split_leaks(rows, "subject")
    source_leaks = _count_value_split_leaks(rows, "source_path")
    derivative_sources = sum(1 for row in rows if str(row.get("source_path") or "").startswith("derivatives/"))
    if subject_leaks:
        findings.append(DecisionFinding(
            severity="error",
            code="leakage.subject_across_splits",
            message=f"{subject_leaks} subject(s) appear in more than one split.",
            recommendation="Use subject-aware splitting and regenerate the artifact.",
        ))
    if source_leaks:
        findings.append(DecisionFinding(
            severity="error",
            code="leakage.source_across_splits",
            message=f"{source_leaks} source file(s) appear in more than one split.",
            recommendation="Split before windowing or group by source file.",
        ))
    if derivative_sources:
        findings.append(DecisionFinding(
            severity="warning",
            code="leakage.derivative_sources",
            message=f"{derivative_sources} sample(s) originate from derivative paths.",
            recommendation="Check that derivatives do not encode target information.",
        ))
    status: DecisionStatus = "not_possible" if any(f.severity == "error" for f in findings) else "possible"
    return LeakageReport(
        artifact_path=str(artifact_path),
        status=status,
        n_rows=len(rows),
        n_subject_leaks=subject_leaks,
        n_source_leaks=source_leaks,
        n_derivative_sources=derivative_sources,
        findings=findings,
    )


def content_status(path: Path, *, manifest: Manifest | None = None) -> ContentStatusReport:
    if not path.exists():
        return ContentStatusReport(
            path=str(path),
            status="not_possible",
            n_files=0,
            findings=[DecisionFinding(severity="error", code="content.path_missing", message="Path does not exist.")],
        )
    files = [p for p in path.rglob("*") if p.is_file()]
    zero_byte = [p for p in files if p.stat().st_size == 0]
    annex = [p for p in files if _looks_like_annex_pointer(p)]
    missing = extra = mismatches = 0
    findings: list[DecisionFinding] = []
    if zero_byte:
        findings.append(DecisionFinding(
            severity="error",
            code="content.zero_byte",
            message=f"{len(zero_byte)} zero-byte file(s) found.",
            path=str(zero_byte[0].relative_to(path)),
        ))
    if annex:
        findings.append(DecisionFinding(
            severity="warning",
            code="content.annex_pointer_like",
            message=f"{len(annex)} git-annex pointer-like file(s) found.",
            path=str(annex[0].relative_to(path)),
        ))
    if manifest is not None:
        index = index_local_bids(path, manifest=manifest, use_pybids=False)
        missing = index.n_missing
        extra = index.n_extra
        mismatches = index.n_size_mismatches
        if missing:
            findings.append(DecisionFinding(
                severity="warning",
                code="content.missing_remote",
                message=f"{missing} manifest file(s) are not present locally.",
            ))
        if extra:
            findings.append(DecisionFinding(
                severity="warning",
                code="content.extra_local",
                message=f"{extra} local file(s) are not in the manifest.",
            ))
        if mismatches:
            findings.append(DecisionFinding(
                severity="warning",
                code="content.size_mismatch",
                message=f"{mismatches} local file(s) differ from manifest size.",
            ))
    status: DecisionStatus = "not_possible" if zero_byte else "uncertain" if findings else "possible"
    return ContentStatusReport(
        path=str(path),
        status=status,
        n_files=len(files),
        n_zero_byte=len(zero_byte),
        n_annex_pointer_like=len(annex),
        n_missing_remote=missing,
        n_extra_local=extra,
        n_size_mismatches=mismatches,
        findings=findings,
    )


def write_recipe(recipe: Recipe, path: Path) -> Path:
    path.write_text(json.dumps(recipe.model_dump(), indent=2), encoding="utf-8")
    return path


def read_recipe(path: Path) -> Recipe:
    return Recipe(**json.loads(path.read_text(encoding="utf-8")))


def _choose_recording(graph: ManifestGraph, *, modality: str | None, needs_events: bool):
    for recording in graph.recordings():
        if modality and recording.modality != modality:
            continue
        if needs_events and not recording.has_events:
            continue
        if not recording.downloadable:
            continue
        return recording
    return None


def _decision_from_readiness(finding: ReadinessFinding) -> DecisionFinding:
    return DecisionFinding(
        severity=finding.severity,
        code=finding.code,
        message=finding.message,
        path=finding.path,
        recommendation=finding.recommendation,
    )


def _next_actions(manifest: Manifest, n_label_ready: int, local_path: Path | None) -> list[str]:
    actions = [
        f"qortex metadata {manifest.dataset_id} --limit 20",
        f"qortex minimum {manifest.dataset_id} --goal label-check",
    ]
    if local_path is None:
        actions.append(f"qortex metadata {manifest.dataset_id} --download --output-dir ./data/{manifest.dataset_id}-metadata")
    if n_label_ready:
        actions.append(f"qortex can-train {manifest.dataset_id} --local-path ./data/{manifest.dataset_id}-metadata")
    else:
        actions.append("Download metadata and rerun doctor/can-train with --local-path to confirm labels.")
    return actions


def _doctor_summary(n_recordings: int, n_event_complete: int, n_label_ready: int) -> str:
    if n_label_ready:
        return "Local labels are confirmed for at least part of the dataset."
    if n_event_complete:
        return "Event files are present, but labels need local confirmation."
    if n_recordings:
        return "Recordings are present, but no label-ready path is confirmed yet."
    return "No primary recordings were found."


def _training_leakage_risks(manifest: Manifest) -> list[str]:
    risks: list[str] = []
    if manifest.summary.n_subjects < 2:
        risks.append("Fewer than two subjects; subject-safe train/test splits are not meaningful.")
    if manifest.summary.has_derivatives:
        risks.append("Dataset contains derivatives; check derivative leakage before training.")
    if not manifest.summary.has_events:
        risks.append("No event files are present; supervised training labels may be unavailable.")
    return risks


def _first_batch_from_artifact(path: Path, *, limit: int) -> FirstBatchReport:
    Artifact.open(path)
    rows = _read_artifact_rows(path)
    preview = rows[:limit]
    columns = list(preview[0]) if preview else []
    return FirstBatchReport(
        status="possible" if preview else "not_possible",
        source=str(path),
        n_rows=len(preview),
        columns=columns,
        rows=preview,
        message=None if preview else "Artifact contains no rows.",
    )


def _read_artifact_rows(path: Path) -> list[dict[str, Any]]:
    shards = sorted(path.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found in artifact: {path}")
    try:
        import polars as pl
    except ImportError:
        import pyarrow.parquet as pq

        rows: list[dict[str, Any]] = []
        for shard in shards:
            rows.extend(pq.read_table(shard).to_pylist())
        return rows
    frames = [pl.read_parquet(shard) for shard in shards]
    return pl.concat(frames).to_dicts()


def _count_value_split_leaks(rows: list[dict[str, Any]], field: str) -> int:
    values: dict[str, set[str]] = {}
    for row in rows:
        value = row.get(field)
        split = row.get("split")
        if value is None or split is None:
            continue
        values.setdefault(str(value), set()).add(str(split))
    return sum(1 for splits in values.values() if len(splits) > 1)


def _looks_like_annex_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 512:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return (
        text.startswith("/annex/objects/")
        or "git-annex" in text
        or text.startswith("version https://git-lfs.github.com/spec/v1")
    )
