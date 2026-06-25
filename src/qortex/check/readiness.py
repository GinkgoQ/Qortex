"""Decision-oriented readiness checks for Qortex datasets."""

from __future__ import annotations

from pathlib import Path

from qortex.core.entities import (
    Manifest,
    ReadinessFinding,
    ReadinessReport,
)
from qortex.manifest.graph import LABEL_COLUMNS, ManifestGraph


def compute_readiness(
    manifest: Manifest,
    *,
    local_path: Path | None = None,
    conversion_target: str | None = None,
    inspect_loaders: bool = False,
) -> ReadinessReport:
    """Return an actionable readiness report for a manifest/local dataset."""
    graph = ManifestGraph(manifest)
    recordings = graph.recordings()
    findings: list[ReadinessFinding] = []
    n_loadable = 0
    n_event_complete = 0
    n_label_ready = 0

    if not manifest.summary.has_participants_tsv:
        findings.append(ReadinessFinding(
            severity="warning",
            code="metadata.participants_missing",
            message="participants.tsv is missing; subject-level metadata and some split checks will be limited.",
            recommendation="Prefer datasets with participants.tsv for ML reuse.",
        ))

    if not recordings:
        findings.append(ReadinessFinding(
            severity="error",
            code="convert.no_primary_recordings",
            message="No primary modality files were found in the manifest.",
            recommendation="Check the dataset snapshot or include derivatives if the dataset only contains derivatives.",
        ))

    registry = None
    if inspect_loaders:
        from qortex.parse._registry import LoaderRegistry

        registry = LoaderRegistry()
        registry.discover()

    for rec in recordings:
        primary = rec.primary
        if rec.downloadable:
            n_loadable += 1
        else:
            findings.append(ReadinessFinding(
                severity="error",
                code="download.no_url",
                message="Primary file has no download URL.",
                path=primary.path,
                recording_id=rec.id,
            ))

        if rec.has_events:
            n_event_complete += 1
        elif primary.task and primary.modality in {"eeg", "meg", "ieeg", "fnirs", "fmri"}:
            findings.append(ReadinessFinding(
                severity="warning",
                code="labels.events_missing",
                message="Task recording has no matching events.tsv file.",
                path=primary.path,
                recording_id=rec.id,
                recommendation="Use label_ready/event_complete selection to avoid unlabeled recordings.",
            ))

        label_ready = rec.has_labels
        if local_path is not None and rec.companions.events is not None:
            label_ready = _events_file_has_labels(local_path / rec.companions.events.path)
            if not label_ready:
                findings.append(ReadinessFinding(
                    severity="warning",
                    code="labels.column_missing",
                    message="Events file exists but no standard label column was detected.",
                    path=rec.companions.events.path,
                    recording_id=rec.id,
                    recommendation="Pass an explicit LabelPolicy column for this dataset.",
                ))
        elif rec.has_label_candidates:
            findings.append(ReadinessFinding(
                severity="info",
                code="labels.candidate_unverified",
                message="A matching events.tsv file exists, but label columns have not been verified locally.",
                path=rec.companions.events.path if rec.companions.events else None,
                recording_id=rec.id,
                recommendation="Run readiness with local_path or define an explicit LabelPolicy before conversion.",
            ))
        if label_ready:
            n_label_ready += 1

        if primary.modality in {"eeg", "meg", "ieeg", "fnirs"} and rec.companions.channels is None:
            findings.append(ReadinessFinding(
                severity="warning",
                code="load.channels_missing",
                message="No matching channels.tsv file was found.",
                path=primary.path,
                recording_id=rec.id,
            ))

        if primary.datatype == "dwi" and (rec.companions.bvec is None or rec.companions.bval is None):
            findings.append(ReadinessFinding(
                severity="error",
                code="load.dwi_gradients_missing",
                message="DWI image is missing bvec/bval gradient companions.",
                path=primary.path,
                recording_id=rec.id,
            ))

        if inspect_loaders and local_path is not None and (local_path / primary.path).exists():
            assert registry is not None
            loader = registry.resolve(primary)
            if loader is None:
                findings.append(ReadinessFinding(
                    severity="error",
                    code="load.loader_missing",
                    message=f"No loader is registered for modality {primary.modality!r}.",
                    path=primary.path,
                    recording_id=rec.id,
                    recommendation="Install the relevant Qortex extra or register a custom loader.",
                ))
            else:
                try:
                    loader.inspect(primary, local_path / primary.path)
                except Exception as exc:
                    findings.append(ReadinessFinding(
                        severity="error",
                        code="load.inspect_failed",
                        message=str(exc),
                        path=primary.path,
                        recording_id=rec.id,
                    ))

    subjects = {rec.subject for rec in recordings if rec.subject}
    if len(subjects) < 2 and recordings:
        findings.append(ReadinessFinding(
            severity="warning",
            code="split.too_few_subjects",
            message="Fewer than two subjects are available; subject-safe train/test splits are not meaningful.",
            recommendation="Use more subjects or choose a different split strategy.",
        ))

    if conversion_target and conversion_target in {"torch", "lightning", "sklearn"} and n_label_ready == 0:
        findings.append(ReadinessFinding(
            severity="warning",
            code="train.no_labels",
            message=f"No label-ready recordings were found for {conversion_target} training.",
            recommendation="Use label_ready=True selection or define a LabelPolicy.",
        ))

    return ReadinessReport(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        n_recordings=len(recordings),
        n_loadable=n_loadable,
        n_event_complete=n_event_complete,
        n_label_ready=n_label_ready,
        estimated_bytes=sum(r.estimated_bytes for r in recordings),
        findings=findings,
    )


def _events_file_has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import polars as pl

        df = pl.read_csv(
            path,
            separator="\t",
            n_rows=20,
            null_values=["n/a", "N/A", "NA", "NaN", "nan", ""],
            ignore_errors=True,
        )
    except Exception:
        return False
    return any(column in LABEL_COLUMNS for column in df.columns)
