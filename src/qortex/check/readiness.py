"""Decision-oriented readiness checks for Qortex datasets.

LabelPolicy integration
-----------------------
Pass an explicit ``LabelPolicy`` to ``compute_readiness()`` to override the
heuristic ``LABEL_COLUMNS`` scan with a deterministic policy:

    from qortex.core.entities import LabelPolicy
    from qortex.check.readiness import compute_readiness

    policy = LabelPolicy(
        source="events",
        column="trial_type",
        task="rest",
        missing="drop",
        positive_values=["target", "probe"],
    )
    report = compute_readiness(manifest, local_path=p, label_policy=policy)

Without an explicit policy the checker falls back to scanning for any column
in ``LABEL_COLUMNS`` (the seven standard BIDS event column names).
"""

from __future__ import annotations

from pathlib import Path

from qortex.core.entities import (
    LabelPolicy,
    Manifest,
    ReadinessFinding,
    ReadinessReport,
)
from qortex.manifest.graph import LABEL_COLUMNS, get_manifest_graph


def compute_readiness(
    manifest: Manifest,
    *,
    local_path: Path | None = None,
    conversion_target: str | None = None,
    inspect_loaders: bool = False,
    label_policy: LabelPolicy | None = None,
) -> ReadinessReport:
    """Return an actionable readiness report for a manifest/local dataset."""
    graph = get_manifest_graph(manifest)
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
            events_path = local_path / rec.companions.events.path
            label_ready, label_finding = _check_events_labels(
                events_path, rec, label_policy
            )
            if label_finding is not None:
                findings.append(label_finding)
        elif rec.has_label_candidates:
            policy_hint = (
                f" Explicit column: {label_policy.column!r}."
                if label_policy and label_policy.column
                else " Pass an explicit LabelPolicy column."
            )
            findings.append(ReadinessFinding(
                severity="info",
                code="labels.candidate_unverified",
                message=(
                    "A matching events.tsv file exists, but label columns have not "
                    "been verified locally." + policy_hint
                ),
                path=rec.companions.events.path if rec.companions.events else None,
                recording_id=rec.id,
                recommendation=(
                    "Run readiness with local_path or define an explicit LabelPolicy "
                    "before conversion."
                ),
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


def _check_events_labels(
    path: Path,
    rec: "LogicalRecording",
    policy: LabelPolicy | None,
) -> tuple[bool, ReadinessFinding | None]:
    """Check whether an events.tsv has usable labels given an optional LabelPolicy.

    Returns
    -------
    (label_ready, finding_or_None)
    """
    if not path.exists():
        return False, ReadinessFinding(
            severity="warning",
            code="labels.events_file_missing",
            message=f"Events file listed in manifest does not exist on disk: {path.name}",
            path=str(rec.companions.events.path if rec.companions.events else path),
            recording_id=rec.id,
            recommendation="Download or regenerate the events file before conversion.",
        )

    try:
        import polars as pl
        df = pl.read_csv(
            path,
            separator="\t",
            n_rows=50,
            null_values=["n/a", "N/A", "NA", "NaN", "nan", ""],
            ignore_errors=True,
        )
    except Exception as exc:
        return False, ReadinessFinding(
            severity="warning",
            code="labels.events_parse_error",
            message=f"Could not parse events file: {exc}",
            path=str(path),
            recording_id=rec.id,
        )

    if policy is not None and policy.column is not None:
        # Explicit policy: require the named column.
        if policy.column not in df.columns:
            return False, ReadinessFinding(
                severity="warning",
                code="labels.policy_column_missing",
                message=(
                    f"LabelPolicy requires column {policy.column!r} but it was not "
                    f"found in {path.name}. "
                    f"Available columns: {df.columns}"
                ),
                path=str(path),
                recording_id=rec.id,
                recommendation=(
                    f"Check the column name or update LabelPolicy.column. "
                    f"Found: {df.columns}"
                ),
            )
        col = df[policy.column]
        n_nulls = col.null_count()
        n_rows = len(col)
        if n_nulls == n_rows:
            return False, ReadinessFinding(
                severity="warning",
                code="labels.policy_column_all_null",
                message=(
                    f"Column {policy.column!r} is present but all values are null/n.a. "
                    f"in {path.name}."
                ),
                path=str(path),
                recording_id=rec.id,
                recommendation="Verify the events file content or choose a different column.",
            )
        if policy.missing == "error" and n_nulls > 0:
            return False, ReadinessFinding(
                severity="error",
                code="labels.policy_missing_values",
                message=(
                    f"Column {policy.column!r} has {n_nulls}/{n_rows} null values "
                    f"and LabelPolicy.missing='error'."
                ),
                path=str(path),
                recording_id=rec.id,
                recommendation=(
                    "Set LabelPolicy.missing='drop' to silently skip null rows, "
                    "or fix the events file."
                ),
            )
        if policy.positive_values:
            unique_vals = set(col.drop_nulls().unique().to_list())
            matched = unique_vals & set(str(v) for v in policy.positive_values)
            if not matched:
                return False, ReadinessFinding(
                    severity="warning",
                    code="labels.policy_no_positive_values",
                    message=(
                        f"None of the positive_values {policy.positive_values} "
                        f"appear in column {policy.column!r}. "
                        f"Found: {sorted(unique_vals)[:10]}"
                    ),
                    path=str(path),
                    recording_id=rec.id,
                    recommendation="Update LabelPolicy.positive_values to match the data.",
                )
        return True, None

    # Fallback: heuristic scan of standard BIDS label column names.
    if any(column in LABEL_COLUMNS for column in df.columns):
        return True, None

    return False, ReadinessFinding(
        severity="warning",
        code="labels.column_missing",
        message=(
            f"Events file {path.name} exists but contains none of the standard "
            f"BIDS label columns {sorted(LABEL_COLUMNS)}."
        ),
        path=str(path),
        recording_id=rec.id,
        recommendation=(
            "Pass an explicit LabelPolicy(column='your_column') or rename the "
            "events column to a standard BIDS name."
        ),
    )


# Keep backward-compatible name for any external callers.
def _events_file_has_labels(path: Path) -> bool:
    """Heuristic check — prefers explicit LabelPolicy via _check_events_labels."""
    if not path.exists():
        return False
    try:
        import polars as pl
        df = pl.read_csv(
            path, separator="\t", n_rows=20,
            null_values=["n/a", "N/A", "NA", "NaN", "nan", ""],
            ignore_errors=True,
        )
    except Exception:
        return False
    return any(column in LABEL_COLUMNS for column in df.columns)
