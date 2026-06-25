"""Dataset and modality summary statistics — metadata-only, no file I/O."""

from __future__ import annotations

from qortex.core.entities import (
    DatasetSummary,
    FileRecord,
    Manifest,
    ModalitySummary,
)


def build_dataset_summary(manifest: Manifest) -> DatasetSummary:
    """Compute a top-level DatasetSummary from a Manifest."""
    s = manifest.summary
    return DatasetSummary(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        doi=manifest.doi,
        n_files=s.file_count,
        n_subjects=s.n_subjects,
        n_sessions=len(s.sessions),
        n_tasks=len(s.tasks),
        total_size=s.total_size,
        modalities=s.modalities,
        has_derivatives=s.has_derivatives,
        has_events=s.has_events,
    )


def build_modality_summaries(
    manifest: Manifest,
) -> dict[str, ModalitySummary]:
    """Build per-modality summaries from a Manifest."""
    result: dict[str, ModalitySummary] = {}

    non_dir = [f for f in manifest.files if not f.is_dir and f.modality]

    by_modality: dict[str, list[FileRecord]] = {}
    for f in non_dir:
        mod = f.modality or "unknown"
        by_modality.setdefault(mod, []).append(f)

    for modality, files in by_modality.items():
        subjects = {f.entities.subject for f in files if f.entities.subject}
        tasks = {f.entities.task for f in files if f.entities.task}
        extensions = {f.extension for f in files}
        total_size = sum(f.size or 0 for f in files)

        result[modality] = ModalitySummary(
            modality=modality,
            n_files=len(files),
            n_subjects=len(subjects),
            total_size=total_size,
            extensions=sorted(extensions),
            tasks=sorted(tasks),
        )

    return result


def file_table(manifest: Manifest):
    """Return all file records as a Polars DataFrame for interactive EDA."""
    import polars as pl

    rows = []
    for f in manifest.files:
        rows.append({
            "path": f.path,
            "filename": f.filename,
            "extension": f.extension,
            "size_mb": round(f.size / 1e6, 3) if f.size else None,
            "datatype": f.datatype,
            "modality": f.modality,
            "suffix": f.suffix,
            "subject": f.entities.subject,
            "session": f.entities.session,
            "task": f.entities.task,
            "run": f.entities.run,
            "is_dir": f.is_dir,
        })
    return pl.DataFrame(rows)


def coverage_matrix(manifest: Manifest):
    """Return a subject × session × task coverage matrix."""
    import polars as pl

    rows = []
    seen: set[tuple] = set()
    for f in manifest.files:
        if f.is_dir or not f.entities.subject:
            continue
        key = (f.entities.subject, f.entities.session, f.entities.task, f.modality)
        if key not in seen:
            seen.add(key)
            rows.append({
                "subject": f"sub-{f.entities.subject}",
                "session": f"ses-{f.entities.session}" if f.entities.session else None,
                "task": f.entities.task,
                "modality": f.modality,
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"subject": pl.Utf8, "session": pl.Utf8,
                "task": pl.Utf8, "modality": pl.Utf8}
    )
