"""Observed BIDS subject/recording coverage without fabricated expectations."""

from __future__ import annotations

from typing import Any

from qortex.core.entities import Manifest

_DATA_EXTENSIONS = {
    ".nii", ".nii.gz", ".edf", ".bdf", ".set", ".fif", ".vhdr",
    ".nwb", ".snirf", ".mef", ".mefd", ".ds",
}


def observed_coverage_report(
    manifest: Manifest,
    *,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Return a paginated subject-by-observed-recording matrix.

    Empty cells are deliberately named ``not_observed``. BIDS metadata does
    not generally declare a Cartesian set of expected sessions/tasks/runs, so
    calling an absent combination ``missing`` would invent a requirement.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")

    observed: dict[str, set[tuple[str, str, str, str, str]]] = {}
    paths: dict[tuple[str, tuple[str, str, str, str, str]], list[str]] = {}
    for record in manifest.files:
        subject = record.entities.subject
        if record.is_dir or not subject or record.extension.lower() not in _DATA_EXTENSIONS:
            continue
        key = (
            record.entities.session or "",
            record.entities.task or "",
            record.entities.run or "",
            record.modality or "",
            record.suffix or "",
        )
        observed.setdefault(subject, set()).add(key)
        paths.setdefault((subject, key), []).append(record.path)

    subjects = sorted(observed, key=_bids_natural_key)
    columns = sorted(
        {column for values in observed.values() for column in values},
        key=lambda value: tuple(_bids_natural_key(item) for item in value),
    )
    page_subjects = subjects[offset: offset + limit]
    column_rows = [
        {
            "id": f"recording-{index}",
            "session": session or None,
            "task": task or None,
            "run": run or None,
            "modality": modality or None,
            "suffix": suffix or None,
            "label": _column_label(session, task, run, modality, suffix),
        }
        for index, (session, task, run, modality, suffix) in enumerate(columns)
    ]
    rows = []
    available_cells = 0
    for subject in page_subjects:
        cells = []
        for index, column in enumerate(columns):
            available = column in observed[subject]
            available_cells += int(available)
            cells.append({
                "column_id": f"recording-{index}",
                "status": "available" if available else "not_observed",
                "paths": sorted(paths.get((subject, column), [])),
            })
        rows.append({"subject": f"sub-{subject}", "cells": cells})

    denominator = len(page_subjects) * len(columns)
    return {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "absence_semantics": "not_observed",
        "absence_note": (
            "Empty cells are combinations observed elsewhere in this snapshot but not for this "
            "subject; they are not asserted to be required or missing."
        ),
        "subjects": rows,
        "columns": column_rows,
        "total_subjects": len(subjects),
        "offset": offset,
        "limit": limit,
        "available_cells": available_cells,
        "visible_cells": denominator,
        "observed_fraction": available_cells / denominator if denominator else None,
    }


def _column_label(session: str, task: str, run: str, modality: str, suffix: str) -> str:
    parts = []
    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    if modality:
        parts.append(modality)
    if suffix and suffix.lower() != (modality or "").lower():
        parts.append(suffix)
    if not modality and not suffix:
        parts.append("recording")
    return " · ".join(parts)


def _bids_natural_key(value: str) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
        if part
    )


__all__ = ["observed_coverage_report"]
