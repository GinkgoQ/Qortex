"""Observed BIDS subject/recording coverage without fabricated expectations."""

from __future__ import annotations

import hashlib
import json
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


def evaluate_coverage_expectations(
    manifest: Manifest,
    expectations: list[dict[str, Any]],
    *,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Evaluate an explicit subject-by-recording study-design contract.

    Selectors use exact BIDS entity fields. No absent cell is called missing
    unless its subject is declared in ``expected_subjects`` for that selector.
    """
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    if not 1 <= len(expectations) <= 500:
        raise ValueError("expectations must contain between 1 and 500 entries")

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

    known_subjects = set(observed) | {str(subject).removeprefix("sub-") for subject in manifest.summary.subjects}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, item in enumerate(expectations):
        selector = item.get("selector")
        if not isinstance(selector, dict):
            raise ValueError(f"expectations[{index}].selector must be an object")
        unexpected = set(selector) - {"session", "task", "run", "modality", "suffix"}
        if unexpected:
            raise ValueError(f"expectations[{index}].selector has unknown fields {sorted(unexpected)}")
        key = tuple(str(selector.get(name) or "") for name in ("session", "task", "run", "modality", "suffix"))
        if key in seen:
            raise ValueError(f"Duplicate expectation selector at index {index}")
        if not key[3] or not key[4]:
            raise ValueError(f"expectations[{index}] must declare modality and suffix")
        seen.add(key)
        raw_subjects = item.get("expected_subjects")
        if not isinstance(raw_subjects, list) or not raw_subjects:
            raise ValueError(f"expectations[{index}].expected_subjects must be a non-empty list")
        expected_subjects = {str(subject).removeprefix("sub-") for subject in raw_subjects}
        unknown_subjects = expected_subjects - known_subjects
        if unknown_subjects:
            raise ValueError(f"expectations[{index}] references unknown subjects {sorted(unknown_subjects)}")
        normalized.append({
            "id": str(item.get("id") or f"expectation-{index}"),
            "key": key,
            "expected_subjects": expected_subjects,
            "label": _column_label(*key),
        })

    subjects = sorted(known_subjects, key=_bids_natural_key)
    page_subjects = subjects[offset: offset + limit]
    rows = []
    counts = {"available": 0, "missing": 0, "not_expected": 0, "unexpected_available": 0}
    for subject in page_subjects:
        cells = []
        for item in normalized:
            key = item["key"]
            available = key in observed.get(subject, set())
            expected = subject in item["expected_subjects"]
            if available and expected:
                status = "available"
            elif available:
                status = "unexpected_available"
            elif expected:
                status = "missing"
            else:
                status = "not_expected"
            counts[status] += 1
            cells.append({
                "column_id": item["id"],
                "status": status,
                "expected": expected,
                "paths": sorted(paths.get((subject, key), [])),
            })
        rows.append({"subject": f"sub-{subject}", "cells": cells})

    contract_payload = [
        {
            "id": item["id"],
            "selector": dict(zip(("session", "task", "run", "modality", "suffix"), item["key"])),
            "expected_subjects": sorted(item["expected_subjects"], key=_bids_natural_key),
        }
        for item in normalized
    ]
    contract_hash = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "absence_semantics": "explicit_design_contract",
        "absence_note": (
            "Missing and not-expected states come only from the supplied exact selector and "
            "expected-subject contract; availability still comes from immutable manifest records."
        ),
        "contract": contract_payload,
        "contract_sha256": contract_hash,
        "subjects": rows,
        "columns": [
            {
                "id": item["id"],
                "session": item["key"][0] or None,
                "task": item["key"][1] or None,
                "run": item["key"][2] or None,
                "modality": item["key"][3],
                "suffix": item["key"][4],
                "label": item["label"],
                "expected_subject_count": len(item["expected_subjects"]),
            }
            for item in normalized
        ],
        "total_subjects": len(subjects),
        "offset": offset,
        "limit": limit,
        "counts": counts,
        "visible_cells": len(page_subjects) * len(normalized),
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


def _bids_natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    import re

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


__all__ = ["evaluate_coverage_expectations", "observed_coverage_report"]
