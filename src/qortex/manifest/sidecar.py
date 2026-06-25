"""Sidecar grouping and inheritance chain detection.

BIDS sidecar files (``.json``, ``.tsv``, ``.bvec``/``.bval``) are resolved
via an inheritance chain from the dataset root down to the file level.

This module groups raw FileRecords so the ETL and EDA layers can quickly find
sidecars for any data file without a full PyBIDS scan.
"""

from __future__ import annotations

from collections import defaultdict

from qortex.core.entities import FileRecord

# Extensions that are always sidecars (never primary data)
SIDECAR_EXTENSIONS = frozenset({
    ".json", ".tsv", ".bvec", ".bval",
})

# Suffixes that identify sidecar roles
SIDECAR_SUFFIXES = frozenset({
    "events", "channels", "electrodes", "coordsystem",
    "scans", "sessions", "participants",
    "bold.json",          # json attached to bold
})


def group_sidecars(files: list[FileRecord]) -> dict[str, list[FileRecord]]:
    """Return a mapping: data file path → list of its likely sidecars.

    Uses a simple heuristic: a sidecar matches a data file when they share
    the same subject/session/task/run entities and the sidecar extension is
    in ``SIDECAR_EXTENSIONS``.
    """
    data_files = [f for f in files if f.extension not in SIDECAR_EXTENSIONS and not f.is_dir]
    sidecar_files = [f for f in files if f.extension in SIDECAR_EXTENSIONS and not f.is_dir]

    result: dict[str, list[FileRecord]] = defaultdict(list)

    for data_file in data_files:
        de = data_file.entities
        for scar in sidecar_files:
            se = scar.entities
            if _entities_compatible(de, se):
                result[data_file.path].append(scar)

    return dict(result)


def find_events_files(files: list[FileRecord]) -> dict[str, FileRecord]:
    """Return mapping: data file path → its events TSV (if any)."""
    events = [
        f for f in files
        if f.suffix == "events" and f.extension == ".tsv"
    ]
    data_files = [f for f in files if f.modality in {"eeg", "meg", "ieeg", "fmri"}]

    result: dict[str, FileRecord] = {}
    for data_file in data_files:
        de = data_file.entities
        for ev in events:
            se = ev.entities
            if (
                de.subject == se.subject
                and de.session == se.session
                and de.task == se.task
                and (de.run == se.run or se.run is None)
            ):
                result[data_file.path] = ev
                break

    return result


# ── Internal ──────────────────────────────────────────────────────────────────

def _entities_compatible(data_entities, sidecar_entities) -> bool:
    """True when all non-None sidecar entities match the data file's entities."""
    for field in ("subject", "session", "task", "run"):
        sc_val = getattr(sidecar_entities, field)
        da_val = getattr(data_entities, field)
        if sc_val is not None and da_val != sc_val:
            return False
    return True
