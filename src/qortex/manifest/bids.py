"""BIDS entity parser — inline implementation, no PyBIDS dependency.

PyBIDS is reserved for local dataset indexing (the ``lake`` module).
This parser handles the file-tree metadata layer efficiently.

BIDS filename format:
    [key-value_]...[key-value_]suffix.extension

Examples:
    sub-01_ses-meg_task-facerecognition_run-01_meg.fif
    sub-01_T1w.nii.gz
    participants.tsv
    dataset_description.json
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from qortex.core.entities import BIDSEntities

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical BIDS entity keys (short form as used in filenames)
BIDS_ENTITY_KEYS = {
    "sub", "ses", "task", "acq", "ce", "dir", "rec",
    "run", "mod", "echo", "flip", "inv", "mt", "part",
    "proc", "hemi", "space", "split", "res", "den",
    "sample", "stain", "chunk", "label", "desc",
}

# BIDS datatype directories
BIDS_DATATYPES = frozenset({
    "anat", "func", "dwi", "fmap", "beh",
    "eeg", "meg", "ieeg", "fnirs", "pet",
    "perf", "micr", "nirs",
})

# Compound extensions (order matters — longest first)
_COMPOUND_EXTENSIONS = (
    ".nii.gz", ".tar.gz", ".tsv.gz", ".json.gz", ".surf.gii",
)

# Mapping from entity short key to BIDSEntities field name
_ENTITY_FIELD_MAP: dict[str, str] = {
    "sub": "subject",
    "ses": "session",
    "task": "task",
    "run": "run",
    "acq": "acquisition",
    "dir": "direction",
    "space": "space",
    "res": "resolution",
    "echo": "echo",
    "part": "part",
    "hemi": "hemisphere",
    "den": "density",
    "proc": "processing",
    "split": "split",
}

# Unified modality labels derived from BIDS datatypes + suffix
_DATATYPE_TO_MODALITY: dict[str, str] = {
    "eeg": "eeg",
    "meg": "meg",
    "ieeg": "ieeg",
    "fnirs": "fnirs",
    "nirs": "fnirs",
    "anat": "mri",
    "func": "fmri",
    "dwi": "dwi",
    "perf": "fmri",
    "pet": "pet",
    "beh": "behavior",
    "fmap": "fmap",
}

# Suffixes that override the datatype-based modality
_SUFFIX_MODALITY_OVERRIDE: dict[str, str] = {
    "eeg": "eeg",
    "meg": "meg",
    "ieeg": "ieeg",
    "nirs": "fnirs",
    "bold": "fmri",
    "T1w": "mri",
    "T2w": "mri",
    "FLAIR": "mri",
    "T1rho": "mri",
    "dwi": "dwi",
    "pet": "pet",
    "events": "behavior",
    "physio": "behavior",
    "stim": "behavior",
}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict[str, str]:
    """Parse a BIDS filename into a flat entity dict.

    Returns a dict with keys from ``_ENTITY_FIELD_MAP`` plus:
        ``suffix``    — BIDS suffix (e.g. "bold", "T1w", "eeg")
        ``extension`` — file extension (e.g. ".nii.gz", ".fif")
        ``extra``     — dict of non-standard entity keys (rare)
    """
    ext = _extract_extension(filename)
    stem = filename[: len(filename) - len(ext)]  # strip extension only

    parts = stem.split("_")
    result: dict[str, str] = {"extension": ext}
    extra: dict[str, str] = {}

    # Last part that contains no hyphen is the suffix
    if parts and "-" not in parts[-1]:
        result["suffix"] = parts.pop()
    else:
        result["suffix"] = ""

    for part in parts:
        if "-" in part:
            key, _, value = part.partition("-")
            field = _ENTITY_FIELD_MAP.get(key)
            if field:
                result[field] = value
            else:
                extra[key] = value
        # bare parts without '-' are non-standard; ignore

    result["_extra"] = extra  # type: ignore[assignment]
    return result


def parse_entities(filename: str) -> BIDSEntities:
    """Parse a BIDS filename and return a typed ``BIDSEntities`` object."""
    raw = parse_filename(filename)
    extra = raw.pop("_extra", {})
    # Remove keys that don't belong to BIDSEntities fields
    raw.pop("suffix", None)
    raw.pop("extension", None)
    return BIDSEntities(**{k: v for k, v in raw.items() if k in BIDSEntities.model_fields},
                        extra=extra)


def extract_datatype(path: str) -> str | None:
    """Return the BIDS datatype directory from a relative path."""
    parts = PurePosixPath(path).parts
    for part in parts:
        if part in BIDS_DATATYPES:
            return part
    return None


def infer_modality(datatype: str | None, suffix: str | None) -> str | None:
    """Return a unified modality label from datatype + suffix."""
    if suffix and suffix in _SUFFIX_MODALITY_OVERRIDE:
        return _SUFFIX_MODALITY_OVERRIDE[suffix]
    if datatype:
        return _DATATYPE_TO_MODALITY.get(datatype)
    return None


def sidecar_group_key(path: str) -> str:
    """Produce a stable hash identifying the sidecar inheritance chain.

    Two files share a sidecar group if their BIDS inheritance contexts are
    identical (subject + session + task + run + datatype, ignoring run for
    multi-run datasets where the .json is shared).
    """
    entities = parse_filename(PurePosixPath(path).name)
    key = ":".join([
        entities.get("subject", ""),
        entities.get("session", ""),
        entities.get("task", ""),
        entities.get("datatype", extract_datatype(path) or ""),
    ])
    return hashlib.md5(key.encode()).hexdigest()[:8]


# ── Internal ──────────────────────────────────────────────────────────────────

def _extract_extension(filename: str) -> str:
    """Extract extension, handling compound forms like ``.nii.gz``."""
    for ext in _COMPOUND_EXTENSIONS:
        if filename.endswith(ext):
            return ext
    return Path(filename).suffix or ""
