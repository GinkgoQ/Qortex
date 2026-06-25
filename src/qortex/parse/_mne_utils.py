"""Shared MNE/MNE-BIDS utilities for signal loaders.

All functions in this module guard their imports so that the module itself
can be imported without MNE installed; the ImportError is raised at call time
with a user-friendly message.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qortex.core.entities import FileRecord

log = logging.getLogger(__name__)


# ── Dependency guards ─────────────────────────────────────────────────────────

def require_mne(extra: str = "eeg"):
    try:
        import mne
        return mne
    except ImportError:
        raise ImportError(
            f"Signal loading requires MNE-Python: pip install 'qortex[{extra}]'"
        )


def require_mne_bids(extra: str = "eeg"):
    try:
        import mne_bids
        return mne_bids
    except ImportError:
        raise ImportError(
            f"BIDS-aware loading requires MNE-BIDS: pip install 'qortex[{extra}]'"
        )


# ── BIDS root detection ───────────────────────────────────────────────────────

def find_bids_root(local_path: Path) -> Path | None:
    """Walk up the directory tree to find the BIDS root (contains dataset_description.json).

    Returns None if not found within 8 levels — signals a non-BIDS tree.
    """
    candidate = local_path.parent
    for _ in range(8):
        if (candidate / "dataset_description.json").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return None


def bids_root_from_manifest_path(file_path: str, local_path: Path) -> Path | None:
    """Derive BIDS root from the manifest's relative path + the absolute local path.

    If file_path is 'sub-01/eeg/sub-01_task-rest_eeg.set' and local_path is
    '/cache/datasets/ds001/data/sub-01/eeg/sub-01_task-rest_eeg.set', the root
    is '/cache/datasets/ds001/data'.

    This is more reliable than walking the FS when the cache structure is known.
    """
    parts = [p for p in file_path.split("/") if p]
    depth = len(parts) - 1  # number of directories above the file
    if depth < 0:
        return None
    try:
        return local_path.parents[depth]
    except IndexError:
        return None


def resolve_bids_root(file: FileRecord, local_path: Path) -> Path | None:
    """Resolve BIDS root using manifest path first, then filesystem walk as fallback."""
    root = bids_root_from_manifest_path(file.path, local_path)
    if root is not None and (root / "dataset_description.json").exists():
        return root
    return find_bids_root(local_path)


# ── Channel table parsing ─────────────────────────────────────────────────────

def load_channels_tsv(bids_root: Path, file: FileRecord) -> dict[str, dict]:
    """Parse *_channels.tsv to extract bad-channel and type annotations.

    Returns a dict mapping channel name → {status, type, units, ...}.
    """
    # Build candidate paths in inheritance order (most specific first)
    ents = file.entities
    candidates = []
    base = bids_root

    for sub_dir in [
        f"sub-{ents.subject}/ses-{ents.session}" if ents.session else f"sub-{ents.subject}",
        f"sub-{ents.subject}",
        ".",
    ]:
        for task_part in [
            f"sub-{ents.subject}{'_ses-' + ents.session if ents.session else ''}"
            f"{'_task-' + ents.task if ents.task else ''}"
            f"{'_run-' + ents.run if ents.run else ''}_channels.tsv",
            f"sub-{ents.subject}{'_ses-' + ents.session if ents.session else ''}"
            f"{'_task-' + ents.task if ents.task else ''}_channels.tsv",
            f"sub-{ents.subject}{'_ses-' + ents.session if ents.session else ''}_channels.tsv",
        ]:
            p = base / sub_dir / file.datatype / task_part
            if p not in candidates:
                candidates.append(p)

    for path in candidates:
        if path.exists():
            try:
                import polars as pl
                df = pl.read_csv(str(path), separator="\t", null_values=["n/a", "N/A"])
                result: dict[str, dict] = {}
                for row in df.iter_rows(named=True):
                    name = row.get("name", "")
                    if name:
                        result[name] = {k: v for k, v in row.items() if k != "name"}
                return result
            except Exception:
                pass
    return {}


# ── JSON sidecar loading ──────────────────────────────────────────────────────

def load_json_sidecar(local_path: Path) -> dict:
    """Load the JSON sidecar for a data file (BIDS sidecar inheritance not applied).

    Looks for <stem>.json next to the data file.
    """
    json_path = local_path.with_suffix("").with_suffix(".json")
    if not json_path.exists():
        # Handle compound extensions like .nii.gz
        stem = local_path.name
        for ext in (".nii.gz", ".tar.gz"):
            if stem.endswith(ext):
                json_path = local_path.parent / (stem[: -len(ext)] + ".json")
                break
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ── Signal record builder ─────────────────────────────────────────────────────

def _build_channel_info(raw, channels_meta: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    """Return (ch_names, ch_types, bad_channels) using channels.tsv annotations."""
    mne = require_mne()
    bad_channels: list[str] = []
    ch_types: list[str] = []

    for i, ch_name in enumerate(raw.ch_names):
        ch_type = mne.channel_type(raw.info, i)
        ch_types.append(ch_type)
        meta = channels_meta.get(ch_name, {})
        if str(meta.get("status", "good")).lower() == "bad":
            bad_channels.append(ch_name)

    return raw.ch_names, ch_types, bad_channels


def read_raw_with_bids_fallback(
    file: FileRecord,
    local_path: Path,
    datatype: str,
    suffix: str,
    preload: bool,
    extra_kwargs: dict,
) -> tuple:
    """Attempt MNE-BIDS read with full sidecar inheritance; fall back to plain MNE.

    Returns (raw, bids_root, channels_meta).
    """
    mne = require_mne()
    mne_bids = require_mne_bids()
    ents = file.entities

    bids_root = resolve_bids_root(file, local_path)
    channels_meta: dict[str, dict] = {}

    raw = None
    if bids_root is not None:
        channels_meta = load_channels_tsv(bids_root, file)
        try:
            bids_path = mne_bids.BIDSPath(
                subject=ents.subject,
                session=ents.session,
                task=ents.task,
                run=ents.run,
                datatype=datatype,
                root=bids_root,
                suffix=suffix,
                extension=file.extension,
            )
            extra = dict(extra_kwargs)
            extra.pop("preload", None)
            raw = mne_bids.read_raw_bids(bids_path, preload=preload, **extra)
            log.debug("MNE-BIDS read succeeded: %s", local_path)
        except Exception as exc:
            log.debug("MNE-BIDS read failed (%s), falling back to plain MNE: %s", exc, local_path)
            raw = None

    if raw is None:
        kw = dict(extra_kwargs)
        kw.pop("preload", None)
        raw = mne.io.read_raw(str(local_path), preload=preload, **kw)

    # Apply bad-channel annotations from channels.tsv
    if channels_meta and raw is not None:
        bads_from_tsv = [
            ch for ch, meta in channels_meta.items()
            if str(meta.get("status", "good")).lower() == "bad"
            and ch in raw.ch_names
        ]
        if bads_from_tsv:
            raw.info["bads"] = list(set(raw.info.get("bads", []) + bads_from_tsv))

    return raw, bids_root, channels_meta


# ── Numpy extraction ──────────────────────────────────────────────────────────

def raw_to_numpy(raw, picks=None):
    """Extract data array from MNE Raw object.

    Returns shape (n_channels, n_times) in float64.
    If preload=False, triggers a load.
    """
    import numpy as np
    if not raw.preload:
        raw.load_data()
    if picks is not None:
        data, _ = raw[picks]
    else:
        data, _ = raw[:]
    return data.astype(np.float64)
