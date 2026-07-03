"""BIDS sidecar inheritance resolver.

BIDS sidecar files (``.json``, channels.tsv, events.tsv, …) follow a strict
inheritance hierarchy: more-specific files override more-general ones.  The
resolution order for a data file ``sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-01_eeg.set``
is (most-general → most-specific, later values win):

  1. ``<suffix>.json``                                  (dataset root)
  2. ``task-{task}_<suffix>.json``                      (dataset root, task-specific)
  3. ``sub-{sub}/<suffix>.json``                        (subject root)
  4. ``sub-{sub}/task-{task}_<suffix>.json``            (subject root, task-specific)
  5. ``sub-{sub}/ses-{ses}/<suffix>.json``              (session root)
  6. ``sub-{sub}/ses-{ses}/task-{task}_<suffix>.json``  (session root, task-specific)
  7. ``sub-{sub}/ses-{ses}/{datatype}/<suffix>.json``   (datatype dir, no entities)
  8. ``sub-{sub}/{datatype}/<suffix>.json``             (subject datatype dir, no session)
  9. ``sub-{sub}/ses-{ses}/{datatype}/sub-{sub}_ses-{ses}_<suffix>.json``
 10. ``sub-{sub}/ses-{ses}/{datatype}/sub-{sub}_ses-{ses}_task-{task}_<suffix>.json``
 11. ``sub-{sub}/ses-{ses}/{datatype}/sub-{sub}_ses-{ses}_task-{task}_run-{run}_<suffix>.json``

The ``SidecarResolver`` class resolves the chain from a manifest's file list,
then optionally merges JSON content from disk.  Path enumeration is pure
Python string manipulation — no I/O required for chain discovery.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from qortex.core.entities import FileRecord

log = logging.getLogger(__name__)

SIDECAR_EXTENSIONS = frozenset({".json", ".tsv", ".bvec", ".bval"})

SIDECAR_SUFFIXES = frozenset({
    "events", "channels", "electrodes", "coordsystem",
    "scans", "sessions", "participants",
})


# ── Resolver ──────────────────────────────────────────────────────────────────

class SidecarResolver:
    """Resolve BIDS sidecar inheritance chains from a flat file list.

    Build once from a manifest; then call ``.resolve()`` per data file for
    chain discovery, or ``.load_merged()`` to also read and merge JSON content.

    Parameters
    ----------
    files:
        All ``FileRecord`` objects from a ``Manifest``.  Directories are
        automatically excluded.
    """

    def __init__(self, files: list[FileRecord]) -> None:
        # Index all JSON sidecars by their path for O(1) lookup
        self._json_index: dict[str, FileRecord] = {}
        for f in files:
            if f.is_dir:
                continue
            if f.extension == ".json":
                self._json_index[f.path] = f

        # Fast lookup for events / channels TSVs by entity key
        self._events_index: dict[tuple, FileRecord] = {}
        self._channels_index: dict[tuple, FileRecord] = {}
        for f in files:
            if f.is_dir or f.extension != ".tsv":
                continue
            key = (f.subject, f.session, f.task, f.run)
            if f.suffix == "events":
                self._events_index[key] = f
            elif f.suffix == "channels":
                self._channels_index[key] = f

    def resolve(self, data_file: FileRecord) -> list[FileRecord]:
        """Return the ordered sidecar chain for *data_file*.

        Ordered most-general to most-specific so that ``.load_merged()``
        can simply iterate and let later values win on key collisions.

        Parameters
        ----------
        data_file:
            The primary BIDS data file whose sidecar context to resolve.

        Returns
        -------
        list[FileRecord]
            JSON sidecar files that apply, ordered from most-general to
            most-specific.  Empty list if none found.
        """
        suffix = data_file.suffix
        if not suffix:
            return []

        sub = data_file.subject
        ses = data_file.session
        task = data_file.task
        run = data_file.run
        datatype = data_file.datatype

        candidates = _build_candidate_paths(
            suffix=suffix,
            sub=sub,
            ses=ses,
            task=task,
            run=run,
            datatype=datatype,
        )

        # The sub/ses/task/run-only candidates above miss any other BIDS
        # entity present in the filename (acq-, dir-, echo-, part-, rec-,
        # ce-, ...) — extremely common for anatomical scans
        # ("acq-mprage_T1w.nii.gz"). Derive extra candidates directly from
        # the data file's own filename so a sidecar carrying the identical
        # entity set is always found, regardless of which entities it uses.
        extra_root, extra_sub_root, extra_same_dir = _entity_derived_candidates(
            data_file, suffix=suffix, sub=sub,
        )
        if extra_root:
            candidates.insert(0, extra_root)
        if extra_sub_root:
            candidates.append(extra_sub_root)
        if extra_same_dir:
            candidates.append(extra_same_dir)  # most specific: same directory, identical entities

        chain: list[FileRecord] = []
        seen: set[str] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if path in self._json_index:
                chain.append(self._json_index[path])
        return chain

    def load_merged(
        self,
        data_file: FileRecord,
        data_dir: Path,
    ) -> dict[str, Any]:
        """Resolve the sidecar chain and merge all JSON content.

        The merge follows BIDS: most-specific values win.  Returns an empty
        dict when no sidecars are present or none can be read.

        Parameters
        ----------
        data_file:
            The primary BIDS data file.
        data_dir:
            Root of the local BIDS download tree.

        Returns
        -------
        dict[str, Any]
            Merged sidecar parameters.
        """
        chain = self.resolve(data_file)
        merged: dict[str, Any] = {}
        for sidecar_fr in chain:
            local = data_dir / sidecar_fr.path
            if not local.exists():
                log.debug("Sidecar not found locally: %s", sidecar_fr.path)
                continue
            try:
                content = json.loads(local.read_text(encoding="utf-8"))
                if isinstance(content, dict):
                    merged.update(content)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to read sidecar %s: %s", sidecar_fr.path, exc)
        return merged

    def find_events(self, data_file: FileRecord) -> FileRecord | None:
        """Return the events TSV for *data_file* following BIDS entity matching.

        Tries exact match first (sub+ses+task+run), then degrades by removing
        run, then task, to handle missing entities in events filenames.
        """
        sub = data_file.subject
        ses = data_file.session
        task = data_file.task
        run = data_file.run

        return (
            self._events_index.get((sub, ses, task, run))
            or self._events_index.get((sub, ses, task, None))
            or self._events_index.get((sub, None, task, run))
            or self._events_index.get((sub, None, task, None))
        )

    def find_channels(self, data_file: FileRecord) -> FileRecord | None:
        """Return the channels TSV for *data_file*."""
        sub = data_file.subject
        ses = data_file.session
        task = data_file.task
        run = data_file.run

        return (
            self._channels_index.get((sub, ses, task, run))
            or self._channels_index.get((sub, ses, task, None))
            or self._channels_index.get((sub, None, task, run))
            or self._channels_index.get((sub, None, task, None))
        )


# ── Legacy helpers (kept for backward compatibility) ──────────────────────────

def group_sidecars(files: list[FileRecord]) -> dict[str, list[FileRecord]]:
    """Return mapping: data file path → list of applicable sidecar files.

    Uses ``SidecarResolver`` internally for accurate BIDS inheritance.
    """
    resolver = SidecarResolver(files)
    data_files = [
        f for f in files
        if not f.is_dir and f.extension not in SIDECAR_EXTENSIONS and f.modality
    ]
    result: dict[str, list[FileRecord]] = {}
    for df in data_files:
        chain = resolver.resolve(df)
        if chain:
            result[df.path] = chain
    return result


def find_events_files(files: list[FileRecord]) -> dict[str, FileRecord]:
    """Return mapping: data file path → its events TSV (if any)."""
    resolver = SidecarResolver(files)
    signal_modalities = {"eeg", "meg", "ieeg", "fmri", "fnirs"}
    result: dict[str, FileRecord] = {}
    for f in files:
        if f.is_dir or f.modality not in signal_modalities:
            continue
        ev = resolver.find_events(f)
        if ev:
            result[f.path] = ev
    return result


# ── Path generation ───────────────────────────────────────────────────────────

def _entity_derived_candidates(
    data_file: FileRecord, *, suffix: str, sub: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Derive sidecar candidates directly from the data file's own filename.

    ``_build_candidate_paths`` only reasons about sub/ses/task/run, so a
    sidecar carrying any other BIDS entity (``acq-``, ``dir-``, ``echo-``,
    ``part-``, ``ce-``, ``rec-``, ...) — e.g. ``sub-01_ses-mri_acq-mprage_T1w.json``
    next to ``sub-01_ses-mri_acq-mprage_T1w.nii.gz`` — is otherwise never found,
    silently degrading ``sidecar()`` to an empty dict.

    Returns ``(root_candidate, subject_root_candidate, same_dir_candidate)``,
    each ``None`` when not meaningfully different from the base case.
    """
    stem = data_file.filename
    if data_file.extension and stem.endswith(data_file.extension):
        stem = stem[: -len(data_file.extension)]
    suffix_token = f"_{suffix}"
    entity_string = stem[: -len(suffix_token)] if stem.endswith(suffix_token) else stem
    tokens = [t for t in entity_string.split("_") if t]
    if not tokens:
        return None, None, None

    # Same directory, identical entity set — the common "sidecar right next
    # to the data file" case, and the most specific candidate of the three.
    directory = str(PurePosixPath(data_file.path).parent)
    same_dir_name = f"{entity_string}_{suffix}.json"
    same_dir = same_dir_name if directory in ("", ".") else f"{directory}/{same_dir_name}"

    non_sub_ses = [t for t in tokens if not (t.startswith("sub-") or t.startswith("ses-"))]
    root = f"{'_'.join(non_sub_ses)}_{suffix}.json" if non_sub_ses else None

    sub_root = None
    if sub:
        sub_tokens = [t for t in tokens if not t.startswith("ses-")]
        if len(sub_tokens) > 1:  # more than just "sub-XX" alone
            sub_root = f"sub-{sub}/{'_'.join(sub_tokens)}_{suffix}.json"

    return root, sub_root, same_dir


def _build_candidate_paths(
    *,
    suffix: str,
    sub: str | None,
    ses: str | None,
    task: str | None,
    run: str | None,
    datatype: str | None,
) -> list[str]:
    """Return candidate sidecar JSON paths, most-general to most-specific.

    Each path in the returned list *might* exist in the manifest.  The caller
    checks which ones actually do and filters accordingly.
    """
    base_name = f"{suffix}.json"
    task_name = f"task-{task}_{suffix}.json" if task else None

    candidates: list[str] = []

    # ── Level 1: Dataset root ─────────────────────────────────────────────
    candidates.append(base_name)
    if task_name:
        candidates.append(task_name)

    # ── Level 2: Subject root ─────────────────────────────────────────────
    if sub:
        candidates.append(f"sub-{sub}/{base_name}")
        if task_name:
            candidates.append(f"sub-{sub}/{task_name}")

        # ── Level 3: Session root ─────────────────────────────────────────
        if ses:
            candidates.append(f"sub-{sub}/ses-{ses}/{base_name}")
            if task_name:
                candidates.append(f"sub-{sub}/ses-{ses}/{task_name}")

            # ── Level 4: Datatype dir, no entities ────────────────────────
            if datatype:
                candidates.append(f"sub-{sub}/ses-{ses}/{datatype}/{base_name}")
                if task_name:
                    candidates.append(f"sub-{sub}/ses-{ses}/{datatype}/{task_name}")

                # ── Level 5: Subject+session entities ─────────────────────
                stem_sub_ses = f"sub-{sub}_ses-{ses}_{suffix}.json"
                candidates.append(f"sub-{sub}/ses-{ses}/{datatype}/{stem_sub_ses}")

                # ── Level 6: With task ────────────────────────────────────
                if task:
                    stem_task = f"sub-{sub}_ses-{ses}_task-{task}_{suffix}.json"
                    candidates.append(f"sub-{sub}/ses-{ses}/{datatype}/{stem_task}")

                    # ── Level 7: With run ─────────────────────────────────
                    if run:
                        stem_run = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{suffix}.json"
                        candidates.append(f"sub-{sub}/ses-{ses}/{datatype}/{stem_run}")

        else:
            # No session: subject datatype dir
            if datatype:
                candidates.append(f"sub-{sub}/{datatype}/{base_name}")
                if task_name:
                    candidates.append(f"sub-{sub}/{datatype}/{task_name}")

                stem_sub = f"sub-{sub}_{suffix}.json"
                candidates.append(f"sub-{sub}/{datatype}/{stem_sub}")

                if task:
                    stem_task = f"sub-{sub}_task-{task}_{suffix}.json"
                    candidates.append(f"sub-{sub}/{datatype}/{stem_task}")

                    if run:
                        stem_run = f"sub-{sub}_task-{task}_run-{run}_{suffix}.json"
                        candidates.append(f"sub-{sub}/{datatype}/{stem_run}")

    return candidates
