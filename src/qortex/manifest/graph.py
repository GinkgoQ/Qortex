"""Semantic graph over a BIDS/OpenNeuro manifest.

The graph is the bridge between file-level manifests and user-level work:
selection, readiness, download planning, conversion, and training should reason
about logical recordings and companion files, not isolated paths.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from qortex.core.entities import (
    CompanionSet,
    FileRecord,
    LogicalRecording,
    Manifest,
)

SIDECAR_EXTENSIONS = frozenset({".json", ".tsv", ".bvec", ".bval"})
PRIMARY_MODALITIES = frozenset({
    "eeg",
    "meg",
    "ieeg",
    "fnirs",
    "mri",
    "fmri",
    "dwi",
    "pet",
})
LABEL_COLUMNS = frozenset({
    "trial_type",
    "event_type",
    "stim_type",
    "condition",
    "category",
    "label",
})


class ManifestGraph:
    """Build logical recordings and companion-file closures from a Manifest."""

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        self.files = [f for f in manifest.files if not f.is_dir]
        self.by_path = {f.path: f for f in self.files}
        self._essentials = [f for f in self.files if f.is_essential]
        self._recordings: list[LogicalRecording] | None = None

    def recordings(self) -> list[LogicalRecording]:
        """Return semantic primary-data units with companion files attached."""
        if self._recordings is not None:
            return self._recordings

        recordings: list[LogicalRecording] = []
        for primary in self.files:
            if not self._is_primary(primary):
                continue
            companions = self.companions_for(primary)
            issues = self._recording_issues(primary, companions)
            recording = LogicalRecording(
                id=_recording_id(primary),
                primary=primary,
                companions=companions,
                modality=primary.modality,
                datatype=primary.datatype,
                subject=primary.subject,
                session=primary.session,
                task=primary.task,
                run=primary.run,
                has_events=companions.events is not None,
                has_label_candidates=_events_look_label_candidate(companions.events),
                has_labels=False,
                downloadable=bool(primary.urls),
                loadable=bool(primary.urls),
                estimated_bytes=sum(f.size or 0 for f in [primary, *companions.files]),
                issues=issues,
            )
            recordings.append(recording)
        self._recordings = recordings
        return recordings

    def companions_for(self, primary: FileRecord) -> CompanionSet:
        """Return likely required companions for *primary*.

        This is intentionally structural: it uses BIDS entities, suffixes, and
        sidecar role semantics rather than substring routing.
        """
        sidecars = [
            f for f in self.files
            if f.extension == ".json"
            and self._json_sidecar_applies(primary, f)
        ]
        events = self._best_role(primary, suffix="events", extension=".tsv")
        channels = self._best_role(primary, suffix="channels", extension=".tsv")
        electrodes = self._best_role(primary, suffix="electrodes", extension=".tsv")
        coordsystem = self._best_role(primary, suffix="coordsystem", extension=".json")
        scans = self._best_role(primary, suffix="scans", extension=".tsv")
        participants = self._top_level("participants.tsv")
        dataset_description = self._top_level("dataset_description.json")
        bvec = self._same_stem(primary, ".bvec") if primary.datatype == "dwi" else None
        bval = self._same_stem(primary, ".bval") if primary.datatype == "dwi" else None

        excluded_paths = {
            f.path for f in (participants, dataset_description) if f is not None
        }
        extra = [f for f in self._essentials if f.path not in excluded_paths]
        return CompanionSet(
            primary=primary,
            sidecars=sidecars,
            events=events,
            channels=channels,
            electrodes=electrodes,
            coordsystem=coordsystem,
            scans=scans,
            bvec=bvec,
            bval=bval,
            participants=participants,
            dataset_description=dataset_description,
            extra=extra,
        )

    def companion_closure(self, files: list[FileRecord]) -> list[FileRecord]:
        """Expand primary files to include required companion files."""
        selected: dict[str, FileRecord] = {f.path: f for f in files}
        primary_paths = {r.primary.path: r for r in self.recordings()}
        for file in files:
            recording = primary_paths.get(file.path)
            if recording is None:
                continue
            for companion in recording.companions.files:
                selected.setdefault(companion.path, companion)
        return list(selected.values())

    def recording_for_path(self, path: str) -> LogicalRecording | None:
        for recording in self.recordings():
            if recording.primary.path == path:
                return recording
        return None

    def recording_requiring_path(self, path: str) -> LogicalRecording | None:
        """Return the recording that caused *path* to be included."""
        for recording in self.recordings():
            if any(file.path == path for file in recording.files):
                return recording
        return None

    def _is_primary(self, file: FileRecord) -> bool:
        if file.is_essential or file.extension in SIDECAR_EXTENSIONS:
            return False
        if file.modality not in PRIMARY_MODALITIES:
            return False
        if file.path.startswith("derivatives/"):
            return True
        return file.datatype is not None

    def _top_level(self, filename: str) -> FileRecord | None:
        return self.by_path.get(filename)

    def _best_role(
        self,
        primary: FileRecord,
        *,
        suffix: str,
        extension: str,
    ) -> FileRecord | None:
        candidates = [
            f for f in self.files
            if f.suffix == suffix
            and f.extension == extension
            and self._sidecar_applies(primary, f)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda f: _specificity(f), reverse=True)
        return candidates[0]

    def _same_stem(self, primary: FileRecord, extension: str) -> FileRecord | None:
        stem = primary.filename.removesuffix(primary.extension)
        parent = str(PurePosixPath(primary.path).parent)
        for file in self.files:
            if file.extension == extension and file.filename == f"{stem}{extension}":
                if str(PurePosixPath(file.path).parent) == parent:
                    return file
        return None

    def _sidecar_applies(self, primary: FileRecord, sidecar: FileRecord) -> bool:
        pe = primary.entities
        se = sidecar.entities
        for field in ("subject", "session", "task", "run", "acquisition", "direction"):
            value = getattr(se, field)
            if value is not None and getattr(pe, field) != value:
                return False
        if sidecar.datatype is not None and primary.datatype != sidecar.datatype:
            return False
        return True

    def _json_sidecar_applies(self, primary: FileRecord, sidecar: FileRecord) -> bool:
        if sidecar.is_essential:
            return False
        if not self._sidecar_applies(primary, sidecar):
            return False
        if not _path_can_inherit(primary.path, sidecar.path):
            return False
        if sidecar.suffix is None:
            return False
        return sidecar.suffix in {
            primary.suffix,
            primary.datatype,
            primary.modality,
        }

    def _recording_issues(
        self,
        primary: FileRecord,
        companions: CompanionSet,
    ) -> list[str]:
        issues: list[str] = []
        if not primary.urls:
            issues.append("No download URL available.")
        if primary.modality in {"eeg", "meg", "ieeg", "fnirs", "fmri"}:
            if primary.task and companions.events is None:
                issues.append("Task recording has no matching events file.")
        if primary.modality in {"eeg", "meg", "ieeg", "fnirs"}:
            if companions.channels is None:
                issues.append("No matching channels.tsv file found.")
        if primary.datatype == "dwi":
            if companions.bvec is None or companions.bval is None:
                issues.append("DWI file is missing bvec/bval companions.")
        return issues


def _recording_id(primary: FileRecord) -> str:
    parts = [
        primary.modality or "unknown",
        primary.subject or "nosub",
        primary.session or "noses",
        primary.task or "notask",
        primary.run or "norun",
        primary.path,
    ]
    return "|".join(parts)


def _specificity(file: FileRecord) -> int:
    return sum(
        1
        for value in (
            file.subject,
            file.session,
            file.task,
            file.run,
            file.entities.acquisition,
            file.entities.direction,
        )
        if value is not None
    )


def _events_look_label_candidate(events: FileRecord | None) -> bool:
    if events is None:
        return False
    # A manifest-only graph cannot read the TSV. Treat events files as label
    # candidates; the readiness layer upgrades/downgrades this when local data
    # are available and the actual columns can be inspected.
    return events.suffix == "events"


def _path_can_inherit(primary_path: str, sidecar_path: str) -> bool:
    primary_parent = PurePosixPath(primary_path).parent
    sidecar_parent = PurePosixPath(sidecar_path).parent
    if str(sidecar_parent) == ".":
        return True
    if sidecar_parent == primary_parent:
        return True
    return sidecar_parent in primary_parent.parents
