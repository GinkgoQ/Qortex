"""BIDS dataset source adapter.

Probes a locally downloaded BIDS directory and presents it as a SourceProfile.
Reads participants.tsv, dataset_description.json, and subject-level manifest
to determine what the source can provide — without loading any signal data.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from qortex.neuroai.contracts import (
    AxisConvention,
    ChannelSpec,
    EvidenceStatus,
    Modality,
    QortexTimeSeries,
    QortexVolume,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.sources.local import LocalFileAdapter
from qortex.neuroai.spec import SourceSpec, WindowSpec
from qortex.core.exceptions import SourceAdapterError

log = logging.getLogger(__name__)

_BIDS_MODALITY_MAP = {
    "eeg": "eeg", "meg": "meg", "ieeg": "ieeg", "fnirs": "fnirs",
    "anat": "mri", "func": "fmri", "dwi": "dwi", "pet": "pet",
    "fmap": "mri",
}
_SIGNAL_EXTS = {".edf", ".bdf", ".fif", ".set", ".vhdr"}
_VOLUME_EXTS = {".nii", ".gz"}


class BIDSSourceAdapter(SourceAdapter):
    """Source adapter for locally downloaded BIDS datasets.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="bids"`` and ``path=<bids_root>``.
    window_spec:
        Optional windowing for signal streaming.
    channel_names:
        Optional subset of channel names.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        window_spec: WindowSpec | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        if not spec.path:
            raise ValueError("BIDSSourceAdapter requires spec.path (BIDS root directory)")
        self._root = Path(spec.path).expanduser().resolve()
        if not self._root.is_dir():
            raise NotADirectoryError(f"BIDS root not found: {self._root}")
        self._spec = spec
        self._window_spec = window_spec
        self._channel_names = channel_names
        self._target_modality = spec.modality
        self._target_suffix = spec.suffix
        self._target_subjects = spec.subjects
        self._max_profile_files = int(spec.extra.get("max_profile_files", 64))

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        desc = self._read_dataset_description()
        subjects = self._discover_subjects()
        modalities = self._discover_modalities(subjects)
        warnings: list[WarningItem] = []

        target_files = self._collect_target_files()
        profile_files = target_files[: self._max_profile_files] if self._max_profile_files > 0 else target_files
        recording_profiles: list[SourceProfile] = []
        for file_path in profile_files:
            try:
                profile = LocalFileAdapter(
                    SourceSpec(type="local_file", path=str(file_path)),
                    window_spec=self._window_spec,
                    channel_names=self._channel_names,
                ).probe()
                profile.source_id = f"bids:{file_path.relative_to(self._root).as_posix()}"
                profile.source_type = "bids_recording"
                profile.extra = dict(profile.extra or {})
                profile.extra.update(_parse_bids_entities(file_path.name))
                profile.extra["relative_path"] = file_path.relative_to(self._root).as_posix()
                recording_profiles.append(profile)
            except Exception as exc:
                warnings.append(WarningItem(
                    code="BIDS_RECORDING_PROBE_FAILED",
                    message=f"Cannot probe {file_path.relative_to(self._root)}: {exc}",
                    severity="warning",
                ))

        consistency = _build_consistency_report(recording_profiles)
        representative = recording_profiles[0] if recording_profiles else None

        primary_modality = self._target_modality or (modalities[0] if modalities else None)
        abstraction = getattr(representative, "abstraction", None)

        return SourceProfile(
            source_id=self.source_id,
            source_type="bids",
            path=str(self._root),
            modality=primary_modality,
            abstraction=abstraction,
            n_subjects=len(subjects),
            n_channels=getattr(representative, "n_channels", None),
            sampling_rate_hz=getattr(representative, "sampling_rate_hz", None),
            channel_names=list(getattr(representative, "channel_names", []) or []),
            available_suffixes=list(modalities),
            spatial_shape=getattr(representative, "spatial_shape", None),
            voxel_sizes_mm=getattr(representative, "voxel_sizes_mm", None),
            n_volumes=getattr(representative, "n_volumes", None),
            tr_s=getattr(representative, "tr_s", None),
            axis_convention=getattr(representative, "axis_convention", None)
            or AxisConvention.channels_time,
            evidence_status=(
                EvidenceStatus.confirmed
                if recording_profiles and not any(v.get("status") == "variable" for v in consistency.values())
                else EvidenceStatus.inferred
            ),
            warnings=warnings,
            extra={
                "name": desc.get("Name", ""),
                "bids_version": desc.get("BIDSVersion", ""),
                "n_sessions": self._count_sessions(subjects),
                "n_recordings_total": len(target_files),
                "n_recordings_profiled": len(recording_profiles),
                "recording_profiles": [_profile_summary(p) for p in recording_profiles],
                "consistency_report": consistency,
            },
        )

    def read_batch(self) -> list[QortexData]:
        files = self._collect_target_files()
        results: list[QortexData] = []
        errors: list[str] = []
        for f in files:
            adapter = LocalFileAdapter(
                SourceSpec(type="local_file", path=str(f)),
                window_spec=self._window_spec,
                channel_names=self._channel_names,
            )
            try:
                results.extend(adapter.read_batch())
            except Exception as exc:
                errors.append(f"{f.relative_to(self._root)}: {exc}")
        if errors and not results:
            raise SourceAdapterError(
                "No BIDS recordings could be loaded. " + "; ".join(errors[:5]),
                source_type="bids",
                path=str(self._root),
            )
        if errors:
            log.warning("BIDSSourceAdapter: %d recording(s) failed to load", len(errors))
        return results

    def stream(self) -> Iterator[QortexData]:
        for f in self._collect_target_files():
            adapter = LocalFileAdapter(
                SourceSpec(type="local_file", path=str(f)),
                window_spec=self._window_spec,
                channel_names=self._channel_names,
            )
            try:
                yield from adapter.stream()
            except Exception as exc:
                raise SourceAdapterError(
                    f"Stream error for {f.relative_to(self._root)}: {exc}",
                    source_type="bids",
                    path=str(f),
                ) from exc

    @property
    def source_id(self) -> str:
        return f"bids:{self._root.name}"

    # ── BIDS discovery ────────────────────────────────────────────────────────

    def _discover_subjects(self) -> list[str]:
        subs = sorted(p.name for p in self._root.iterdir()
                      if p.is_dir() and p.name.startswith("sub-"))
        if self._target_subjects:
            keep = {f"sub-{s}" if not s.startswith("sub-") else s for s in self._target_subjects}
            subs = [s for s in subs if s in keep]
        return subs

    def _discover_modalities(self, subjects: list[str]) -> list[str]:
        mods: set[str] = set()
        for sub in subjects:
            sub_dir = self._root / sub
            for d in sub_dir.iterdir():
                if d.is_dir() and not d.name.startswith("ses-"):
                    mods.add(d.name)
                elif d.is_dir() and d.name.startswith("ses-"):
                    for dd in d.iterdir():
                        if dd.is_dir():
                            mods.add(dd.name)
        return sorted(mods)

    def _collect_target_files(self) -> list[Path]:
        subjects = self._discover_subjects()
        target_exts: set[str] = set()

        if self._target_modality in ("eeg", "meg", "ieeg", "fnirs"):
            target_exts = _SIGNAL_EXTS
        elif self._target_modality in ("mri", "fmri", "dwi", "pet", "anat", "func"):
            target_exts = {".nii", ".nii.gz"}

        result: list[Path] = []
        for sub in subjects:
            sub_dir = self._root / sub
            # Flat BIDS layout
            search_roots = [sub_dir] + sorted(sub_dir.glob("ses-*"))
            for search_root in search_roots:
                for folder in search_root.iterdir():
                    if not folder.is_dir():
                        continue
                    mod = folder.name
                    if self._target_modality and mod != self._target_modality:
                        continue
                    for f in sorted(folder.iterdir()):
                        if f.is_file():
                            ext = ".nii.gz" if f.name.endswith(".nii.gz") else f.suffix
                            if not target_exts or ext in target_exts:
                                if self._target_suffix:
                                    stem = f.name
                                    for chk in (".nii.gz", ".nii", ".edf", ".bdf", ".fif"):
                                        stem = stem.removesuffix(chk)
                                    parts = stem.rsplit("_", 1)
                                    suffix = parts[-1] if parts else stem
                                    if suffix != self._target_suffix:
                                        continue
                                result.append(f)
        return result

    def _read_dataset_description(self) -> dict:
        desc_path = self._root / "dataset_description.json"
        if desc_path.is_file():
            try:
                return json.loads(desc_path.read_text())
            except Exception:
                pass
        return {}

    def _read_sidecar(self, data_file: Path) -> dict:
        stem = data_file.name
        for ext in (".nii.gz", ".nii", ".edf", ".bdf", ".fif", ".set"):
            stem = stem.removesuffix(ext)
        sidecar = data_file.parent / f"{stem}.json"
        if sidecar.is_file():
            try:
                return json.loads(sidecar.read_text())
            except Exception:
                pass
        return {}

    def _count_sessions(self, subjects: list[str]) -> int:
        sessions: set[str] = set()
        for sub in subjects:
            sub_dir = self._root / sub
            for d in sub_dir.iterdir():
                if d.is_dir() and d.name.startswith("ses-"):
                    sessions.add(d.name)
        return len(sessions)
