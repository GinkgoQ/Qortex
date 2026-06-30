"""BIDS dataset source adapter.

Probes a locally downloaded BIDS directory and presents it as a SourceProfile.
Reads participants.tsv, dataset_description.json, and subject-level manifest
to determine what the source can provide — without loading any signal data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from qortex.core.exceptions import SourceAdapterError
from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.sources.local import LocalFileAdapter
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)

_BIDS_MODALITY_MAP = {
    "eeg": "eeg", "meg": "meg", "ieeg": "ieeg", "fnirs": "fnirs",
    "anat": "mri", "func": "fmri", "dwi": "dwi", "pet": "pet",
    "fmap": "mri",
}
_SIGNAL_EXTS = {".edf", ".bdf", ".fif", ".set", ".vhdr"}
_VOLUME_EXTS = {".nii", ".nii.gz"}
_SUPPORTED_DATA_EXTS = _SIGNAL_EXTS | _VOLUME_EXTS


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
        profiled_all = bool(target_files) and len(recording_profiles) == len(target_files)
        internally_constant = recording_profiles and not any(
            v.get("status") == "variable" for v in consistency.values()
        )

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
                if internally_constant and profiled_all
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
        else:
            target_exts = set(_SUPPORTED_DATA_EXTS)

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
                    if self._target_modality and not _folder_matches_modality(
                        mod, self._target_modality
                    ):
                        continue
                    for f in sorted(folder.iterdir()):
                        if f.is_file():
                            ext = ".nii.gz" if f.name.endswith(".nii.gz") else f.suffix
                            if ext not in target_exts:
                                continue
                            entities = _parse_bids_entities(f.name)
                            if self._target_suffix and entities.get("suffix") != self._target_suffix:
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


def _folder_matches_modality(folder: str, target: str) -> bool:
    """Return True when a BIDS datatype folder satisfies a requested modality."""
    folder_norm = folder.strip().lower()
    target_norm = target.strip().lower()
    if folder_norm == target_norm:
        return True
    return _BIDS_MODALITY_MAP.get(folder_norm) == target_norm


def _parse_bids_entities(filename: str) -> dict[str, str]:
    """Parse BIDS entity fields from a data filename.

    The parser follows the structural BIDS ``key-value_key-value_suffix.ext``
    convention. It does not infer semantics from free text or route by filename
    fragments; it only records explicit entity fields and the terminal suffix.
    """
    stem = filename
    for ext in (".nii.gz", ".nii", ".edf", ".bdf", ".fif", ".set", ".vhdr"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    entity_aliases = {
        "sub": "subject",
        "ses": "session",
        "task": "task",
        "run": "run",
        "acq": "acquisition",
        "rec": "recording",
        "space": "space",
        "desc": "description",
    }
    entities: dict[str, str] = {}
    suffix: str | None = None
    for part in stem.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            if key and value:
                entities[entity_aliases.get(key, key)] = value
        elif part:
            suffix = part
    if suffix:
        entities["suffix"] = suffix
    return entities


def _profile_summary(profile: SourceProfile) -> dict[str, Any]:
    """Compact, serialisable summary for a profiled BIDS recording."""
    extra = dict(profile.extra or {})
    summary: dict[str, Any] = {
        "source_id": profile.source_id,
        "source_type": profile.source_type,
        "path": profile.path,
        "modality": _json_safe(profile.modality),
        "abstraction": profile.abstraction,
        "n_channels": profile.n_channels,
        "sampling_rate_hz": profile.sampling_rate_hz,
        "duration_s": profile.duration_s,
        "spatial_shape": _json_safe(profile.spatial_shape),
        "voxel_sizes_mm": _json_safe(profile.voxel_sizes_mm),
        "n_volumes": profile.n_volumes,
        "tr_s": profile.tr_s,
        "dtype": profile.dtype,
        "axis_convention": _json_safe(profile.axis_convention),
        "evidence_status": _json_safe(profile.evidence_status),
        "relative_path": extra.get("relative_path"),
        "entities": {
            key: value
            for key, value in extra.items()
            if key not in {"relative_path"} and isinstance(value, (str, int, float, bool))
        },
    }
    if profile.channel_names:
        summary["channel_names"] = list(profile.channel_names[:128])
        summary["n_channel_names_reported"] = len(profile.channel_names)
    return summary


def _build_consistency_report(profiles: list[SourceProfile]) -> dict[str, dict[str, Any]]:
    """Summarise whether profiled BIDS recordings agree on core header fields."""
    fields = {
        "modality": lambda p: p.modality,
        "abstraction": lambda p: p.abstraction,
        "n_channels": lambda p: p.n_channels,
        "sampling_rate_hz": lambda p: p.sampling_rate_hz,
        "channel_set": lambda p: tuple(p.channel_names or []),
        "spatial_shape": lambda p: p.spatial_shape,
        "voxel_sizes_mm": lambda p: p.voxel_sizes_mm,
        "n_volumes": lambda p: p.n_volumes,
        "tr_s": lambda p: p.tr_s,
        "dtype": lambda p: p.dtype,
        "axis_convention": lambda p: p.axis_convention,
    }
    return {
        name: _summarise_values(getter(p) for p in profiles)
        for name, getter in fields.items()
    }


def _summarise_values(values: Any) -> dict[str, Any]:
    present = [_json_safe(v) for v in values if v is not None]
    if not present:
        return {"status": "absent", "n_values": 0, "values": []}
    unique_json = sorted({json.dumps(v, sort_keys=True, ensure_ascii=True) for v in present})
    unique_values = [json.loads(v) for v in unique_json]
    return {
        "status": "constant" if len(unique_values) == 1 else "variable",
        "n_values": len(unique_values),
        "values": unique_values[:16],
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)
