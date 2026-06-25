"""MEG loader — MNE-BIDS with gradiometer/magnetometer selection and SSS detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qortex.core.entities import FileRecord, SampleRecord, SignalRecord
from qortex.core.exceptions import LoadError
from qortex.parse._mne_utils import (
    load_json_sidecar,
    raw_to_numpy,
    read_raw_with_bids_fallback,
    require_mne,
)

log = logging.getLogger(__name__)

_MEG_EXTENSIONS = frozenset({
    ".fif",   # Elekta/Neuromag/MNE native
    ".ds",    # CTF dataset directory marker
    ".sqd",   # KIT/Yokogawa
    ".con",   # KIT/Yokogawa continuous
    ".4d",    # BTi/4D Neuroimaging
    ".kdf",   # Kriss MED
    ".mff",   # EGI MFF (MEG variant)
    ".nxe",   # Nexus MEG
})


class MEGLoader:
    modality = "meg"
    supported_extensions = _MEG_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "meg"
            and file.extension in self.supported_extensions
            and not file.is_dir
        )

    # ── inspect ───────────────────────────────────────────────────────────

    def inspect(self, file: FileRecord, local_path: Path) -> dict[str, Any]:
        mne = require_mne()
        try:
            info = mne.io.read_info(str(local_path))
            ch_type_counts: dict[str, int] = {}
            for i in range(info["nchan"]):
                ct = mne.channel_type(info, i)
                ch_type_counts[ct] = ch_type_counts.get(ct, 0) + 1

            sidecar = load_json_sidecar(local_path)
            sss_applied = bool(info.get("proc_history")) and any(
                "sss" in str(ph.get("max_info", {})).lower()
                for ph in info.get("proc_history", [])
            )
            return {
                "sfreq": info["sfreq"],
                "n_channels": info["nchan"],
                "channel_type_counts": ch_type_counts,
                "meas_date": str(info.get("meas_date")) if info.get("meas_date") else None,
                "n_bad_channels": len(info.get("bads", [])),
                "sss_applied": sss_applied,
                "powerline_freq": sidecar.get("PowerLineFrequency"),
                "meg_channel_count": ch_type_counts.get("grad", 0) + ch_type_counts.get("mag", 0),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect MEG {local_path}: {exc}"
            ) from exc

    # ── load / lazy_load ──────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> SignalRecord:
        return self._read(file, local_path, preload=True, **kwargs)

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> SignalRecord:
        return self._read(file, local_path, preload=False, **kwargs)

    def _read(
        self, file: FileRecord, local_path: Path, preload: bool = True, **kwargs
    ) -> SignalRecord:
        mne = require_mne()
        try:
            raw, bids_root, channels_meta = read_raw_with_bids_fallback(
                file, local_path, datatype="meg", suffix="meg",
                preload=preload, extra_kwargs=kwargs,
            )
        except ImportError:
            raise
        except Exception as exc:
            raise LoadError(
                f"Cannot load MEG {local_path}: {exc}"
            ) from exc

        sidecar = load_json_sidecar(local_path)

        # Select magnetometers + gradiometers; exclude ref channels and bads
        meg_picks = mne.pick_types(
            raw.info, meg=True, ref_meg=False, stim=False, exclude="bads"
        )
        ch_names = [raw.ch_names[i] for i in meg_picks] if meg_picks.size else raw.ch_names
        ch_types = [mne.channel_type(raw.info, i) for i in meg_picks] if meg_picks.size else raw.get_channel_types()

        # Detect whether Maxwell Spatial Suppression was applied
        proc_history = raw.info.get("proc_history", [])
        sss_applied = any("sss" in str(ph.get("max_info", {})).lower() for ph in proc_history)

        return SignalRecord(
            file=file,
            raw=raw,
            sfreq=raw.info["sfreq"],
            n_channels=len(ch_names),
            duration=float(raw.times[-1]) if len(raw.times) else 0.0,
            channel_names=ch_names,
            channel_types=ch_types,
            metadata={
                "picks": list(meg_picks),
                "n_bad_channels": len(raw.info.get("bads", [])),
                "bad_channels": list(raw.info.get("bads", [])),
                "sss_applied": sss_applied,
                "bids_root": str(bids_root) if bids_root else None,
                "powerline_freq": sidecar.get("PowerLineFrequency"),
                "dewar_position": sidecar.get("DewarPosition"),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: SignalRecord, **kwargs) -> np.ndarray:
        picks = record.metadata.get("picks")
        return raw_to_numpy(record.raw, picks=picks or None)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: SignalRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord (n_meg_ch, n_times) for the full recording."""
        data = self.to_numpy(record)
        ents = record.file.entities
        yield SampleRecord(
            data=data,
            modality=self.modality,
            subject=ents.subject,
            session=ents.session,
            task=ents.task,
            run=ents.run,
            sfreq=record.sfreq,
            onset=0.0,
            duration=record.duration,
            provenance={
                "source": record.file.path,
                "n_channels": record.n_channels,
                "bad_channels": record.metadata.get("bad_channels", []),
                "sss_applied": record.metadata.get("sss_applied", False),
            },
        )
