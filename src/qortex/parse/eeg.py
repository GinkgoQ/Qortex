"""EEG loader — MNE-BIDS with full BIDS sidecar inheritance and channel-table integration."""

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

# EEG file formats accepted by MNE
_EEG_EXTENSIONS = frozenset({
    ".set",   # EEGLAB
    ".edf",   # European Data Format
    ".bdf",   # BioSemi Data Format
    ".fif",   # Elekta/MNE native
    ".vhdr",  # BrainVision header
    ".cnt",   # Neuroscan
    ".mff",   # EGI MFF
    ".nxe",   # Nexus
    ".gdf",   # General Data Format
    ".raw",   # EGI raw (when datatype=eeg)
})

# Picks that select only EEG-type channels (excludes EOG/ECG/EMG/stim)
_EEG_CHANNEL_TYPES = frozenset({"eeg", "eog", "ecg", "emg"})


class EEGLoader:
    modality = "eeg"
    supported_extensions = _EEG_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "eeg"
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
            return {
                "sfreq": info["sfreq"],
                "n_channels": info["nchan"],
                "channel_type_counts": ch_type_counts,
                "meas_date": str(info.get("meas_date")) if info.get("meas_date") else None,
                "n_bad_channels": len(info.get("bads", [])),
                "powerline_freq": sidecar.get("PowerLineFrequency"),
                "eeg_reference": sidecar.get("EEGReference"),
                "software_filters": sidecar.get("SoftwareFilters"),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect EEG {local_path}: {exc}"
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
                file, local_path, datatype="eeg", suffix="eeg",
                preload=preload, extra_kwargs=kwargs,
            )
        except ImportError:
            raise
        except Exception as exc:
            raise LoadError(
                f"Cannot load EEG {local_path}: {exc}"
            ) from exc

        sidecar = load_json_sidecar(local_path)

        # Pick EEG + biophysical channels; exclude stim/misc
        eeg_picks = mne.pick_types(
            raw.info, eeg=True, eog=False, ecg=False, emg=False,
            stim=False, exclude="bads",
        )
        eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
        all_picks = list(eeg_picks) + list(eog_picks)

        ch_names = [raw.ch_names[i] for i in all_picks] if all_picks else raw.ch_names
        ch_types = [mne.channel_type(raw.info, i) for i in all_picks] if all_picks else raw.get_channel_types()

        return SignalRecord(
            file=file,
            raw=raw,
            sfreq=raw.info["sfreq"],
            n_channels=len(ch_names),
            duration=float(raw.times[-1]) if len(raw.times) else 0.0,
            channel_names=ch_names,
            channel_types=ch_types,
            metadata={
                "picks": all_picks,
                "n_bad_channels": len(raw.info.get("bads", [])),
                "bad_channels": list(raw.info.get("bads", [])),
                "bids_root": str(bids_root) if bids_root else None,
                "powerline_freq": sidecar.get("PowerLineFrequency"),
                "eeg_reference": sidecar.get("EEGReference"),
                "recording_duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: SignalRecord, **kwargs) -> np.ndarray:
        picks = record.metadata.get("picks")
        return raw_to_numpy(record.raw, picks=picks or None)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(
        self, record: SignalRecord, **kwargs
    ) -> Iterator[SampleRecord]:
        """Yield a single SampleRecord containing the full EEG array (n_ch, n_times).

        Windowing is handled downstream by ConversionPipeline.
        The signal is in float64 µV (MNE convention).
        """
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
                "eeg_reference": record.metadata.get("eeg_reference"),
            },
        )
