"""iEEG loader — MNE-BIDS with electrode localization and reference handling."""

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

_IEEG_EXTENSIONS = frozenset({
    ".edf",   # European Data Format (most common for clinical iEEG)
    ".bdf",   # BioSemi
    ".vhdr",  # BrainVision
    ".eeg",   # BrainVision data file (paired with .vhdr)
    ".set",   # EEGLAB
    ".fif",   # MNE native
    ".nwb",   # Neurodata Without Borders
})


class IEEGLoader:
    modality = "ieeg"
    supported_extensions = _IEEG_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "ieeg"
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
            elec_type = sidecar.get("iEEGElectrodeGroups", sidecar.get("ElectrodeManufacturer"))
            return {
                "sfreq": info["sfreq"],
                "n_channels": info["nchan"],
                "channel_type_counts": ch_type_counts,
                "n_bad_channels": len(info.get("bads", [])),
                "meas_date": str(info.get("meas_date")) if info.get("meas_date") else None,
                "ieeg_reference": sidecar.get("iEEGReference"),
                "electrode_manufacturer": elec_type,
                "institution": sidecar.get("InstitutionName"),
                "recording_type": sidecar.get("iEEGRecordingType"),  # "continuous" | "epoched"
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect iEEG {local_path}: {exc}"
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
                file, local_path, datatype="ieeg", suffix="ieeg",
                preload=preload, extra_kwargs=kwargs,
            )
        except ImportError:
            raise
        except Exception as exc:
            raise LoadError(
                f"Cannot load iEEG {local_path}: {exc}"
            ) from exc

        sidecar = load_json_sidecar(local_path)

        # iEEG has SEEG, ECoG, DBS channel types — select all neural + misc
        ieeg_picks = mne.pick_types(
            raw.info, seeg=True, ecog=True, dbs=True,
            stim=False, misc=False, exclude="bads",
        )
        # Fallback to all non-stim channels when no specialized picks found
        if not ieeg_picks.size:
            ieeg_picks = mne.pick_types(raw.info, stim=False, exclude="bads")

        ch_names = [raw.ch_names[i] for i in ieeg_picks] if ieeg_picks.size else raw.ch_names
        ch_types = [mne.channel_type(raw.info, i) for i in ieeg_picks] if ieeg_picks.size else raw.get_channel_types()

        # Load electrode coordinate table if available
        electrode_coords = _load_electrode_coords(bids_root, file) if bids_root else {}

        return SignalRecord(
            file=file,
            raw=raw,
            sfreq=raw.info["sfreq"],
            n_channels=len(ch_names),
            duration=float(raw.times[-1]) if len(raw.times) else 0.0,
            channel_names=ch_names,
            channel_types=ch_types,
            metadata={
                "picks": list(ieeg_picks),
                "n_bad_channels": len(raw.info.get("bads", [])),
                "bad_channels": list(raw.info.get("bads", [])),
                "bids_root": str(bids_root) if bids_root else None,
                "ieeg_reference": sidecar.get("iEEGReference"),
                "recording_type": sidecar.get("iEEGRecordingType"),
                "electrode_coords": electrode_coords,
                "n_electrodes_with_coords": len(electrode_coords),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: SignalRecord, **kwargs) -> np.ndarray:
        picks = record.metadata.get("picks")
        return raw_to_numpy(record.raw, picks=picks or None)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: SignalRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord (n_ieeg_ch, n_times) for the full recording."""
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
                "ieeg_reference": record.metadata.get("ieeg_reference"),
                "n_electrodes_with_coords": record.metadata.get("n_electrodes_with_coords", 0),
            },
        )


# ── Electrode coordinate helper ───────────────────────────────────────────────

def _load_electrode_coords(bids_root: Path, file: FileRecord) -> dict[str, tuple[float, float, float]]:
    """Load x/y/z coordinates from *_electrodes.tsv (BIDS iEEG spec).

    Returns dict mapping electrode name → (x, y, z) in mm.
    """
    ents = file.entities
    candidates = []
    for subdir in [
        f"sub-{ents.subject}/ses-{ents.session}/ieeg" if ents.session else f"sub-{ents.subject}/ieeg",
        f"sub-{ents.subject}/ieeg",
    ]:
        suffix_variants = []
        if ents.session:
            suffix_variants.append(f"sub-{ents.subject}_ses-{ents.session}_electrodes.tsv")
        suffix_variants.append(f"sub-{ents.subject}_electrodes.tsv")
        for sv in suffix_variants:
            candidates.append(bids_root / subdir / sv)

    for path in candidates:
        if path.exists():
            try:
                import polars as pl
                df = pl.read_csv(str(path), separator="\t", null_values=["n/a", "N/A"])
                coords: dict[str, tuple[float, float, float]] = {}
                for row in df.iter_rows(named=True):
                    name = row.get("name", "")
                    x, y, z = row.get("x"), row.get("y"), row.get("z")
                    if name and x is not None and y is not None and z is not None:
                        try:
                            coords[name] = (float(x), float(y), float(z))
                        except (TypeError, ValueError):
                            pass
                return coords
            except Exception:
                pass
    return {}
