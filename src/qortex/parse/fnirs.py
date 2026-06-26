"""fNIRS loader — MNE-BIDS with haemodynamic channel-type selection and optode metadata."""

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
    require_mne,
    require_mne_bids,
    resolve_bids_root,
)

log = logging.getLogger(__name__)

_FNIRS_EXTENSIONS = frozenset({
    ".snirf",  # Shared Near Infrared spectroscopy Format (BIDS required)
    ".nirs",   # Homer2 legacy format
})


class FNIRSLoader:
    modality = "fnirs"
    supported_extensions = _FNIRS_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype in {"fnirs", "nirs"}
            and file.extension in self.supported_extensions
            and not file.is_dir
        )

    # ── inspect ───────────────────────────────────────────────────────────

    def inspect(self, file: FileRecord, local_path: Path) -> dict[str, Any]:
        mne = require_mne()
        try:
            if file.extension == ".snirf":
                raw = mne.io.read_raw_snirf(str(local_path), preload=False)
            else:
                raw = mne.io.read_raw_nirx(str(local_path), preload=False)

            ch_type_counts: dict[str, int] = {}
            for i in range(raw.info["nchan"]):
                ct = mne.channel_type(raw.info, i)
                ch_type_counts[ct] = ch_type_counts.get(ct, 0) + 1

            sidecar = load_json_sidecar(local_path)
            return {
                "sfreq": raw.info["sfreq"],
                "n_channels": raw.info["nchan"],
                "channel_type_counts": ch_type_counts,
                "duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
                "manufacturer": sidecar.get("Manufacturer"),
                "wavelengths_nm": sidecar.get("NIRSSourceOptodeDescription"),
                "sampling_frequency": sidecar.get("SamplingFrequency"),
                "short_channel_count": sidecar.get("ShortChannelCount", 0),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect fNIRS {local_path}: {exc}"
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
        raw = None

        # Try MNE-BIDS first (full sidecar inheritance)
        bids_root = resolve_bids_root(file, local_path)
        if bids_root is not None:
            try:
                import mne_bids
                ents = file.entities
                bids_path = mne_bids.BIDSPath(
                    subject=ents.subject,
                    session=ents.session,
                    task=ents.task,
                    run=ents.run,
                    datatype="nirs",
                    root=bids_root,
                    suffix="nirs",
                    extension=file.extension,
                )
                kw = {k: v for k, v in kwargs.items() if k != "preload"}
                raw = mne_bids.read_raw_bids(bids_path, preload=preload, **kw)
                log.debug("MNE-BIDS fNIRS read succeeded: %s", local_path)
            except Exception as exc:
                log.debug("MNE-BIDS fNIRS failed (%s), falling back", exc)
                raw = None

        if raw is None:
            try:
                kw = {k: v for k, v in kwargs.items() if k != "preload"}
                if file.extension == ".snirf":
                    raw = mne.io.read_raw_snirf(str(local_path), preload=preload, **kw)
                else:
                    raw = mne.io.read_raw_nirx(str(local_path.parent), preload=preload, **kw)
            except Exception as exc:
                raise LoadError(
                    f"Cannot load fNIRS {local_path}: {exc}"
                ) from exc

        sidecar = load_json_sidecar(local_path)

        # fNIRS channel types: hbo / hbr (haemoglobin), fnirs_cw_amplitude, fnirs_od
        hb_picks = mne.pick_types(raw.info, fnirs=True, exclude="bads")
        if not hb_picks.size:
            hb_picks = mne.pick_types(raw.info, exclude="bads")

        ch_names = [raw.ch_names[i] for i in hb_picks] if hb_picks.size else raw.ch_names
        ch_types = [mne.channel_type(raw.info, i) for i in hb_picks] if hb_picks.size else raw.get_channel_types()

        # Separate HbO and HbR channel indices for downstream analysis
        hbo_picks = mne.pick_types(raw.info, fnirs="hbo", exclude="bads")
        hbr_picks = mne.pick_types(raw.info, fnirs="hbr", exclude="bads")

        return SignalRecord(
            file=file,
            raw=raw,
            sfreq=raw.info["sfreq"],
            n_channels=len(ch_names),
            duration=float(raw.times[-1]) if len(raw.times) else 0.0,
            channel_names=ch_names,
            channel_types=ch_types,
            metadata={
                "picks": list(hb_picks),
                "hbo_picks": list(hbo_picks),
                "hbr_picks": list(hbr_picks),
                "n_bad_channels": len(raw.info.get("bads", [])),
                "bad_channels": list(raw.info.get("bads", [])),
                "bids_root": str(bids_root) if bids_root else None,
                "manufacturer": sidecar.get("Manufacturer"),
                "short_channel_count": sidecar.get("ShortChannelCount", 0),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: SignalRecord, **kwargs) -> np.ndarray:
        picks = record.metadata.get("picks")
        return raw_to_numpy(record.raw, picks=picks or None)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: SignalRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord with haemodynamic channels (n_ch, n_times).

        Data is in the native MNE unit (mol/L for HbO/HbR after Beer-Lambert,
        or raw amplitude/OD before conversion).
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
                "n_hbo_channels": len(record.metadata.get("hbo_picks", [])),
                "n_hbr_channels": len(record.metadata.get("hbr_picks", [])),
                "bad_channels": record.metadata.get("bad_channels", []),
                "manufacturer": record.metadata.get("manufacturer"),
            },
        )
