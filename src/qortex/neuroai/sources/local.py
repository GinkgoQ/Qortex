"""Local file source adapter.

Probes and streams from local EDF, BDF, FIF, NIfTI (.nii / .nii.gz), DICOM,
Parquet/CSV/TSV, and XDF files without any OpenNeuro dependency.

Each format has a dedicated ``_probe_*`` and ``_read_*`` function so the main
adapter stays thin.  All heavy imports are deferred to the point of use so the
module can import without optional extras installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

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
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)


_EXT_TO_MODALITY: dict[str, str] = {
    ".edf":    "eeg",
    ".bdf":    "eeg",
    ".fif":    "meg",
    ".set":    "eeg",
    ".vhdr":   "eeg",
    ".nii":    "mri",
    ".gz":     "mri",     # .nii.gz
    ".dcm":    "dicom",
    ".parquet":"tabular",
    ".csv":    "tabular",
    ".tsv":    "tabular",
    ".xdf":    "eeg",
}


class LocalFileAdapter(SourceAdapter):
    """Source adapter for local signal and imaging files.

    Supports:  EDF, BDF, FIF, SET, VHDR (signal) + NIfTI (volume) + Parquet/CSV/TSV (tabular)

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="local_file"`` and ``path=...``.
    window_spec:
        Optional windowing configuration for streaming signal data.
    channel_names:
        Subset of channel names to load.  ``None`` = all channels.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        window_spec: WindowSpec | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        if not spec.path:
            raise ValueError("LocalFileAdapter requires spec.path")
        self._path = Path(spec.path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Source file not found: {self._path}")
        self._spec = spec
        self._window_spec = window_spec
        self._channel_names = channel_names
        self._ext = self._detect_ext()

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        """Header-only probe — no full data load."""
        if self._is_signal():
            return self._probe_signal()
        elif self._is_volume():
            return self._probe_volume()
        else:
            return self._probe_generic()

    def read_batch(self) -> list[QortexData]:
        if self._is_signal():
            return [self._load_signal()]
        elif self._is_volume():
            return [self._load_volume()]
        else:
            return [self._load_generic()]

    def stream(self) -> Iterator[QortexData]:
        if self._is_signal() and self._window_spec is not None:
            yield from self._stream_windowed()
        else:
            yield from self.read_batch()

    @property
    def source_id(self) -> str:
        return f"local_file:{self._path.name}"

    # ── Signal probing ────────────────────────────────────────────────────────

    def _probe_signal(self) -> SourceProfile:
        try:
            import mne
        except ImportError:
            raise ImportError(
                "Signal file probing requires MNE. "
                "Install with: pip install 'qortex[eeg]'"
            ) from None

        try:
            raw = self._open_mne_raw(preload=False)
        except Exception as exc:
            return SourceProfile(
                source_id=self.source_id,
                source_type="local_file",
                path=str(self._path),
                modality=self._guess_modality(),
                evidence_status=EvidenceStatus.missing,
                warnings=[WarningItem(
                    code="PROBE_FAILED",
                    message=f"Cannot open {self._path.name}: {exc}",
                    severity="error",
                )],
            )

        info = raw.info
        ch_names = info.ch_names
        sfreq = info.get("sfreq", None)
        n_times = raw.n_times
        dur_s = n_times / sfreq if sfreq else None

        ch_specs = [
            ChannelSpec(
                name=ch,
                index=i,
                sampling_rate_hz=sfreq,
                channel_type=str(info.get_channel_types([ch])[0]) if hasattr(info, "get_channel_types") else None,
            )
            for i, ch in enumerate(ch_names)
        ]

        return SourceProfile(
            source_id=self.source_id,
            source_type="local_file",
            path=str(self._path),
            modality=self._guess_modality(),
            abstraction="timeseries",
            n_channels=len(ch_names),
            sampling_rate_hz=sfreq,
            channel_names=ch_names,
            channel_specs=ch_specs,
            duration_s=dur_s,
            dtype="float64",
            axis_convention=AxisConvention.channels_time,
            evidence_status=EvidenceStatus.confirmed,
        )

    def _load_signal(self) -> QortexTimeSeries:
        import mne
        raw = self._open_mne_raw(preload=True)
        if self._channel_names:
            raw.pick_channels(self._channel_names)
        data = raw.get_data().astype(np.float32)  # (n_ch, n_times)
        info = raw.info
        ch_names = info.ch_names
        sfreq = info["sfreq"]
        return QortexTimeSeries(
            data=data,
            shape=data.shape,
            axes=["channels", "times"],
            dtype=str(data.dtype),
            channel_names=list(ch_names),
            sampling_frequency_hz=sfreq,
            timebase="seconds_since_recording_start",
            source_provenance={"path": str(self._path), "n_channels": len(ch_names)},
        )

    def _stream_windowed(self) -> Iterator[QortexTimeSeries]:
        import mne
        ws = self._window_spec
        raw = self._open_mne_raw(preload=True)
        if self._channel_names:
            raw.pick_channels(self._channel_names)
        sfreq = raw.info["sfreq"]
        ch_names = list(raw.info.ch_names)
        total_dur = raw.n_times / sfreq

        if ws.duration_s is None:
            yield self._load_signal()
            return

        win_dur = ws.duration_s
        step = ws.step_s if ws.step_s is not None else (win_dur * (1.0 - ws.overlap_frac))
        if step <= 0:
            step = win_dur

        tstart = ws.tmin
        while tstart + win_dur <= total_dur + 1e-6:
            tend = min(tstart + win_dur, total_dur)
            if tend - tstart < win_dur * 0.5 and ws.drop_short:
                break
            data, _ = raw.get_data(tmin=tstart, tmax=tend, return_times=True)
            data = data.astype(np.float32)
            yield QortexTimeSeries(
                data=data,
                shape=data.shape,
                axes=["channels", "times"],
                dtype=str(data.dtype),
                channel_names=ch_names,
                sampling_frequency_hz=sfreq,
                timebase="seconds_since_recording_start",
                source_provenance={
                    "path": str(self._path),
                    "tmin": tstart,
                    "tmax": tend,
                },
            )
            tstart += step

    def _open_mne_raw(self, preload: bool = False):
        import mne
        ext = self._ext
        path_str = str(self._path)
        if ext in (".edf", ".bdf"):
            return mne.io.read_raw_edf(path_str, preload=preload, verbose=False)
        elif ext == ".fif":
            return mne.io.read_raw_fif(path_str, preload=preload, verbose=False)
        elif ext == ".set":
            return mne.io.read_raw_eeglab(path_str, preload=preload, verbose=False)
        elif ext == ".vhdr":
            return mne.io.read_raw_brainvision(path_str, preload=preload, verbose=False)
        elif ext == ".xdf":
            return mne.io.read_raw(path_str, preload=preload, verbose=False)
        else:
            return mne.io.read_raw(path_str, preload=preload, verbose=False)

    # ── Volume probing ────────────────────────────────────────────────────────

    def _probe_volume(self) -> SourceProfile:
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "NIfTI probing requires nibabel. "
                "Install with: pip install 'qortex[mri]'"
            ) from None

        img = nib.load(str(self._path))
        shape = img.shape
        vox = tuple(abs(float(v)) for v in img.header.get_zooms()[:3])
        zooms = img.header.get_zooms()
        tr_s = float(zooms[3]) if len(zooms) > 3 else None

        return SourceProfile(
            source_id=self.source_id,
            source_type="local_file",
            path=str(self._path),
            modality=Modality.mri,
            abstraction="volume",
            spatial_shape=tuple(shape[:3]),
            n_volumes=shape[3] if len(shape) > 3 else None,
            voxel_sizes_mm=vox,
            tr_s=tr_s,
            dtype=str(np.dtype(img.get_data_dtype())),
            axis_convention=AxisConvention.RAS,
            evidence_status=EvidenceStatus.confirmed,
        )

    def _load_volume(self) -> QortexVolume:
        import nibabel as nib
        img = nib.load(str(self._path))
        img = nib.as_closest_canonical(img)
        data = img.get_fdata(dtype=np.float32)
        shape = data.shape
        vox = tuple(abs(float(v)) for v in img.header.get_zooms()[:3])
        affine = img.affine.tolist()
        return QortexVolume(
            data=data,
            shape=shape,
            axes=["x", "y", "z"] if len(shape) == 3 else ["x", "y", "z", "t"],
            dtype=str(data.dtype),
            voxel_sizes_mm=vox,
            affine=affine,
            coordinate_frame="RAS",
            source_provenance={"path": str(self._path)},
        )

    # ── Generic (tabular) ─────────────────────────────────────────────────────

    def _probe_generic(self) -> SourceProfile:
        columns: list[str] = []
        n_events: int = 0
        ext = self._ext
        try:
            if ext == ".parquet":
                import polars as pl
                lazy = pl.scan_parquet(str(self._path))
                columns = lazy.columns
                n_events = lazy.select(pl.len()).collect().item()
            elif ext in (".csv", ".tsv"):
                sep = "\t" if ext == ".tsv" else ","
                import polars as pl
                lazy = pl.scan_csv(str(self._path), separator=sep)
                columns = lazy.columns
                n_events = lazy.select(pl.len()).collect().item()
        except Exception:
            pass
        return SourceProfile(
            source_id=self.source_id,
            source_type="local_file",
            path=str(self._path),
            modality=Modality.tabular,
            n_channels=len(columns) if columns else None,
            channel_names=columns or None,
            evidence_status=EvidenceStatus.confirmed if columns else EvidenceStatus.inferred,
            extra={"n_rows": n_events, "columns": columns},
        )

    def _load_generic(self) -> "QortexEventTable":
        from qortex.neuroai.contracts import QortexEventTable
        ext = self._ext
        if ext == ".parquet":
            import polars as pl
            df = pl.read_parquet(str(self._path))
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            import polars as pl
            df = pl.read_csv(str(self._path), separator=sep)
        else:
            df = None
        columns = list(df.columns) if df is not None else []
        n_events = len(df) if df is not None else 0
        tbl = QortexEventTable(
            shape=(n_events, len(columns)),
            axes=["rows", "columns"],
            dtype="mixed",
            columns=columns,
            n_events=n_events,
            source_provenance={"path": str(self._path)},
        )
        # Store the dataframe as the data payload (not a numpy array, but
        # downstream code can detect the type and handle it).
        tbl.data = df
        return tbl

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_ext(self) -> str:
        name = self._path.name.lower()
        if name.endswith(".nii.gz"):
            return ".gz"
        return self._path.suffix.lower()

    def _is_signal(self) -> bool:
        return self._ext in (".edf", ".bdf", ".fif", ".set", ".vhdr", ".xdf")

    def _is_volume(self) -> bool:
        return self._ext in (".nii", ".gz")

    def _guess_modality(self) -> str:
        return _EXT_TO_MODALITY.get(self._ext, "unknown")
