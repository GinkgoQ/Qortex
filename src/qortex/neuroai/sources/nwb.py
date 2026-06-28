"""NWB (Neurodata Without Borders) source adapter.

Uses pynwb to open NWB files (HDF5 backend), scans the acquisition group for
ElectricalSeries / TimeSeries / SpatialSeries, and yields QortexTimeSeries
windows or a full batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    ChannelSpec,
    EvidenceStatus,
    QortexTimeSeries,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)


class NWBAdapter(SourceAdapter):
    """Source adapter for local NWB files.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="nwb"`` and ``path=...`` pointing to a
        ``.nwb`` file.
    window_spec:
        Optional windowing configuration for streaming.
    channel_names:
        Optional channel subset.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        window_spec: WindowSpec | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        if not spec.path:
            raise ValueError("NWBAdapter requires spec.path")
        self._path = Path(spec.path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"NWB file not found: {self._path}")
        self._spec = spec
        self._window_spec = window_spec
        self._channel_names = channel_names

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        pynwb = _require_pynwb()
        with pynwb.NWBHDF5IO(str(self._path), mode="r") as io:
            nwbfile = io.read()
            series_list = _find_electrical_series(nwbfile)
            if not series_list:
                series_list = list(nwbfile.acquisition.values())

            if not series_list:
                return SourceProfile(
                    source_id=f"nwb:{self._path.name}",
                    source_type="nwb",
                    path=str(self._path),
                    modality="unknown",
                    abstraction="timeseries",
                    n_channels=0,
                    sampling_rate_hz=None,
                    evidence_status=EvidenceStatus.missing,
                    warnings=[
                        WarningItem(
                            code="NWB_NO_ACQUISITION_SERIES",
                            message="No readable acquisition TimeSeries found in NWB file.",
                            severity="error",
                        )
                    ],
                    evidence={"n_channels": EvidenceStatus.missing},
                )

            first = series_list[0]
            n_channels = _get_n_channels(first)
            srate = _get_srate(first)
            duration_s = _get_duration(first)
            ch_names = _get_channel_names(first, n_channels)
            channel_specs = [
                ChannelSpec(name=n, index=i, unit=getattr(first, "unit", "V"))
                for i, n in enumerate(ch_names)
            ]

            return SourceProfile(
                source_id=f"nwb:{self._path.name}",
                source_type="nwb",
                path=str(self._path),
                modality="eeg",
                abstraction="timeseries",
                n_channels=n_channels,
                sampling_rate_hz=srate,
                channel_names=ch_names,
                channel_specs=channel_specs,
                dtype="float64",
                axis_convention=AxisConvention.channels_time,
                duration_s=duration_s,
                evidence_status=EvidenceStatus.confirmed,
                extra={
                    "duration_s": duration_s,
                    "n_series": len(series_list),
                    "series_names": [s.name for s in series_list[:5]],
                },
                evidence={
                    "n_channels": EvidenceStatus.confirmed,
                    "sampling_rate": EvidenceStatus.confirmed if srate else EvidenceStatus.missing,
                    "duration": EvidenceStatus.inferred if duration_s else EvidenceStatus.unknown,
                },
            )

    def read_batch(self) -> list[QortexData]:
        pynwb = _require_pynwb()
        results = []
        with pynwb.NWBHDF5IO(str(self._path), mode="r") as io:
            nwbfile = io.read()
            for series in _find_electrical_series(nwbfile) or list(nwbfile.acquisition.values()):
                try:
                    data = np.array(series.data[:], dtype=np.float32)
                    if data.ndim == 1:
                        data = data[np.newaxis, :]  # [1, T]
                    else:
                        data = data.T  # NWB stores [T, Ch] → [Ch, T]
                    srate = _get_srate(series)
                    results.append(QortexTimeSeries(
                        data=data,
                        shape=data.shape,
                        axes=["channels", "times"],
                        dtype=str(data.dtype),
                        units=getattr(series, "unit", "V"),
                        sampling_frequency_hz=srate,
                        channel_names=_get_channel_names(series, data.shape[0]),
                        timebase="seconds_since_recording_start",
                        source_provenance={
                            "source_type": "nwb",
                            "path": str(self._path),
                            "series_name": series.name,
                        },
                    ))
                except Exception as exc:
                    log.warning("Could not load NWB series %s: %s", series.name, exc)
        return results

    def stream(self) -> Iterator[QortexData]:
        if self._window_spec is None:
            yield from self.read_batch()
            return

        for ts in self.read_batch():
            yield from _window_timeseries(ts, self._window_spec)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_pynwb():
    try:
        import pynwb
        return pynwb
    except ImportError:
        raise ImportError(
            "NWB support requires pynwb. "
            "Install with: pip install 'qortex[nwb]'"
        )


def _find_electrical_series(nwbfile):
    try:
        from pynwb.ecephys import ElectricalSeries
        return [v for v in nwbfile.acquisition.values() if isinstance(v, ElectricalSeries)]
    except Exception:
        return []


def _get_n_channels(series) -> int:
    try:
        data = series.data
        if hasattr(data, "shape"):
            s = data.shape
            return s[1] if len(s) > 1 else 1
        arr = np.array(data[:10])
        return arr.shape[1] if arr.ndim > 1 else 1
    except Exception:
        return 1


def _get_srate(series) -> float | None:
    for attr in ("rate", "sampling_rate", "timestamps_unit"):
        val = getattr(series, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return float(val)
    if hasattr(series, "timestamps") and series.timestamps is not None:
        ts = np.array(series.timestamps[:100])
        if len(ts) > 1:
            return float(1.0 / np.mean(np.diff(ts)))
    return None


def _get_duration(series) -> float | None:
    try:
        n = series.data.shape[0]
        srate = _get_srate(series)
        return n / srate if srate else None
    except Exception:
        return None


def _get_channel_names(series, n_channels: int) -> list[str]:
    try:
        if hasattr(series, "electrodes") and series.electrodes is not None:
            names = series.electrodes["label"].data[:]
            return [str(n) for n in names]
    except Exception:
        pass
    return [f"ch_{i}" for i in range(n_channels)]


def _window_timeseries(ts: QortexTimeSeries, window_spec: WindowSpec) -> Iterator[QortexTimeSeries]:
    srate = ts.sampling_frequency_hz or 1.0
    if window_spec.duration_s is None:
        yield ts
        return
    win_size = int(window_spec.duration_s * srate)
    step_s = window_spec.step_s if window_spec.step_s is not None else (
        window_spec.duration_s * (1.0 - window_spec.overlap_frac)
    )
    step_size = max(1, int(step_s * srate))
    data = ts.data
    n_samples = data.shape[-1]
    start = 0
    while start + win_size <= n_samples:
        chunk = data[..., start:start + win_size]
        yield QortexTimeSeries(
            data=chunk,
            shape=chunk.shape,
            axes=ts.axes,
            dtype=ts.dtype,
            units=ts.units,
            sampling_frequency_hz=ts.sampling_frequency_hz,
            channel_names=ts.channel_names,
            timebase=ts.timebase,
            source_provenance={
                **ts.source_provenance,
                "window_start_s": start / srate,
            },
        )
        start += step_size
