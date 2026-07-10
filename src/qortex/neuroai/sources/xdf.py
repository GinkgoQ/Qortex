"""XDF (Extensible Data Format) source adapter.

Uses pyxdf to load XDF recordings, selects streams by spec.query, and yields
windowed QortexTimeSeries for each EEG (or specified) stream.
"""

from __future__ import annotations

import logging
import time
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


class XDFAdapter(SourceAdapter):
    """Source adapter for XDF (Extensible Data Format) files.

    XDF is the de-facto standard for multi-stream neurophysiology recording.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="xdf"`` and ``path=...``.
        ``spec.query`` may contain ``{"type": "EEG"}`` or ``{"name": "my_stream"}``
        to filter streams.
    window_spec:
        Optional windowing for streaming.
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
            raise ValueError("XDFAdapter requires spec.path")
        self._path = Path(spec.path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"XDF file not found: {self._path}")
        self._spec = spec
        self._window_spec = window_spec
        self._channel_names = channel_names
        self._query = spec.query or {}

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        pyxdf = _require_pyxdf()
        # Load header only — use select_streams=[] which is still fast
        try:
            streams, header = pyxdf.load_xdf(str(self._path), select_streams=[])
        except TypeError:
            # Some versions don't support select_streams
            streams, header = pyxdf.load_xdf(str(self._path))

        target = self._select_streams(streams)
        if not target:
            return SourceProfile(
                source_id=f"xdf:{self._path.name}",
                source_type="xdf",
                modality="unknown",
                n_channels=0,
                sampling_rate_hz=None,
                path=str(self._path),
                evidence_status=EvidenceStatus.missing,
                evidence={"n_channels": EvidenceStatus.missing},
                warnings=[WarningItem(
                    code="NO_MATCHING_STREAM",
                    message=f"No stream matched query {self._query}",
                    severity="warning",
                    suggestion="Check spec.query type/name.",
                )],
            )

        first_info = target[0]["info"]
        n_channels = int(first_info.get("channel_count", [1])[0])
        srate = float(first_info.get("nominal_srate", [0.0])[0])
        stream_type = str(first_info.get("type", ["unknown"])[0]).lower()
        ch_names = _extract_channel_names(first_info, n_channels)
        channel_specs = [
            ChannelSpec(name=n, index=i) for i, n in enumerate(ch_names)
        ]

        n_samples = 0
        if "time_series" in target[0]:
            data = target[0]["time_series"]
            n_samples = len(data) if data is not None else 0

        return SourceProfile(
            source_id=f"xdf:{self._path.name}",
            source_type="xdf",
            modality=stream_type if stream_type in ("eeg", "meg", "ecg", "emg") else "signal",
            n_channels=n_channels,
            sampling_rate_hz=srate if srate > 0 else None,
            channel_names=[s.name for s in channel_specs],
            channel_specs=channel_specs,
            dtype="float32",
            axis_convention=AxisConvention.channels_time,
            path=str(self._path),
            extra={
                "stream_count": len(streams),
                "matched_streams": len(target),
                "n_samples": n_samples,
            },
            evidence={
                "n_channels": EvidenceStatus.confirmed,
                "sampling_rate": EvidenceStatus.confirmed if srate > 0 else EvidenceStatus.missing,
            },
        )

    def read_batch(self) -> list[QortexData]:
        pyxdf = _require_pyxdf()
        streams, _ = pyxdf.load_xdf(str(self._path))
        target = self._select_streams(streams)
        results = []
        for stream in target:
            ts = self._stream_to_timeseries(stream)
            if ts is not None:
                results.append(ts)
        return results

    def stream(self) -> Iterator[QortexData]:
        for ts in self.read_batch():
            if self._window_spec is not None:
                yield from _window_timeseries(ts, self._window_spec)
            else:
                yield ts

    def replay(self, speed: float = 1.0) -> Iterator[QortexData]:
        """Replay with accurate timing based on window duration / speed."""
        for ts in self.read_batch():
            window_iter = (
                _window_timeseries(ts, self._window_spec)
                if self._window_spec
                else iter([ts])
            )
            srate = ts.sampling_frequency_hz or 1.0
            win_size = int(
                (self._window_spec.duration_s if self._window_spec else ts.shape[-1] / srate)
                * srate
            )
            sleep_s = win_size / srate / max(speed, 1e-6)
            for chunk in window_iter:
                yield chunk
                if speed > 0:
                    time.sleep(sleep_s)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_streams(self, streams: list[dict]) -> list[dict]:
        if not self._query:
            return streams
        q_type = self._query.get("type", "").lower()
        q_name = self._query.get("name", "").lower()
        result = []
        for s in streams:
            info = s.get("info", {})
            s_type = str(info.get("type", [""])[0]).lower()
            s_name = str(info.get("name", [""])[0]).lower()
            if q_type and q_type not in s_type:
                continue
            if q_name and q_name not in s_name:
                continue
            result.append(s)
        return result

    def _stream_to_timeseries(self, stream: dict) -> QortexTimeSeries | None:
        info = stream.get("info", {})
        data = stream.get("time_series")
        if data is None or len(data) == 0:
            return None

        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        else:
            arr = arr.T  # XDF stores [T, Ch] → [Ch, T]

        n_channels = arr.shape[0]
        srate = float(info.get("nominal_srate", [0.0])[0])
        ch_names = _extract_channel_names(info, n_channels)
        stream_type = str(info.get("type", ["signal"])[0]).lower()

        return QortexTimeSeries(
            data=arr,
            shape=arr.shape,
            axes=["channels", "time"],
            dtype="float32",
            units="uV",
            sampling_frequency_hz=srate if srate > 0 else None,
            channel_names=ch_names,
            source_provenance={
                "source_type": "xdf",
                "path": str(self._path),
                "stream_name": str(info.get("name", ["unknown"])[0]),
                "stream_type": stream_type,
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_pyxdf():
    try:
        import pyxdf
        return pyxdf
    except ImportError:
        raise ImportError(
            "XDF support requires pyxdf. "
            "Install with: pip install 'qortex[xdf]' or pip install pyxdf"
        )


def _extract_channel_names(info: dict, n_channels: int) -> list[str]:
    try:
        ch_info = info.get("desc", [{}])[0].get("channels", [{}])[0].get("channel", [])
        names = [c.get("label", [f"ch_{i}"])[0] for i, c in enumerate(ch_info)]
        if len(names) == n_channels:
            return names
    except Exception:
        pass
    return [f"ch_{i}" for i in range(n_channels)]


def _window_timeseries(ts: QortexTimeSeries, window_spec: WindowSpec) -> Iterator[QortexTimeSeries]:
    srate = ts.sampling_frequency_hz or 1.0
    win_size = int(window_spec.duration_s * srate)
    step_size = int(getattr(window_spec, "step_s", window_spec.duration_s) * srate)
    n_samples = ts.shape[-1]
    start = 0
    while start + win_size <= n_samples:
        chunk = ts.data[..., start:start + win_size]
        yield QortexTimeSeries(
            data=chunk,
            shape=chunk.shape,
            axes=ts.axes,
            dtype=ts.dtype,
            units=ts.units,
            sampling_frequency_hz=ts.sampling_frequency_hz,
            channel_names=ts.channel_names,
            source_provenance={**ts.source_provenance, "window_start_s": start / srate},
        )
        start += step_size
