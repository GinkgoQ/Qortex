"""LSL (Lab Streaming Layer) live-stream source adapter.

Resolves LSL streams matching spec.query, opens an inlet, and yields
windowed QortexTimeSeries using an internal ring buffer.
"""

from __future__ import annotations

import logging
import time
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

_RESOLVE_TIMEOUT_S = 5.0


class LSLSourceAdapter(SourceAdapter):
    """Source adapter for live LSL (Lab Streaming Layer) streams.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="lsl"``.
        ``spec.query`` may contain ``{"type": "EEG"}`` or ``{"name": "stream_name"}``.
        ``spec.extra.get("duration_s", 10)`` controls read_batch() collection time.
    window_spec:
        Windowing configuration.  Required for streaming; if absent, defaults
        to 1-second windows.
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
        self._spec = spec
        self._window_spec = window_spec
        self._channel_names = channel_names
        self._query = spec.query or {}
        self._extra = spec.extra or {}

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        pylsl = _require_pylsl()
        log.info("LSL: resolving streams (timeout=%.1fs)…", _RESOLVE_TIMEOUT_S)

        streams = pylsl.resolve_streams(wait_time=_RESOLVE_TIMEOUT_S)
        target = self._filter_streams(streams)

        if not target:
            return SourceProfile(
                source_id="lsl:no_stream",
                source_type="lsl",
                modality="unknown",
                n_channels=0,
                sampling_rate_hz=None,
                path=None,
                evidence_status=EvidenceStatus.missing,
                evidence={"n_channels": EvidenceStatus.missing},
                warnings=[WarningItem(
                    code="LSL_NO_STREAM",
                    message=f"No LSL stream found matching {self._query}",
                    severity="warning",
                    suggestion="Start the LSL source before calling probe().",
                )],
            )

        info = target[0]
        n_channels = info.channel_count()
        srate = info.nominal_srate()
        stream_type = info.type().lower()

        ch_names = _extract_lsl_channel_names(info)
        channel_specs = [ChannelSpec(name=n, index=i) for i, n in enumerate(ch_names)]

        return SourceProfile(
            source_id=f"lsl:{info.name()}",
            source_type="lsl",
            modality=stream_type if stream_type in ("eeg", "meg", "ecg", "emg") else "signal",
            n_channels=n_channels,
            sampling_rate_hz=srate if srate > 0 else None,
            channel_names=ch_names,
            channel_specs=channel_specs,
            dtype="float32",
            axis_convention=AxisConvention.channels_time,
            path=None,
            extra={
                "stream_name": info.name(),
                "stream_uid": info.uid(),
                "hostname": info.hostname(),
            },
            evidence={
                "n_channels": EvidenceStatus.confirmed,
                "sampling_rate": EvidenceStatus.confirmed if srate > 0 else EvidenceStatus.missing,
            },
        )

    def read_batch(self) -> list[QortexData]:
        """Collect data for spec.extra['duration_s'] seconds."""
        pylsl = _require_pylsl()
        duration_s = float(self._extra.get("duration_s", 10.0))
        streams = pylsl.resolve_streams(wait_time=_RESOLVE_TIMEOUT_S)
        target = self._filter_streams(streams)
        if not target:
            raise RuntimeError("No LSL stream found for read_batch()")

        inlet = pylsl.StreamInlet(target[0], max_buflen=int(duration_s * 2))
        inlet.open_stream()
        all_samples: list[list[float]] = []
        deadline = time.time() + duration_s

        try:
            while time.time() < deadline:
                chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=512)
                if chunk:
                    all_samples.extend(chunk)
        finally:
            inlet.close_stream()

        if not all_samples:
            return []

        arr = np.array(all_samples, dtype=np.float32).T  # [Ch, T]
        srate = target[0].nominal_srate()
        return [QortexTimeSeries(
            data=arr,
            shape=arr.shape,
            axes=["channels", "time"],
            dtype="float32",
            units="uV",
            sampling_frequency_hz=srate,
            channel_names=_extract_lsl_channel_names(target[0]),
            source_provenance={"source_type": "lsl", "stream_name": target[0].name()},
        )]

    def stream(self) -> Iterator[QortexData]:
        pylsl = _require_pylsl()
        streams = pylsl.resolve_streams(wait_time=_RESOLVE_TIMEOUT_S)
        target = self._filter_streams(streams)
        if not target:
            raise RuntimeError("No LSL stream found for stream()")

        info = target[0]
        n_channels = info.channel_count()
        srate = info.nominal_srate() or 256.0
        win_dur = self._window_spec.duration_s if self._window_spec else 1.0
        step_dur = (
            getattr(self._window_spec, "step_s", win_dur)
            if self._window_spec else win_dur
        )
        win_samples = max(1, int(win_dur * srate))
        step_samples = max(1, int(step_dur * srate))
        ch_names = _extract_lsl_channel_names(info)

        from qortex.neuroai.sources._ring_buffer import get_ring_buffer
        buf = get_ring_buffer(
            n_channels=n_channels,
            capacity=win_samples * 8,
            window_size=win_samples,
            step_size=step_samples,
        )

        inlet = pylsl.StreamInlet(info, max_buflen=int(win_dur * 4))
        inlet.open_stream()
        log.info("LSL: streaming from %s at %.1f Hz", info.name(), srate)

        window_idx = 0
        try:
            while True:
                chunk, timestamps = inlet.pull_chunk(timeout=win_dur / 4, max_samples=512)
                if chunk:
                    arr = np.array(chunk, dtype=np.float32).T  # [Ch, n]
                    buf.push(arr)

                win = buf.pop_window()
                while win is not None:
                    yield QortexTimeSeries(
                        data=win,
                        shape=win.shape,
                        axes=["channels", "time"],
                        dtype="float32",
                        units="uV",
                        sampling_frequency_hz=srate,
                        channel_names=ch_names,
                        source_provenance={
                            "source_type": "lsl",
                            "stream_name": info.name(),
                            "window_index": window_idx,
                        },
                    )
                    window_idx += 1
                    win = buf.pop_window()
        finally:
            inlet.close_stream()

    def replay(self, speed: float = 1.0) -> Iterator[QortexData]:
        log.warning("LSL is a live source; replay() falls back to stream()")
        yield from self.stream()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filter_streams(self, streams) -> list:
        if not self._query:
            return list(streams)
        q_type = self._query.get("type", "").lower()
        q_name = self._query.get("name", "").lower()
        result = []
        for s in streams:
            if q_type and q_type.lower() not in s.type().lower():
                continue
            if q_name and q_name.lower() not in s.name().lower():
                continue
            result.append(s)
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_pylsl():
    try:
        import pylsl
        return pylsl
    except ImportError:
        raise ImportError(
            "LSL support requires pylsl. "
            "Install with: pip install 'qortex[lsl]' or pip install pylsl"
        )


def _extract_lsl_channel_names(info) -> list[str]:
    n = info.channel_count()
    try:
        ch = info.desc().child("channels").child("channel")
        names = []
        while ch.empty() is False:
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        if len(names) == n:
            return names
    except Exception:
        pass
    return [f"ch_{i}" for i in range(n)]
