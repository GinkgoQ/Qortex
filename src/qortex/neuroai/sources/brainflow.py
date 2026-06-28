"""BrainFlow source adapter.

Probes and streams from BrainFlow-compatible boards (g.tec, OpenBCI, Muse,
Synthetic, etc.) using the brainflow Python SDK.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    ChannelSpec,
    EvidenceStatus,
    QortexTimeSeries,
    SourceProfile,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)


class BrainFlowAdapter(SourceAdapter):
    """Source adapter for BrainFlow-compatible acquisition boards.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="brainflow"``.
        ``spec.extra`` may contain:
        - ``board_id``: int board ID (default: ``-1`` = synthetic)
        - ``serial_port``: str
        - ``mac_address``: str
        - ``ip_address``: str
        - ``ip_port``: int
        - ``other_info``: str
    window_spec:
        Optional windowing.
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
        self._extra = spec.extra or {}

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        BrainFlowInputParams, BoardShim, BoardIds = _require_brainflow()
        board_id = int(self._extra.get("board_id", BoardIds.SYNTHETIC_BOARD.value))

        try:
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            srate = float(BoardShim.get_sampling_rate(board_id))
            ch_names = list(BoardShim.get_eeg_names(board_id))
        except Exception as exc:
            log.warning("BrainFlow metadata query failed: %s", exc)
            eeg_channels = list(range(8))
            srate = 256.0
            ch_names = [f"ch_{i}" for i in range(8)]

        n_channels = len(eeg_channels)
        channel_specs = [ChannelSpec(name=n, index=i, unit="uV") for i, n in enumerate(ch_names)]

        return SourceProfile(
            source_id=f"brainflow:board_{board_id}",
            source_type="brainflow",
            modality="eeg",
            n_channels=n_channels,
            sampling_rate_hz=srate,
            channel_names=ch_names,
            channel_specs=channel_specs,
            dtype="float64",
            axis_convention=AxisConvention.channels_time,
            path=None,
            extra={
                "board_id": board_id,
                "eeg_channel_indices": eeg_channels,
            },
            evidence={
                "n_channels": EvidenceStatus.confirmed,
                "sampling_rate": EvidenceStatus.confirmed,
                "channel_names": EvidenceStatus.confirmed,
            },
        )

    def read_batch(self) -> list[QortexData]:
        BrainFlowInputParams, BoardShim, BoardIds = _require_brainflow()
        duration_s = float(self._extra.get("duration_s", 5.0))
        board_id = int(self._extra.get("board_id", BoardIds.SYNTHETIC_BOARD.value))
        params = self._make_params(BrainFlowInputParams)

        board = BoardShim(board_id, params)
        BoardShim.disable_board_logger()

        board.prepare_session()
        board.start_stream()
        time.sleep(duration_s)
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_data = data[eeg_channels, :].astype(np.float32)
        srate = float(BoardShim.get_sampling_rate(board_id))
        ch_names = list(BoardShim.get_eeg_names(board_id))

        return [QortexTimeSeries(
            data=eeg_data,
            shape=eeg_data.shape,
            axes=["channels", "time"],
            dtype="float32",
            units="uV",
            sampling_frequency_hz=srate,
            channel_names=ch_names,
            source_provenance={"source_type": "brainflow", "board_id": board_id},
        )]

    def stream(self) -> Iterator[QortexData]:
        BrainFlowInputParams, BoardShim, BoardIds = _require_brainflow()
        board_id = int(self._extra.get("board_id", BoardIds.SYNTHETIC_BOARD.value))
        params = self._make_params(BrainFlowInputParams)

        srate = float(BoardShim.get_sampling_rate(board_id))
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        ch_names = list(BoardShim.get_eeg_names(board_id))
        n_channels = len(eeg_channels)

        win_dur = self._window_spec.duration_s if self._window_spec else 1.0
        step_dur = getattr(self._window_spec, "step_s", win_dur) if self._window_spec else win_dur
        win_samples = max(1, int(win_dur * srate))
        step_samples = max(1, int(step_dur * srate))

        from qortex.neuroai.sources._ring_buffer import get_ring_buffer
        buf = get_ring_buffer(
            n_channels=n_channels,
            capacity=win_samples * 8,
            window_size=win_samples,
            step_size=step_samples,
        )

        board = BoardShim(board_id, params)
        BoardShim.disable_board_logger()
        board.prepare_session()
        board.start_stream()
        log.info("BrainFlow: streaming from board_id=%d at %.1f Hz", board_id, srate)

        window_idx = 0
        try:
            while True:
                time.sleep(win_dur / 4)
                raw = board.get_board_data(win_samples)
                if raw.shape[1] == 0:
                    continue
                eeg = raw[eeg_channels, :].astype(np.float32)
                buf.push(eeg)

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
                            "source_type": "brainflow",
                            "board_id": board_id,
                            "window_index": window_idx,
                        },
                    )
                    window_idx += 1
                    win = buf.pop_window()
        finally:
            board.stop_stream()
            board.release_session()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_params(self, BrainFlowInputParams):
        params = BrainFlowInputParams()
        params.serial_port = str(self._extra.get("serial_port", ""))
        params.mac_address = str(self._extra.get("mac_address", ""))
        params.ip_address = str(self._extra.get("ip_address", ""))
        params.ip_port = int(self._extra.get("ip_port", 0))
        params.other_info = str(self._extra.get("other_info", ""))
        return params


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_brainflow():
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        return BrainFlowInputParams, BoardShim, BoardIds
    except ImportError:
        raise ImportError(
            "BrainFlow support requires brainflow. "
            "Install with: pip install 'qortex[brainflow]' or pip install brainflow"
        )
