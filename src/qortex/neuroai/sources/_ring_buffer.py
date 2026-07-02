"""Ring buffer for real-time windowing — Python fallback.

When ``qortex_rs`` is available (Rust extension), the Rust implementation is
used automatically via ``get_ring_buffer()``.  This pure-Python version is
the fallback for environments where the Rust extension has not been built.

Public API::

    from qortex.neuroai.sources._ring_buffer import get_ring_buffer, batch_window

    buf = get_ring_buffer(n_channels=64, capacity=8192, window_size=512, step_size=128)
    buf.push(chunk)          # chunk: np.ndarray [n_channels, n_samples]
    win = buf.pop_window()   # np.ndarray [n_channels, window_size] or None
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class RingBuffer:
    """Pure-Python lock-free-style ring buffer for real-time EEG windowing.

    Accumulates incoming sample chunks and yields fixed-length windows with
    configurable step (hop) size.  Single-producer, single-consumer — no
    threading primitives are used, matching the Rust SPSC semantics.

    Parameters
    ----------
    n_channels:
        Number of signal channels.
    capacity:
        Ring buffer capacity in samples (must be >= window_size).
    window_size:
        Number of samples per output window.
    step_size:
        Number of samples to advance after each pop (hop size).
        Use ``step_size == window_size`` for non-overlapping windows.
    """

    def __init__(
        self,
        n_channels: int,
        capacity: int,
        window_size: int,
        step_size: int,
    ) -> None:
        if capacity < window_size:
            raise ValueError(
                f"Ring buffer capacity ({capacity}) must be >= window_size ({window_size})"
            )
        self._n_channels = n_channels
        self._capacity = capacity
        self._window_size = window_size
        self._step_size = step_size
        self._buf = np.zeros((n_channels, capacity), dtype=np.float32)
        self._write = 0          # absolute write cursor
        self._n_buffered = 0     # number of unconsumed samples
        self._overruns = 0       # count of consumer-fell-behind events

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(self, samples: np.ndarray) -> None:
        """Push a chunk of samples into the ring buffer.

        Parameters
        ----------
        samples:
            Array of shape ``[n_channels, n_samples]`` (float32 or auto-cast).
        """
        if samples.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected {self._n_channels} channels, got {samples.shape[0]}"
            )
        samples = np.asarray(samples, dtype=np.float32)
        n = samples.shape[1]
        if n == 0:
            return
        if n > self._capacity:
            # Only the most recent `capacity` samples can be retained.
            samples = samples[:, -self._capacity :]
            n = self._capacity

        # Vectorised wraparound write: at most two contiguous slices instead of
        # one Python-level assignment per sample.
        start = self._write % self._capacity
        first = min(n, self._capacity - start)
        self._buf[:, start : start + first] = samples[:, :first]
        if first < n:
            self._buf[:, : n - first] = samples[:, first:]

        self._write += n
        self._n_buffered += n

        # Overrun: unconsumed samples exceeded capacity, so the oldest samples
        # were just overwritten.  Drop them from the count rather than letting
        # the read cursor point at corrupted data.
        if self._n_buffered > self._capacity:
            self._overruns += 1
            if self._overruns == 1 or self._overruns % 100 == 0:
                log.warning(
                    "RingBuffer overrun (#%d): consumer is not keeping up; "
                    "oldest samples dropped. Increase capacity or consume faster.",
                    self._overruns,
                )
            self._n_buffered = self._capacity

    # ── Read ──────────────────────────────────────────────────────────────────

    def pop_window(self) -> np.ndarray | None:
        """Try to pop one window from the buffer.

        Returns ``None`` if fewer than ``window_size`` samples are available.

        Returns
        -------
        np.ndarray or None
            Shape ``[n_channels, window_size]`` float32, or ``None``.
        """
        if self._n_buffered < self._window_size:
            return None

        # read_pos is the absolute index of the oldest unconsumed sample
        read_pos = (self._write - self._n_buffered) % self._capacity
        end = read_pos + self._window_size
        if end <= self._capacity:
            # Contiguous window — slice directly, no fancy indexing.
            win = self._buf[:, read_pos:end].copy()
        else:
            # Wraps the ring boundary — join the two contiguous halves once.
            first = self._capacity - read_pos
            win = np.concatenate(
                (self._buf[:, read_pos:], self._buf[:, : self._window_size - first]),
                axis=1,
            )
        self._n_buffered -= self._step_size
        return win

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_available(self) -> int:
        """Number of samples currently in the buffer (not yet consumed)."""
        return self._n_buffered

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the buffer and reset all cursors."""
        self._buf.fill(0.0)
        self._write = 0
        self._n_buffered = 0


# ── Factory ───────────────────────────────────────────────────────────────────

def get_ring_buffer(
    n_channels: int,
    capacity: int,
    window_size: int,
    step_size: int,
) -> RingBuffer:
    """Return the Rust ``RingBuffer`` if available, else the Python fallback.

    Parameters
    ----------
    n_channels:
        Number of signal channels.
    capacity:
        Ring buffer capacity in samples.
    window_size:
        Samples per window.
    step_size:
        Hop size between consecutive windows.

    Returns
    -------
    RingBuffer
        Rust implementation when ``qortex_rs`` is importable, else Python.
    """
    try:
        from qortex_rs import RingBuffer as RustRingBuffer  # type: ignore[import]
        log.debug("Using Rust RingBuffer from qortex_rs")
        return RustRingBuffer(n_channels, capacity, window_size, step_size)
    except ImportError:
        log.debug("qortex_rs not available — using Python RingBuffer fallback")
        return RingBuffer(n_channels, capacity, window_size, step_size)


# ── Batch windowing ───────────────────────────────────────────────────────────

def batch_window(
    data: np.ndarray,
    window_size: int,
    step_size: int,
) -> list[np.ndarray]:
    """Split a ``[n_channels, n_samples]`` array into overlapping windows.

    Uses the Rust ``batch_window`` function when ``qortex_rs`` is available,
    otherwise falls back to a pure-NumPy implementation.

    Parameters
    ----------
    data:
        Input array of shape ``[n_channels, n_samples]``.
    window_size:
        Samples per window.
    step_size:
        Hop size between consecutive windows.

    Returns
    -------
    list of np.ndarray
        Each element has shape ``[n_channels, window_size]`` (float32).
    """
    try:
        from qortex_rs import batch_window as _rs_batch_window  # type: ignore[import]
        return _rs_batch_window(data.astype(np.float32), window_size, step_size)
    except ImportError:
        pass

    # Pure-NumPy fallback
    data = np.asarray(data, dtype=np.float32)
    n_samples = data.shape[1]
    windows: list[np.ndarray] = []
    start = 0
    while start + window_size <= n_samples:
        windows.append(data[:, start : start + window_size].copy())
        start += step_size
    return windows
