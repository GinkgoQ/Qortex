"""Regression tests for the NeuroAI preprocessing hot path.

Covers the numerically sensitive and performance-critical transforms in
``qortex.neuroai.preprocess.planner`` and the streaming ring buffer.  These
guard the correctness of the optimised implementations (vectorised EMA
standardisation, centred pad/crop, rational-ratio resampling, cached bandpass
design) against future changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from qortex.neuroai.contracts import (
    PreprocessPlan,
    TransformDescriptor,
    TransformKind,
)
from qortex.neuroai.preprocess.planner import (
    TransformError,
    TransformExecutor,
    _exponential_moving_standardize,
    _pad_or_crop,
)
from qortex.neuroai.sources._ring_buffer import RingBuffer


# ── Exponential moving standardization ──────────────────────────────────────────

def _reference_ema(arr, factor_new, init_block_size, eps):
    """Naive per-element recurrence — the definition the vectorised code replaces."""
    out = np.empty_like(arr, dtype=np.float32)
    flat = arr.reshape((-1, arr.shape[-1])).astype(np.float32, copy=False)
    out_flat = out.reshape((-1, arr.shape[-1]))
    for row_idx, row in enumerate(flat):
        n_init = max(1, min(init_block_size, row.shape[0]))
        mean = float(row[:n_init].mean())
        var = float(row[:n_init].var())
        for t, value in enumerate(row):
            if t >= n_init:
                mean = (1.0 - factor_new) * mean + factor_new * float(value)
                diff = float(value) - mean
                var = (1.0 - factor_new) * var + factor_new * diff * diff
            out_flat[row_idx, t] = (float(value) - mean) / max(var ** 0.5, eps)
    return out


def test_ema_matches_reference_recurrence():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 400)).astype(np.float32)
    fast = _exponential_moving_standardize(x, factor_new=0.001, init_block_size=100, eps=1e-4)
    slow = _reference_ema(x, 0.001, 100, 1e-4)
    assert fast.shape == x.shape
    # float32 accumulation noise only — not an algorithmic difference.
    np.testing.assert_allclose(fast, slow, atol=1e-4)


def test_ema_preserves_input_shape():
    x = np.random.default_rng(1).standard_normal((8, 250)).astype(np.float32)
    out = _exponential_moving_standardize(x, factor_new=0.01, init_block_size=50, eps=1e-6)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


# ── Centred pad / crop ──────────────────────────────────────────────────────────

def test_pad_or_crop_is_centred_when_cropping():
    v = np.arange(10 * 10 * 10, dtype=np.float32).reshape(10, 10, 10)
    cropped = _pad_or_crop(v, (6, 6, 6))
    # 10 -> 6 keeps the centre: offset (10-6)//2 == 2
    np.testing.assert_array_equal(cropped, v[2:8, 2:8, 2:8])


def test_pad_or_crop_is_centred_when_padding():
    v = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    padded = _pad_or_crop(v, (8, 8, 8))
    assert padded.shape == (8, 8, 8)
    # original sits in the centre; borders are zero-padded
    np.testing.assert_array_equal(padded[2:6, 2:6, 2:6], v)
    assert padded[0, 0, 0] == 0.0


def test_pad_or_crop_rejects_dim_mismatch():
    v = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(TransformError):
        _pad_or_crop(v, (4, 4))


# ── Resample precision ──────────────────────────────────────────────────────────

def _resample(from_hz, to_hz, n=5120, channels=4):
    plan = PreprocessPlan(transforms=[TransformDescriptor(
        kind=TransformKind.resample, required_by="t",
        params={"from_hz": from_hz, "to_hz": to_hz},
    )])
    x = np.random.default_rng(2).standard_normal((channels, n)).astype(np.float32)
    return TransformExecutor(plan).apply(x)


def test_resample_fractional_rate_produces_expected_length():
    pytest.importorskip("scipy")
    out = _resample(512.03, 256.0, n=5120)
    assert out.shape[-1] == round(5120 * 256.0 / 512.03)


def test_resample_noop_when_rates_match():
    pytest.importorskip("scipy")
    out = _resample(250.0, 250.0, n=1000)
    assert out.shape[-1] == 1000


# ── Bandpass filter caching ─────────────────────────────────────────────────────

def test_bandpass_designs_filter_once_across_windows():
    pytest.importorskip("scipy")
    plan = PreprocessPlan(transforms=[TransformDescriptor(
        kind=TransformKind.bandpass, required_by="t",
        params={"low_hz": 1.0, "high_hz": 40.0, "sfreq": 250.0},
    )])
    ex = TransformExecutor(plan)
    x = np.random.default_rng(4).standard_normal((16, 500)).astype(np.float32)
    y1 = ex.apply(x.copy())
    y2 = ex.apply(x.copy())
    assert len(ex._sos_cache) == 1
    np.testing.assert_array_equal(y1, y2)


def test_bandpass_requires_a_band():
    pytest.importorskip("scipy")
    plan = PreprocessPlan(transforms=[TransformDescriptor(
        kind=TransformKind.bandpass, required_by="t", params={"sfreq": 250.0},
    )])
    x = np.random.default_rng(5).standard_normal((4, 200)).astype(np.float32)
    with pytest.raises(TransformError):
        TransformExecutor(plan).apply(x)


# ── Ring buffer ─────────────────────────────────────────────────────────────────

class _RefRing:
    """Per-sample reference used to validate the vectorised RingBuffer."""

    def __init__(self, nc, cap, ws, ss):
        self.cap, self.ws, self.ss = cap, ws, ss
        self.buf = np.zeros((nc, cap), np.float32)
        self.w = 0
        self.nb = 0

    def push(self, x):
        for i in range(x.shape[1]):
            self.buf[:, self.w % self.cap] = x[:, i]
            self.w += 1
        self.nb += x.shape[1]
        if self.nb > self.cap:
            self.nb = self.cap

    def pop(self):
        if self.nb < self.ws:
            return None
        rp = self.w - self.nb
        idx = [(rp + j) % self.cap for j in range(self.ws)]
        win = self.buf[:, idx].copy()
        self.nb -= self.ss
        return win


def test_ring_buffer_matches_reference_with_wraparound():
    rng = np.random.default_rng(3)
    fast = RingBuffer(4, 64, 16, 8)
    slow = _RefRing(4, 64, 16, 8)
    fast_windows, slow_windows = [], []
    for _ in range(60):
        chunk = rng.standard_normal((4, int(rng.integers(1, 20)))).astype(np.float32)
        fast.push(chunk)
        slow.push(chunk)
        wf, ws = fast.pop_window(), slow.pop()
        if wf is not None:
            fast_windows.append(wf)
        if ws is not None:
            slow_windows.append(ws)
    assert len(fast_windows) == len(slow_windows)
    for a, b in zip(fast_windows, slow_windows):
        np.testing.assert_array_equal(a, b)


def test_ring_buffer_caps_available_and_counts_overruns():
    buf = RingBuffer(2, 32, 8, 8)
    # Repeated pushes without popping force the consumer to fall behind.
    rng = np.random.default_rng(6)
    for _ in range(5):
        buf.push(rng.standard_normal((2, 16)).astype(np.float32))
    assert buf.n_available <= 32
    assert buf._overruns > 0


def test_ring_buffer_rejects_wrong_channel_count():
    buf = RingBuffer(4, 64, 16, 8)
    with pytest.raises(ValueError):
        buf.push(np.zeros((3, 10), dtype=np.float32))
