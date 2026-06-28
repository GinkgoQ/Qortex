"""Information-theoretic signal diagnostics.

Acceptable per AGENTS_2.md § Information Theory and Dynamical Systems Policy:
  - Spectral entropy as a QC signal descriptor
  - Autocorrelation as a temporal redundancy diagnostic

These methods produce numerical evidence — no clinical interpretations.

Algorithms
----------
SpectralEntropyReport
    Shannon entropy (log₂, in bits) of the Welch-estimated normalised PSD.
    H = -Σ p_i log₂(p_i), where p_i = PSD_i / Σ PSD.
    Higher H → more broadband (complex) spectrum.
    Lower H → narrow-band or noise-dominated.

AutocorrelationReport
    Per-channel normalised autocorrelation function (ACF) up to max_lag_s seconds.
    Reports:
      lag1       — lag-1 ACF coefficient (temporal redundancy proxy)
      decay_ms   — half-life: first lag where ACF < 0.5 (if reached)
      max_acf    — maximum ACF value (excluding lag 0)
      is_white_noise — True if lag-1 ACF ≈ 0 (|lag1| < 1/sqrt(n))

Requires numpy.  scipy optional (for Welch PSD; falls back to FFT-based estimate).

Install extras:
    pip install 'qortex[neuroclassic]'
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qortex.neuroclassic._base import (
    MethodConfidence,
    MetricResult,
    NeuroClassicResult,
    NeuroClassicSpec,
    _timer,
)

__version__ = "0.1.0"

_SPEC_SE = NeuroClassicSpec(
    method_name="spectral_entropy",
    modality="eeg",
    target_workflow="visualize,convert,train",
    required_evidence=["data_array", "sampling_frequency"],
    optional_evidence=["channel_names"],
    assumptions=[
        "Data is [n_channels, n_times] layout.",
        "Sampling frequency is positive and constant.",
        "Spectral entropy is a QC signal descriptor, not a clinical measure.",
    ],
    invalid_input_states=["Zero or negative sampling frequency", "Empty data array"],
)

_SPEC_ACF = NeuroClassicSpec(
    method_name="autocorrelation_summary",
    modality="eeg",
    target_workflow="visualize,convert,train",
    required_evidence=["data_array", "sampling_frequency"],
    optional_evidence=["channel_names"],
    assumptions=[
        "Data is [n_channels, n_times] layout.",
        "Autocorrelation is computed on finite samples only.",
        "Reported half-life uses 0.5 as the ACF threshold — this is a temporal redundancy diagnostic, not a Lyapunov stability estimate.",
    ],
    invalid_input_states=["Constant channel (variance = 0)", "< 10 samples"],
)


@dataclass
class ChannelSpectralEntropy:
    """Spectral entropy for one channel."""
    name: str
    index: int
    spectral_entropy_bits: float | None  # higher = more broadband
    n_psd_bins: int | None = None
    low_confidence: bool = False         # True when recording is very short

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "spectral_entropy_bits": self.spectral_entropy_bits,
            "n_psd_bins": self.n_psd_bins,
            "low_confidence": self.low_confidence,
        }


@dataclass
class SpectralEntropyReport:
    """Spectral entropy across all channels.

    H_max = log₂(n_psd_bins) — maximum entropy for uniform PSD.
    Relative entropy = H / H_max is reported alongside absolute H.
    """
    scope: str
    n_channels: int
    n_times: int
    sampling_frequency_hz: float
    channels: list[ChannelSpectralEntropy] = field(default_factory=list)
    mean_entropy: float | None = None
    std_entropy: float | None = None
    h_max: float | None = None           # theoretical maximum for this PSD resolution
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_result(self) -> NeuroClassicResult:
        metrics = [
            MetricResult("n_channels", self.n_channels),
            MetricResult("mean_spectral_entropy_bits", self.mean_entropy, unit="bits",
                         interpretation="Higher = more broadband signal spectrum"),
            MetricResult("std_spectral_entropy_bits", self.std_entropy, unit="bits"),
            MetricResult("h_max_bits", self.h_max, unit="bits",
                         interpretation="log₂(n_psd_bins) — max achievable entropy"),
        ]
        for ch in self.channels:
            metrics.append(MetricResult(
                f"spectral_entropy.{ch.name}",
                ch.spectral_entropy_bits,
                unit="bits",
            ))
        return NeuroClassicResult(
            method_name="spectral_entropy",
            method_version=__version__,
            modality="eeg",
            scope=self.scope,
            inputs={"n_channels": self.n_channels, "n_times": self.n_times},
            parameters={"sampling_frequency_hz": self.sampling_frequency_hz},
            assumptions=_SPEC_SE.assumptions,
            metrics=metrics,
            warnings=self.warnings,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "spectral_entropy", "version": __version__},
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_channels": self.n_channels,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "mean_entropy_bits": self.mean_entropy,
            "std_entropy_bits": self.std_entropy,
            "h_max_bits": self.h_max,
            "channels": [c.to_dict() for c in self.channels],
            "warnings": self.warnings,
            "confidence": self.confidence.value,
        }


@dataclass
class ChannelAutocorrelation:
    """Autocorrelation summary for one channel."""
    name: str
    index: int
    lag1: float | None                   # lag-1 ACF coefficient
    decay_ms: float | None               # first lag (ms) where ACF < 0.5
    max_acf: float | None                # maximum ACF at lags > 0
    is_white_noise: bool = False         # |lag1| < 1/sqrt(n)
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "lag1": self.lag1,
            "decay_ms": self.decay_ms,
            "max_acf": self.max_acf,
            "is_white_noise": self.is_white_noise,
            "low_confidence": self.low_confidence,
        }


@dataclass
class AutocorrelationReport:
    """Autocorrelation summary across all channels.

    Temporal redundancy diagnostic — not a stability or complexity measure.
    """
    scope: str
    n_channels: int
    n_times: int
    sampling_frequency_hz: float
    max_lag_s: float
    channels: list[ChannelAutocorrelation] = field(default_factory=list)
    n_high_autocorr: int = 0             # channels with |lag1| > 0.9
    mean_lag1: float | None = None
    mean_decay_ms: float | None = None
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_result(self) -> NeuroClassicResult:
        metrics = [
            MetricResult("n_channels", self.n_channels),
            MetricResult("mean_lag1_acf", self.mean_lag1,
                         interpretation="Mean lag-1 ACF; near 1 = strongly autocorrelated"),
            MetricResult("mean_acf_decay_ms", self.mean_decay_ms, unit="ms",
                         interpretation="Mean time for ACF to drop below 0.5"),
            MetricResult("n_high_autocorr_channels", self.n_high_autocorr,
                         interpretation="Channels with |lag1| > 0.9 (potentially redundant)"),
        ]
        for ch in self.channels:
            metrics.append(MetricResult(f"lag1.{ch.name}", ch.lag1))
            metrics.append(MetricResult(f"decay_ms.{ch.name}", ch.decay_ms, unit="ms"))
        return NeuroClassicResult(
            method_name="autocorrelation_summary",
            method_version=__version__,
            modality="eeg",
            scope=self.scope,
            inputs={"n_channels": self.n_channels, "n_times": self.n_times},
            parameters={
                "sampling_frequency_hz": self.sampling_frequency_hz,
                "max_lag_s": self.max_lag_s,
            },
            assumptions=_SPEC_ACF.assumptions,
            metrics=metrics,
            warnings=self.warnings,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "autocorrelation_summary", "version": __version__},
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_channels": self.n_channels,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "max_lag_s": self.max_lag_s,
            "mean_lag1": self.mean_lag1,
            "mean_decay_ms": self.mean_decay_ms,
            "n_high_autocorr": self.n_high_autocorr,
            "channels": [c.to_dict() for c in self.channels],
            "warnings": self.warnings,
            "confidence": self.confidence.value,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def compute_spectral_entropy(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    channel_names: list[str] | None = None,
    scope: str = "unknown",
    nperseg_s: float = 4.0,
) -> SpectralEntropyReport:
    """Compute spectral entropy per channel from a 2-D signal array.

    Spectral entropy H = -Σ p_i log₂(p_i) where p_i is the fraction of
    total PSD power in frequency bin i.  Uses Welch method when scipy is
    available; falls back to FFT-magnitude-squared estimate otherwise.

    Parameters
    ----------
    data:
        [n_channels, n_times] float array.
    sampling_frequency_hz:
        Sampling rate; must be > 0.
    channel_names:
        Optional channel labels.
    scope:
        Recording identifier.
    nperseg_s:
        Welch segment length in seconds.

    Returns
    -------
    SpectralEntropyReport
    """
    t0 = time.perf_counter()

    if data.ndim != 2:
        raise ValueError(f"data must be [n_channels, n_times]; got {data.shape}")
    if sampling_frequency_hz <= 0:
        raise ValueError(f"sampling_frequency_hz must be > 0; got {sampling_frequency_hz}")

    n_ch, n_t = data.shape
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_ch)]
    if len(channel_names) != n_ch:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_ch}")

    report = SpectralEntropyReport(
        scope=scope,
        n_channels=n_ch,
        n_times=n_t,
        sampling_frequency_hz=sampling_frequency_hz,
    )

    if n_t == 0:
        report.confidence = MethodConfidence.UNKNOWN
        report.warnings.append("Empty data array.")
        return report

    if n_t < int(sampling_frequency_hz):
        report.confidence = MethodConfidence.LOW_CONFIDENCE
        report.warnings.append(
            "Recording is shorter than 1 second; spectral entropy estimates are low-confidence."
        )

    nperseg = min(int(nperseg_s * sampling_frequency_hz), n_t)
    h_max: float | None = None

    channel_entropies: list[float] = []
    channels: list[ChannelSpectralEntropy] = []

    for i, ch_name in enumerate(channel_names):
        row = data[i]
        finite = row[np.isfinite(row)]
        if len(finite) < max(16, nperseg):
            channels.append(ChannelSpectralEntropy(
                name=ch_name, index=i,
                spectral_entropy_bits=None,
                low_confidence=True,
            ))
            continue

        psd, n_bins = _estimate_psd(finite, sampling_frequency_hz, nperseg)
        if h_max is None and n_bins > 0:
            h_max = math.log2(n_bins)

        se = _shannon_entropy_bits(psd)
        channel_entropies.append(se) if se is not None else None
        channels.append(ChannelSpectralEntropy(
            name=ch_name,
            index=i,
            spectral_entropy_bits=se,
            n_psd_bins=n_bins,
            low_confidence=(n_t < int(sampling_frequency_hz)),
        ))

    report.channels = channels
    report.h_max = h_max
    if channel_entropies:
        report.mean_entropy = sum(channel_entropies) / len(channel_entropies)
        if len(channel_entropies) > 1:
            mean_h = report.mean_entropy
            report.std_entropy = math.sqrt(
                sum((h - mean_h) ** 2 for h in channel_entropies) / (len(channel_entropies) - 1)
            )
    report.runtime_s = time.perf_counter() - t0
    return report


def compute_autocorrelation_summary(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    channel_names: list[str] | None = None,
    scope: str = "unknown",
    max_lag_s: float = 1.0,
    white_noise_threshold: float | None = None,
) -> AutocorrelationReport:
    """Compute per-channel autocorrelation summary.

    Reports lag-1 ACF, ACF half-life, and white-noise flag for each channel.
    This is a temporal redundancy diagnostic — used to detect highly
    autocorrelated channels that may violate i.i.d. assumptions in ML pipelines.

    Parameters
    ----------
    data:
        [n_channels, n_times] float array.
    sampling_frequency_hz:
        Sampling rate; must be > 0.
    channel_names:
        Optional channel labels.
    scope:
        Recording identifier.
    max_lag_s:
        Maximum lag (seconds) to search for ACF half-life.
    white_noise_threshold:
        |lag1| below this → is_white_noise=True.
        Defaults to 1/sqrt(n_times) (95% CI for white noise).

    Returns
    -------
    AutocorrelationReport
    """
    t0 = time.perf_counter()

    if data.ndim != 2:
        raise ValueError(f"data must be [n_channels, n_times]; got {data.shape}")
    if sampling_frequency_hz <= 0:
        raise ValueError(f"sampling_frequency_hz must be > 0; got {sampling_frequency_hz}")

    n_ch, n_t = data.shape
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_ch)]

    wn_thresh = white_noise_threshold if white_noise_threshold is not None else (
        1.0 / math.sqrt(n_t) if n_t > 0 else 0.1
    )

    report = AutocorrelationReport(
        scope=scope,
        n_channels=n_ch,
        n_times=n_t,
        sampling_frequency_hz=sampling_frequency_hz,
        max_lag_s=max_lag_s,
    )

    if n_t == 0:
        report.confidence = MethodConfidence.UNKNOWN
        report.warnings.append("Empty data array.")
        return report

    if n_t < int(sampling_frequency_hz):
        report.confidence = MethodConfidence.LOW_CONFIDENCE
        report.warnings.append(
            "Recording is shorter than 1 second; autocorrelation estimates are low-confidence."
        )

    channels: list[ChannelAutocorrelation] = []
    lag1_vals: list[float] = []
    decay_vals: list[float] = []
    max_lag = min(int(max_lag_s * sampling_frequency_hz), n_t - 1)
    n_high = 0

    for i, ch_name in enumerate(channel_names):
        row = data[i]
        finite = row[np.isfinite(row)]
        if len(finite) < 10:
            channels.append(ChannelAutocorrelation(
                name=ch_name, index=i,
                lag1=None, decay_ms=None, max_acf=None,
                low_confidence=True,
            ))
            continue

        centered = finite - finite.mean()
        var = float(np.mean(centered ** 2))
        if var == 0.0:
            channels.append(ChannelAutocorrelation(
                name=ch_name, index=i,
                lag1=None, decay_ms=None, max_acf=None,
                low_confidence=True,
            ))
            continue

        n_f = len(finite)
        lag1 = float(np.mean(centered[:-1] * centered[1:])) / var

        # Compute ACF up to max_lag; find half-life
        decay_ms: float | None = None
        max_acf: float | None = None
        for lag in range(2, min(max_lag, n_f - 1) + 1):
            acf_val = float(np.mean(centered[:n_f - lag] * centered[lag:])) / var
            if max_acf is None or acf_val > max_acf:
                max_acf = acf_val
            if decay_ms is None and acf_val <= 0.5:
                decay_ms = lag / sampling_frequency_hz * 1000.0

        is_wn = abs(lag1) < wn_thresh
        is_high = abs(lag1) > 0.9
        if is_high:
            n_high += 1

        if lag1_vals is not None:
            lag1_vals.append(lag1)
        if decay_ms is not None:
            decay_vals.append(decay_ms)

        channels.append(ChannelAutocorrelation(
            name=ch_name, index=i,
            lag1=lag1,
            decay_ms=decay_ms,
            max_acf=max_acf,
            is_white_noise=is_wn,
            low_confidence=(n_t < int(sampling_frequency_hz)),
        ))

    report.channels = channels
    report.n_high_autocorr = n_high
    if lag1_vals:
        report.mean_lag1 = sum(lag1_vals) / len(lag1_vals)
    if decay_vals:
        report.mean_decay_ms = sum(decay_vals) / len(decay_vals)

    if n_high > 0:
        report.warnings.append(
            f"{n_high} channels have |lag-1 ACF| > 0.9 — strong temporal autocorrelation. "
            "Sliding-window training may inflate effective sample size."
        )

    report.runtime_s = time.perf_counter() - t0
    return report


# ── PSD helpers ───────────────────────────────────────────────────────────────

def _estimate_psd(row: np.ndarray, sfreq: float, nperseg: int) -> tuple[np.ndarray, int]:
    """Estimate PSD via Welch (scipy) or FFT magnitude-squared fallback."""
    try:
        from scipy.signal import welch
        _, psd = welch(row, fs=sfreq, nperseg=nperseg)
        return psd, len(psd)
    except ImportError:
        pass
    # FFT fallback
    fft_mag = np.abs(np.fft.rfft(row)) ** 2
    return fft_mag, len(fft_mag)


def _shannon_entropy_bits(psd: np.ndarray) -> float | None:
    """Shannon entropy (log₂ / bits) of a normalised PSD."""
    psd_pos = psd[psd > 0]
    if psd_pos.size == 0:
        return None
    p = psd_pos / psd_pos.sum()
    with np.errstate(divide="ignore"):
        return float(-np.sum(p * np.log2(p)))
