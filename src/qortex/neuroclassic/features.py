"""Epoch-level classical feature extraction for EEG/MEG-style signals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from qortex.neuroclassic._base import (
    MethodConfidence,
    MetricResult,
    NeuroClassicResult,
    NeuroClassicSpec,
)
from qortex.neuroclassic.infoth import _higuchi_fd_1d, _shannon_entropy_bits

__version__ = "0.1.0"

DEFAULT_EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

SLEEP_EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

SEIZURE_EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "hfo": (80.0, 120.0),
}

_SPEC_EPOCH_FEATURES = NeuroClassicSpec(
    method_name="epoch_feature_matrix",
    modality="eeg",
    target_workflow="convert,train,neuroai-run",
    required_evidence=["epochs_array", "sampling_frequency"],
    optional_evidence=["channel_names", "frequency_bands"],
    assumptions=[
        "Epochs are [n_epochs, n_channels, n_times].",
        "Features are computed independently per epoch and channel.",
        "Feature extraction is deterministic and carries feature names for downstream artifacts.",
    ],
    invalid_input_states=[
        "Empty epochs array",
        "Zero or negative sampling frequency",
        "Non-finite samples",
        "Frequency band outside Nyquist",
    ],
)


@dataclass
class EpochFeatureReport:
    """Named feature matrix for epoch-based ML workflows."""

    scope: str
    features: np.ndarray
    feature_names: list[str]
    sampling_frequency_hz: float
    bands: dict[str, tuple[float, float]]
    channel_names: list[str]
    families: list[str]
    n_epochs: int
    n_channels: int
    n_times: int
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.features.shape[0]), int(self.features.shape[1]))

    def to_result(self) -> NeuroClassicResult:
        return NeuroClassicResult(
            method_name="epoch_feature_matrix",
            method_version=__version__,
            modality="eeg",
            scope=self.scope,
            inputs={
                "n_epochs": self.n_epochs,
                "n_channels": self.n_channels,
                "n_times": self.n_times,
            },
            parameters={
                "sampling_frequency_hz": self.sampling_frequency_hz,
                "bands": self.bands,
                "families": self.families,
            },
            assumptions=_SPEC_EPOCH_FEATURES.assumptions,
            metrics=[
                MetricResult("feature_shape", list(self.shape)),
                MetricResult("n_features", len(self.feature_names)),
                MetricResult("feature_names", self.feature_names),
            ],
            warnings=self.warnings,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "epoch_feature_matrix", "version": __version__},
        )

    def to_dict(self, include_features: bool = False) -> dict:
        data = {
            "scope": self.scope,
            "shape": list(self.shape),
            "feature_names": self.feature_names,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "bands": self.bands,
            "channel_names": self.channel_names,
            "families": self.families,
            "n_epochs": self.n_epochs,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "warnings": self.warnings,
            "confidence": self.confidence.value,
        }
        if include_features:
            data["features"] = self.features.tolist()
        return data


def compute_epoch_feature_matrix(
    epochs: np.ndarray,
    *,
    sampling_frequency_hz: float,
    channel_names: list[str] | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    include_relative_bandpower: bool = True,
    include_log_bandpower: bool = True,
    include_time_domain: bool = True,
    include_entropy: bool = False,
    include_higuchi: bool = False,
    hfd_k_max: int = 8,
    scope: str = "unknown",
) -> EpochFeatureReport:
    """Compute a named feature matrix from epochs.

    The default feature set is deliberately conservative: absolute, relative,
    and log bandpower plus simple time-domain statistics.  Entropy and Higuchi
    fractal dimension are opt-in because they are more expensive and should be
    chosen intentionally for the workflow.
    """
    t0 = time.perf_counter()
    x = np.asarray(epochs)
    if x.ndim != 3:
        raise ValueError(f"epochs must be [n_epochs, n_channels, n_times]; got {x.shape}")
    if sampling_frequency_hz <= 0 or math.isnan(sampling_frequency_hz):
        raise ValueError(f"sampling_frequency_hz must be > 0; got {sampling_frequency_hz}")
    if not np.isfinite(x).all():
        raise ValueError("epochs must contain only finite values")
    n_epochs, n_channels, n_times = x.shape
    if n_epochs == 0 or n_channels == 0 or n_times < 2:
        raise ValueError(f"epochs must be non-empty with at least two time samples; got {x.shape}")
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_channels}")
    bands = dict(DEFAULT_EEG_BANDS if bands is None else bands)
    _validate_bands(bands, sampling_frequency_hz)
    if include_higuchi and hfd_k_max < 2:
        raise ValueError(f"hfd_k_max must be >= 2; got {hfd_k_max}")

    feature_names = _feature_names(
        channel_names=channel_names,
        bands=bands,
        include_relative_bandpower=include_relative_bandpower,
        include_log_bandpower=include_log_bandpower,
        include_time_domain=include_time_domain,
        include_entropy=include_entropy,
        include_higuchi=include_higuchi,
    )
    rows = np.empty((n_epochs, len(feature_names)), dtype=np.float32)
    warnings: list[str] = []
    for epoch_i in range(n_epochs):
        rows[epoch_i] = _epoch_features(
            x[epoch_i].astype(np.float64, copy=False),
            sampling_frequency_hz=sampling_frequency_hz,
            bands=bands,
            include_relative_bandpower=include_relative_bandpower,
            include_log_bandpower=include_log_bandpower,
            include_time_domain=include_time_domain,
            include_entropy=include_entropy,
            include_higuchi=include_higuchi,
            hfd_k_max=hfd_k_max,
        )

    families = ["absolute_bandpower"]
    if include_relative_bandpower:
        families.append("relative_bandpower")
    if include_log_bandpower:
        families.append("log_bandpower")
    if include_time_domain:
        families.append("time_domain")
    if include_entropy:
        families.append("spectral_entropy")
    if include_higuchi:
        families.append("higuchi_fractal_dimension")

    confidence = MethodConfidence.HIGH
    if n_times < int(sampling_frequency_hz):
        confidence = MethodConfidence.LOW_CONFIDENCE
        warnings.append("Epochs are shorter than 1 second; spectral features are low-confidence.")

    return EpochFeatureReport(
        scope=scope,
        features=rows,
        feature_names=feature_names,
        sampling_frequency_hz=sampling_frequency_hz,
        bands=bands,
        channel_names=channel_names,
        families=families,
        n_epochs=n_epochs,
        n_channels=n_channels,
        n_times=n_times,
        runtime_s=time.perf_counter() - t0,
        warnings=warnings,
        confidence=confidence,
    )


def compute_bandpower_features(
    epochs: np.ndarray,
    *,
    sampling_frequency_hz: float,
    channel_names: list[str] | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = False,
    log_transform: bool = False,
    scope: str = "unknown",
) -> EpochFeatureReport:
    """Compute only bandpower features with the same report contract."""
    return compute_epoch_feature_matrix(
        epochs,
        sampling_frequency_hz=sampling_frequency_hz,
        channel_names=channel_names,
        bands=bands,
        include_relative_bandpower=relative,
        include_log_bandpower=log_transform,
        include_time_domain=False,
        include_entropy=False,
        include_higuchi=False,
        scope=scope,
    )


def _validate_bands(bands: dict[str, tuple[float, float]], sfreq: float) -> None:
    nyquist = sfreq / 2.0
    if not bands:
        raise ValueError("at least one frequency band is required")
    for name, band in bands.items():
        lo, hi = band
        if not (0 <= lo < hi <= nyquist):
            raise ValueError(
                f"band {name!r} must satisfy 0 <= low < high <= Nyquist ({nyquist}); got {band}"
            )


def _feature_names(
    *,
    channel_names: list[str],
    bands: dict[str, tuple[float, float]],
    include_relative_bandpower: bool,
    include_log_bandpower: bool,
    include_time_domain: bool,
    include_entropy: bool,
    include_higuchi: bool,
) -> list[str]:
    names: list[str] = []
    for ch in channel_names:
        for band_name in bands:
            names.append(f"{ch}.bandpower.{band_name}")
            if include_relative_bandpower:
                names.append(f"{ch}.relative_bandpower.{band_name}")
            if include_log_bandpower:
                names.append(f"{ch}.log_bandpower.{band_name}")
        if include_time_domain:
            names.extend([
                f"{ch}.mean",
                f"{ch}.std",
                f"{ch}.ptp",
                f"{ch}.rms",
                f"{ch}.zero_crossing_rate",
            ])
        if include_entropy:
            names.append(f"{ch}.spectral_entropy")
        if include_higuchi:
            names.append(f"{ch}.higuchi_fractal_dimension")
    return names


def _epoch_features(
    epoch: np.ndarray,
    *,
    sampling_frequency_hz: float,
    bands: dict[str, tuple[float, float]],
    include_relative_bandpower: bool,
    include_log_bandpower: bool,
    include_time_domain: bool,
    include_entropy: bool,
    include_higuchi: bool,
    hfd_k_max: int,
) -> np.ndarray:
    values: list[float] = []
    freqs, psd = _welch_psd(epoch, sampling_frequency_hz)
    total_power = np.trapezoid(psd, freqs, axis=1)
    total_power = np.where(total_power > 0, total_power, np.finfo(np.float64).tiny)
    for ch_i, channel in enumerate(epoch):
        for lo, hi in bands.values():
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                power = 0.0
            else:
                power = float(np.trapezoid(psd[ch_i, mask], freqs[mask]))
            values.append(power)
            if include_relative_bandpower:
                values.append(float(power / total_power[ch_i]))
            if include_log_bandpower:
                values.append(float(np.log10(max(power, np.finfo(np.float64).tiny))))
        if include_time_domain:
            values.extend([
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.ptp(channel)),
                float(np.sqrt(np.mean(channel ** 2))),
                _zero_crossing_rate(channel),
            ])
        if include_entropy:
            values.append(float(_shannon_entropy_bits(psd[ch_i]) or 0.0))
        if include_higuchi:
            hfd, _n_scales = _higuchi_fd_1d(channel, k_max=hfd_k_max)
            values.append(float(hfd or 0.0))
    return np.asarray(values, dtype=np.float32)


def _welch_psd(epoch: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    n_times = epoch.shape[1]
    nperseg = min(max(16, int(2 * sfreq)), n_times)
    try:
        from scipy.signal import welch
        freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=1)
        return freqs, psd
    except ImportError:
        centered = epoch - epoch.mean(axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
        window = np.hanning(n_times)
        scale = sfreq * np.sum(window ** 2)
        spectrum = np.fft.rfft(centered * window, axis=1)
        psd = (np.abs(spectrum) ** 2) / max(scale, np.finfo(np.float64).tiny)
        if psd.shape[1] > 2:
            psd[:, 1:-1] *= 2.0
        return freqs, psd


def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    signs = np.signbit(x)
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / (x.size - 1))
