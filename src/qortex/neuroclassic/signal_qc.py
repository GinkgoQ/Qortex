"""Signal quality control for EEG, MEG, iEEG, and fNIRS.

Computes deterministic per-channel and per-recording QC metrics.
Requires: numpy.  Optional: scipy (for Welch PSD, line-noise power).

All metrics are numerical evidence — no clinical interpretation.

Install extras:
    pip install 'qortex[neuroclassic]'
    pip install 'qortex[eeg]'
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroclassic._base import (
    CohortMetricReport,
    MethodConfidence,
    MetricResult,
    NeuroClassicReport,
    NeuroClassicResult,
    NeuroClassicSpec,
    _timer,
)

__version__ = "0.1.0"

_SPEC = NeuroClassicSpec(
    method_name="signal_qc",
    modality="eeg",
    target_workflow="visualize,convert,train,neuroai-run",
    required_evidence=["sampling_frequency", "channel_names", "data_array"],
    optional_evidence=["channel_types", "units", "bad_channels"],
    assumptions=[
        "Data is in [n_channels, n_times] layout.",
        "Sampling frequency is constant across channels.",
        "Data values are in the declared unit (uV for EEG, fT for MEG).",
    ],
    invalid_input_states=[
        "Empty data array",
        "Zero or negative sampling frequency",
        "NaN sampling frequency",
        "Shape mismatch between data and channel_names",
    ],
)


@dataclass
class ChannelQC:
    """Per-channel quality metrics."""
    name: str
    index: int
    is_flatline: bool = False
    flatline_fraction: float = 0.0
    has_nan: bool = False
    nan_fraction: float = 0.0
    has_inf: bool = False
    is_saturated: bool = False
    saturation_fraction: float = 0.0
    peak_to_peak: float | None = None
    robust_variance: float | None = None
    line_noise_power_50hz: float | None = None
    line_noise_power_60hz: float | None = None
    psd_slope: float | None = None
    correlation_outlier: bool = False
    correlation_score: float | None = None  # mean abs correlation with all other channels

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SignalQualityReport:
    """Full signal QC report for one recording.

    Maps onto NeuroClassicResult — use `to_result()` for pipeline integration.
    """
    scope: str                          # file path or subject+session key
    n_channels: int
    n_times: int
    sampling_frequency_hz: float
    duration_s: float
    channel_qc: list[ChannelQC] = field(default_factory=list)
    n_flatline: int = 0
    n_nan: int = 0
    n_inf: int = 0
    n_saturated: int = 0
    n_correlation_outliers: int = 0
    global_nan_fraction: float = 0.0
    global_saturation_fraction: float = 0.0
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_result(self) -> NeuroClassicResult:
        metrics = [
            MetricResult("n_channels", self.n_channels),
            MetricResult("n_times", self.n_times),
            MetricResult("duration_s", self.duration_s, unit="s"),
            MetricResult("n_flatline_channels", self.n_flatline,
                         threshold=0, threshold_source="Qortex default"),
            MetricResult("n_nan_channels", self.n_nan,
                         threshold=0, threshold_source="Qortex default"),
            MetricResult("n_inf_channels", self.n_inf,
                         threshold=0, threshold_source="Qortex default"),
            MetricResult("n_saturated_channels", self.n_saturated,
                         threshold=0, threshold_source="Qortex default"),
            MetricResult("n_correlation_outliers", self.n_correlation_outliers),
            MetricResult("global_nan_fraction", self.global_nan_fraction),
        ]
        per_ch = [MetricResult(f"channel_qc.{ch.name}", ch.to_dict())
                  for ch in self.channel_qc]
        result = NeuroClassicResult(
            method_name="signal_qc",
            method_version=__version__,
            modality="eeg",
            scope=self.scope,
            inputs={"n_channels": self.n_channels, "n_times": self.n_times},
            parameters={"sampling_frequency_hz": self.sampling_frequency_hz},
            assumptions=_SPEC.assumptions,
            metrics=metrics + per_ch,
            warnings=self.warnings,
            blockers=self.blockers,
            unknowns=self.unknowns,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "signal_qc", "version": __version__},
        )
        return result

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "duration_s": self.duration_s,
            "n_flatline": self.n_flatline,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_saturated": self.n_saturated,
            "n_correlation_outliers": self.n_correlation_outliers,
            "global_nan_fraction": self.global_nan_fraction,
            "global_saturation_fraction": self.global_saturation_fraction,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "unknowns": self.unknowns,
            "runtime_s": self.runtime_s,
            "confidence": self.confidence.value,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def compute_signal_qc(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    channel_names: list[str] | None = None,
    scope: str = "unknown",
    flatline_threshold_sd: float = 0.01,
    saturation_pct: float = 99.5,
    compute_psd: bool = True,
    line_noise_hz: float | None = None,
    compute_correlations: bool = True,
    correlation_outlier_threshold: float = 3.0,
) -> SignalQualityReport:
    """Compute per-channel and recording-level QC metrics.

    Parameters
    ----------
    data:
        Signal array, shape [n_channels, n_times] (float).
    sampling_frequency_hz:
        Sampling rate; must be > 0.
    channel_names:
        Optional list of channel names.  Defaults to ch_0, ch_1, …
    scope:
        Identifier for this recording (file path or sub-ses key).
    flatline_threshold_sd:
        A channel is flatline if its std < this fraction of the median std across channels.
    saturation_pct:
        A channel is saturated if it spends > (100 - saturation_pct)% at its min or max value.
    compute_psd:
        Whether to compute spectral metrics (requires scipy).
    line_noise_hz:
        If given, compute line-noise power at this frequency.
    compute_correlations:
        Whether to compute inter-channel correlation outliers.
    correlation_outlier_threshold:
        Number of SDs above the mean absolute correlation to flag as outlier.

    Returns
    -------
    SignalQualityReport
    """
    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D [n_channels, n_times]; got shape {data.shape}")
    if sampling_frequency_hz <= 0 or math.isnan(sampling_frequency_hz):
        raise ValueError(f"Invalid sampling_frequency_hz: {sampling_frequency_hz}")

    n_ch, n_t = data.shape
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_ch)]
    if len(channel_names) != n_ch:
        raise ValueError(
            f"channel_names length {len(channel_names)} != n_channels {n_ch}"
        )

    report = SignalQualityReport(
        scope=scope,
        n_channels=n_ch,
        n_times=n_t,
        sampling_frequency_hz=sampling_frequency_hz,
        duration_s=n_t / sampling_frequency_hz,
    )

    if n_t == 0:
        report.blockers.append("Data array has 0 time samples.")
        report.confidence = MethodConfidence.UNKNOWN
        report.runtime_s = time.perf_counter() - t0
        return report

    # Median std across channels (for flatline detection baseline)
    with np.errstate(invalid="ignore"):
        per_ch_std = np.std(data, axis=1)  # [n_ch]
    med_std = float(np.median(per_ch_std[per_ch_std > 0])) if np.any(per_ch_std > 0) else 1.0
    flatline_abs_threshold = flatline_threshold_sd * med_std

    # Saturation: fraction of samples at phys min or max
    ch_min = data.min(axis=1)  # [n_ch]
    ch_max = data.max(axis=1)

    # PSD (optional)
    psd_results: dict[int, tuple[float | None, float | None, float | None]] = {}
    if compute_psd:
        psd_results = _compute_psd_metrics(data, sampling_frequency_hz, line_noise_hz)

    # Inter-channel correlations (optional)
    corr_scores: np.ndarray | None = None
    corr_mean: float | None = None
    corr_std: float | None = None
    if compute_correlations and n_ch >= 2:
        corr_scores, corr_mean, corr_std = _compute_correlation_outliers(data)

    channel_qcs: list[ChannelQC] = []
    nan_total = 0
    for i, ch in enumerate(channel_names):
        row = data[i]
        nan_mask = ~np.isfinite(row)
        has_nan = bool(np.any(np.isnan(row)))
        has_inf = bool(np.any(np.isinf(row)))
        nan_frac = float(nan_mask.sum() / n_t)
        nan_total += int(nan_mask.sum())

        finite_row = row[~nan_mask] if nan_mask.any() else row
        if len(finite_row) == 0:
            cqc = ChannelQC(name=ch, index=i, has_nan=True, nan_fraction=1.0)
            cqc.is_flatline = True
            channel_qcs.append(cqc)
            continue

        ch_std = float(np.std(finite_row))
        is_flatline = ch_std < flatline_abs_threshold

        # Saturation: samples at physical min or max
        phys_min = float(finite_row.min())
        phys_max = float(finite_row.max())
        if phys_max > phys_min:
            sat_count = int(np.sum((finite_row == phys_min) | (finite_row == phys_max)))
            sat_frac = sat_count / len(finite_row)
            is_sat = sat_frac > (1.0 - saturation_pct / 100.0)
        else:
            sat_frac = 0.0
            is_sat = False

        p2p = phys_max - phys_min
        # Robust variance: MAD-based
        mad = float(np.median(np.abs(finite_row - np.median(finite_row)))) * 1.4826

        psd_50 = psd_60 = slope = None
        if i in psd_results:
            psd_50, psd_60, slope = psd_results[i]

        corr_outlier = False
        corr_score = None
        if corr_scores is not None and corr_mean is not None and corr_std is not None:
            corr_score = float(corr_scores[i])
            if corr_std > 0 and corr_score < corr_mean - correlation_outlier_threshold * corr_std:
                corr_outlier = True

        cqc = ChannelQC(
            name=ch,
            index=i,
            is_flatline=is_flatline,
            flatline_fraction=0.0,
            has_nan=has_nan,
            nan_fraction=nan_frac,
            has_inf=has_inf,
            is_saturated=is_sat,
            saturation_fraction=sat_frac,
            peak_to_peak=p2p,
            robust_variance=mad ** 2,
            line_noise_power_50hz=psd_50,
            line_noise_power_60hz=psd_60,
            psd_slope=slope,
            correlation_outlier=corr_outlier,
            correlation_score=corr_score,
        )
        channel_qcs.append(cqc)

    # Aggregate
    report.channel_qc = channel_qcs
    report.n_flatline = sum(1 for c in channel_qcs if c.is_flatline)
    report.n_nan = sum(1 for c in channel_qcs if c.has_nan)
    report.n_inf = sum(1 for c in channel_qcs if c.has_inf)
    report.n_saturated = sum(1 for c in channel_qcs if c.is_saturated)
    report.n_correlation_outliers = sum(1 for c in channel_qcs if c.correlation_outlier)
    report.global_nan_fraction = nan_total / (n_ch * n_t) if n_ch * n_t > 0 else 0.0

    # Severity bucketing — purely numerical, no clinical claims
    if report.n_flatline > 0:
        report.warnings.append(
            f"{report.n_flatline} flatline channels detected "
            f"(std < {flatline_threshold_sd:.3f} × median std)."
        )
    if report.n_nan > 0:
        report.warnings.append(
            f"{report.n_nan} channels contain NaN values "
            f"(global NaN fraction: {report.global_nan_fraction:.4f})."
        )
    if report.n_inf > 0:
        report.blockers.append(f"{report.n_inf} channels contain Inf values.")
    if report.n_saturated > 0:
        report.warnings.append(
            f"{report.n_saturated} channels appear saturated "
            f"(>{saturation_pct:.1f}% of samples at physical min or max)."
        )

    # Cohort-relative amplitude outliers
    p2p_values = [c.peak_to_peak for c in channel_qcs if c.peak_to_peak is not None]
    if len(p2p_values) >= 2:
        med_p2p = float(np.median(p2p_values))
        for c in channel_qcs:
            if c.peak_to_peak is not None and med_p2p > 0:
                ratio = c.peak_to_peak / med_p2p
                if ratio > 10 or ratio < 0.1:
                    report.warnings.append(
                        f"Channel {c.name}: peak-to-peak {c.peak_to_peak:.2f} is "
                        f"{ratio:.1f}× the cohort median ({med_p2p:.2f})."
                    )

    if n_t < int(sampling_frequency_hz):
        report.unknowns.append(
            "Recording is shorter than 1 second; spectral and correlation metrics may be unreliable."
        )
        report.confidence = MethodConfidence.LOW_CONFIDENCE

    report.runtime_s = time.perf_counter() - t0
    return report


def run_signal_qc_on_dataset(
    dataset_path: Path,
    *,
    modality: str = "eeg",
    max_files: int | None = None,
) -> NeuroClassicReport:
    """Run signal QC across all matching files in a BIDS dataset.

    Requires MNE for file loading: pip install 'qortex[eeg]'
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "Signal QC dataset scan requires MNE. "
            "Install with: pip install 'qortex[eeg]'"
        ) from None

    spec = _SPEC
    nc_report = NeuroClassicReport(
        method_name="signal_qc",
        method_version=__version__,
        modality=modality,
        dataset_path=str(dataset_path),
        spec=spec,
    )

    signal_exts = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_edf,
                   ".fif": mne.io.read_raw_fif}
    files = [
        f for ext in signal_exts
        for f in sorted(Path(dataset_path).rglob(f"*{ext}"))
    ]
    if max_files:
        files = files[:max_files]

    for f in files:
        ext = f.suffix.lower()
        reader = signal_exts.get(ext)
        if reader is None:
            continue
        try:
            t0 = time.perf_counter()
            raw = reader(str(f), preload=True, verbose=False)
            data = raw.get_data().astype(np.float32)
            sfreq = float(raw.info["sfreq"])
            ch_names = list(raw.info.ch_names)
            qc = compute_signal_qc(
                data,
                sampling_frequency_hz=sfreq,
                channel_names=ch_names,
                scope=str(f),
            )
            qc.runtime_s += time.perf_counter() - t0
            nc_report.add_result(qc.to_result())
        except Exception as exc:
            result = NeuroClassicResult(
                method_name="signal_qc",
                method_version=__version__,
                modality=modality,
                scope=str(f),
                inputs={},
                parameters={},
                assumptions=[],
                blockers=[f"Could not load file: {exc}"],
                confidence=MethodConfidence.UNKNOWN,
            )
            nc_report.add_result(result)

    return nc_report


# ── PSD helpers ───────────────────────────────────────────────────────────────

def _compute_psd_metrics(
    data: np.ndarray,
    sfreq: float,
    line_hz: float | None,
) -> dict[int, tuple[float | None, float | None, float | None]]:
    """Per-channel Welch PSD → line noise power and spectral slope."""
    try:
        from scipy.signal import welch
    except ImportError:
        return {}

    n_ch, n_t = data.shape
    nperseg = min(int(sfreq * 4), n_t)
    if nperseg < 16:
        return {}

    out: dict[int, tuple[float | None, float | None, float | None]] = {}
    for i in range(n_ch):
        row = data[i]
        finite = row[np.isfinite(row)]
        if len(finite) < nperseg:
            out[i] = (None, None, None)
            continue
        freqs, psd = welch(finite, fs=sfreq, nperseg=nperseg)

        # Line noise power (sum in ±2 Hz band)
        p50 = p60 = None
        if line_hz is not None:
            mask = np.abs(freqs - line_hz) <= 2
            p50 = float(psd[mask].sum()) if mask.any() else None
        else:
            mask50 = np.abs(freqs - 50) <= 2
            mask60 = np.abs(freqs - 60) <= 2
            p50 = float(psd[mask50].sum()) if mask50.any() else None
            p60 = float(psd[mask60].sum()) if mask60.any() else None

        # Spectral slope (log-log linear fit on 1–40 Hz)
        slope = None
        slope_mask = (freqs >= 1) & (freqs <= 40) & (psd > 0)
        if slope_mask.sum() >= 4:
            log_f = np.log10(freqs[slope_mask])
            log_p = np.log10(psd[slope_mask])
            try:
                slope = float(np.polyfit(log_f, log_p, 1)[0])
            except Exception:
                pass

        out[i] = (p50, p60, slope)
    return out


def _compute_correlation_outliers(
    data: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Mean absolute correlation of each channel with all others."""
    n_ch, n_t = data.shape
    # Fast correlation via standardized rows
    valid_rows = np.where(np.isfinite(data).all(axis=1))[0]
    scores = np.full(n_ch, np.nan)
    if len(valid_rows) < 2:
        return scores, 0.0, 0.0

    sub = data[valid_rows]
    sub -= sub.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    sub /= norms
    corr = sub @ sub.T  # [n_valid, n_valid]
    np.fill_diagonal(corr, np.nan)
    with np.errstate(invalid="ignore"):
        mean_abs_corr = np.nanmean(np.abs(corr), axis=1)
    for j, idx in enumerate(valid_rows):
        scores[idx] = mean_abs_corr[j]

    valid_scores = scores[np.isfinite(scores)]
    mean_c = float(np.mean(valid_scores)) if len(valid_scores) else 0.0
    std_c = float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0
    return scores, mean_c, std_c
