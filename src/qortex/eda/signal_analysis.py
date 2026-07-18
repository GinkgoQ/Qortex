"""Bounded real-file signal analysis for the Atlas analysis workspace."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroclassic.connectivity import (
    compute_graph_metrics,
    compute_pearson_connectivity,
)
from qortex.neuroclassic.infoth import compute_higuchi_fractal_dimension
from qortex.neuroclassic.signal_qc import compute_signal_qc
from qortex.parse._mne_utils import read_raw_with_bids_fallback

_CHANNEL_GROUPS = ("eeg", "mag", "grad", "seeg", "ecog")
_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def analyze_signal_file(
    path: Path | str,
    *,
    file_record: Any | None = None,
    duration_seconds: float = 20.0,
    max_channels: int = 32,
    connectivity_threshold: float = 0.35,
) -> dict[str, Any]:
    """Analyze a bounded segment from an MNE-readable EEG/MEG recording."""
    if duration_seconds <= 0 or duration_seconds > 120:
        raise ValueError("duration_seconds must be in (0, 120]")
    if max_channels < 2 or max_channels > 64:
        raise ValueError("max_channels must be in [2, 64]")
    if not 0 <= connectivity_threshold <= 1:
        raise ValueError("connectivity_threshold must be in [0, 1]")

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    started = time.perf_counter()
    if file_record is not None:
        datatype = source.parent.name
        raw, _, _ = read_raw_with_bids_fallback(
            file_record,
            source,
            datatype,
            file_record.suffix,
            False,
            {"verbose": "ERROR"},
        )
    else:
        import mne

        raw = mne.io.read_raw(str(source), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError("recording has no positive sampling frequency")

    channel_types = raw.get_channel_types()
    counts = {kind: channel_types.count(kind) for kind in sorted(set(channel_types))}
    selected_type = max(
        (kind for kind in _CHANNEL_GROUPS if counts.get(kind, 0)),
        key=lambda kind: counts[kind],
        default=None,
    )
    if selected_type is None:
        raise ValueError("recording contains no supported EEG, MEG, SEEG, or ECoG data channels")
    candidates = np.asarray(
        [
            index
            for index, kind in enumerate(channel_types)
            if kind == selected_type and raw.ch_names[index] not in raw.info.get("bads", [])
        ],
        dtype=int,
    )
    if len(candidates) < 2:
        raise ValueError(f"recording has fewer than two usable {selected_type} channels")
    if len(candidates) > max_channels:
        positions = np.linspace(0, len(candidates) - 1, max_channels).round().astype(int)
        picks = candidates[np.unique(positions)]
    else:
        picks = candidates

    stop = min(raw.n_times, max(2, int(round(duration_seconds * sfreq))))
    data = raw.get_data(picks=picks.tolist(), start=0, stop=stop).astype(np.float64, copy=False)
    names = [raw.ch_names[index] for index in picks]
    actual_duration = data.shape[1] / sfreq
    scope = str(source)

    from scipy.signal import spectrogram, welch

    nperseg = min(data.shape[1], max(8, int(round(2.0 * sfreq))))
    frequencies, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1, detrend="constant")
    upper = min(100.0, sfreq / 2.0)
    frequency_mask = (frequencies >= 1.0) & (frequencies <= upper)
    frequencies = frequencies[frequency_mask]
    psd = psd[:, frequency_mask]
    psd_mean = np.mean(psd, axis=0)
    psd_sem = np.std(psd, axis=0, ddof=1) / math.sqrt(psd.shape[0]) if psd.shape[0] > 1 else np.zeros(psd.shape[1])

    variances = np.var(data, axis=1)
    spectrogram_index = int(np.argmax(variances))
    spec_nperseg = min(data.shape[1], max(8, int(round(sfreq))))
    spec_f, spec_t, spec_power = spectrogram(
        data[spectrogram_index],
        fs=sfreq,
        nperseg=spec_nperseg,
        noverlap=spec_nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    spec_mask = (spec_f >= 1.0) & (spec_f <= upper)

    valid_bands = {name: bounds for name, bounds in _BANDS.items() if bounds[0] < upper}
    total_mask = (frequencies >= 1.0) & (frequencies <= upper)
    total_power = np.trapz(psd[:, total_mask], frequencies[total_mask], axis=1)
    bandpower: list[dict[str, Any]] = []
    for name, (low, high) in valid_bands.items():
        band_mask = (frequencies >= low) & (frequencies < min(high, upper))
        absolute = np.trapz(psd[:, band_mask], frequencies[band_mask], axis=1) if np.count_nonzero(band_mask) >= 2 else np.zeros(len(names))
        relative = np.divide(absolute, total_power, out=np.zeros_like(absolute), where=total_power > 0)
        bandpower.append(
            {
                "name": name,
                "range_hz": [low, min(high, upper)],
                "absolute_by_channel": absolute.astype(float).tolist(),
                "relative_by_channel": relative.astype(float).tolist(),
                "relative_mean": float(np.mean(relative)),
            }
        )

    high = min(40.0, sfreq / 2.0 - max(0.5, sfreq * 1e-6))
    frequency_band = (1.0, high) if high > 1.0 else None
    connectivity = compute_pearson_connectivity(
        data,
        channel_names=names,
        time_window_s=actual_duration,
        sampling_hz=sfreq,
        frequency_band=frequency_band,
        threshold=connectivity_threshold,
        scope=scope,
        input_signal_type=selected_type.upper(),
        node_definition=f"{selected_type}_sensor",
    )
    graph = compute_graph_metrics(connectivity, scope=scope)
    corr = np.asarray(connectivity.matrix, dtype=np.float64)
    fisher_z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
    np.fill_diagonal(fisher_z, 0.0)

    higuchi = compute_higuchi_fractal_dimension(
        data,
        channel_names=names,
        scope=scope,
        k_max=10,
        sampling_frequency_hz=sfreq,
    )
    qc = compute_signal_qc(
        data,
        sampling_frequency_hz=sfreq,
        channel_names=names,
        scope=scope,
        line_noise_hz=float(raw.info.get("line_freq")) if raw.info.get("line_freq") else None,
    )

    sensor_positions = []
    for index, name in zip(picks, names):
        loc = np.asarray(raw.info["chs"][int(index)]["loc"][:3], dtype=float)
        sensor_positions.append(
            {
                "name": name,
                "x_m": _json_number(loc[0]),
                "y_m": _json_number(loc[1]),
                "z_m": _json_number(loc[2]),
                "available": bool(np.isfinite(loc).all() and np.linalg.norm(loc) > 0),
            }
        )

    graph_dict = graph.to_dict()
    ranked_hubs = sorted(
        (
            {
                "channel": name,
                "degree": graph_dict["degree"][index],
                "strength": graph_dict["strength"][index],
                "betweenness": (graph_dict.get("betweenness_centrality") or [None] * len(names))[index],
            }
            for index, name in enumerate(names)
        ),
        key=lambda item: (item["strength"], item["degree"]),
        reverse=True,
    )

    feature_groups = [
        {"name": "bandpower", "count": len(names) * len(valid_bands), "definition": "Welch absolute and relative sensor bandpower"},
        {"name": "connectivity", "count": len(names) * (len(names) - 1) // 2, "definition": "Unique thresholded Pearson sensor pairs"},
        {"name": "graph", "count": 8 + 3 * len(names), "definition": "Global graph summaries plus degree, strength, and betweenness per sensor"},
        {"name": "complexity", "count": len(names), "definition": "Higuchi fractal dimension per sensor"},
        {"name": "signal_qc", "count": len(names), "definition": "Typed per-sensor signal quality records"},
    ]
    return {
        "parameters": {
            "duration_seconds": duration_seconds,
            "max_channels": max_channels,
            "connectivity_threshold": connectivity_threshold,
            "connectivity_frequency_band_hz": list(frequency_band) if frequency_band else None,
        },
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "format": source.suffix.lower(),
            "recording_duration_seconds": float(raw.n_times / sfreq),
            "segment_start_seconds": 0.0,
            "segment_duration_seconds": actual_duration,
            "sampling_frequency_hz": sfreq,
            "channel_type_counts": counts,
            "selected_channel_type": selected_type,
            "selected_channel_count": len(names),
            "bad_channels_excluded": list(raw.info.get("bads", [])),
            "condition": None,
            "condition_evidence": "Continuous recording segment; no event-derived condition grouping was applied.",
        },
        "channels": names,
        "sensor_positions": sensor_positions,
        "psd": {
            "method": "scipy.signal.welch",
            "window_seconds": nperseg / sfreq,
            "frequencies_hz": frequencies.astype(float).tolist(),
            "mean": psd_mean.astype(float).tolist(),
            "sem_across_sensors": psd_sem.astype(float).tolist(),
            "units": "SI-unit squared per Hz",
        },
        "spectrogram": {
            "method": "scipy.signal.spectrogram",
            "channel": names[spectrogram_index],
            "times_seconds": spec_t.astype(float).tolist(),
            "frequencies_hz": spec_f[spec_mask].astype(float).tolist(),
            "power": spec_power[spec_mask].astype(float).tolist(),
            "units": "SI-unit squared per Hz",
        },
        "bandpower": bandpower,
        "connectivity": {
            **connectivity.to_dict(include_matrix=True),
            "fisher_z_matrix": fisher_z.astype(float).tolist(),
            "positive_edge_count": int(np.count_nonzero(np.triu(corr > 0, 1))),
            "negative_edge_count": int(np.count_nonzero(np.triu(corr < 0, 1))),
        },
        "graph": {**graph_dict, "hubs": ranked_hubs[:10]},
        "higuchi": higuchi.to_dict(),
        "signal_qc": qc.to_dict(),
        "feature_registry": {
            "engine": "qortex.neuroclassic",
            "qortex_version": importlib.metadata.version("qortex"),
            "validated": not qc.blockers,
            "groups": feature_groups,
            "total_count": sum(group["count"] for group in feature_groups),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = ["analyze_signal_file"]
