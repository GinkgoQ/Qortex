"""TimeSeriesViewer — frequency-domain and temporal visualization.

Supported data
--------------
* EEG/MEG raw files via MNE (optional): .fif, .edf, .bdf, .set, .cnt, .vhdr
* BOLD fMRI 4D NIfTI: signal at ROI over time + power spectrum
* Any 2-D array (samples × channels)
* Qortex SignalRecord or EventsRecord

Visualizations
--------------
butterfly   — overlay all channels (with amplitude envelope)
psd         — power spectral density per channel (Welch method, log scale)
spectrogram — time-frequency decomposition (STFT-based, no MNE required)
topomap     — EEG/MEG sensor topography at a given time (requires MNE + channel layout)
epoched     — event-locked average with ±SEM shading
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ── MNE helpers ───────────────────────────────────────────────────────────────

def _try_mne():
    try:
        import mne
        return mne
    except ImportError:
        return None


def _require_mne():
    mne = _try_mne()
    if mne is None:
        raise ImportError(
            "TimeSeriesViewer for EEG/MEG requires MNE: "
            "pip install mne  (or pip install 'qortex[eeg]')"
        )
    return mne


def _require_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        return go, sp
    except ImportError:
        raise ImportError("TimeSeriesViewer requires plotly: pip install plotly")


# ── Signal loading ────────────────────────────────────────────────────────────

class _SignalBundle:
    """Internal representation: data matrix + metadata."""

    def __init__(
        self,
        data: np.ndarray,       # (n_channels, n_samples)
        sfreq: float,
        ch_names: list[str] | None = None,
        ch_types: list[str] | None = None,
        events: np.ndarray | None = None,   # (n_events, 3) MNE format
        event_id: dict[str, int] | None = None,
        info_extra: dict | None = None,
    ) -> None:
        self.data = data          # always float32
        self.sfreq = sfreq
        self.n_channels, self.n_samples = data.shape
        self.ch_names = ch_names or [f"ch{i}" for i in range(self.n_channels)]
        self.ch_types = ch_types or ["misc"] * self.n_channels
        self.events = events
        self.event_id = event_id or {}
        self.info_extra = info_extra or {}

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sfreq


def _load_raw_mne(path: Path) -> _SignalBundle:
    mne = _require_mne()
    raw = mne.io.read_raw(str(path), preload=True, verbose=False)
    data = raw.get_data().astype(np.float32)
    return _SignalBundle(
        data=data,
        sfreq=raw.info["sfreq"],
        ch_names=raw.info["ch_names"],
        ch_types=[mne.channel_type(raw.info, i) for i in range(len(raw.info["ch_names"]))],
        info_extra={
            "n_channels": len(raw.info["ch_names"]),
            "sfreq": raw.info["sfreq"],
            "duration_s": raw.times[-1],
            "file_path": str(path),
        },
    )


def _from_ndarray(data: np.ndarray, sfreq: float) -> _SignalBundle:
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2 and data.shape[0] > data.shape[1]:
        # Assume (samples, channels) → transpose
        data = data.T
    return _SignalBundle(data=data.astype(np.float32), sfreq=sfreq)


def _from_bold_nifti(path: Path) -> _SignalBundle:
    """Extract mean BOLD signal from 4D NIfTI as a 1-channel signal."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("BOLD NIfTI requires nibabel: pip install nibabel")

    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape {vol.shape}")

    hdr = img.header
    zooms = hdr.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else 2.0
    sfreq = 1.0 / tr

    # Global mean signal + a rough brain mask (voxels > 10% of max)
    brain_mask = vol.mean(axis=3) > (vol.max() * 0.1)
    mean_signal = vol[brain_mask, :].mean(axis=0)

    return _SignalBundle(
        data=mean_signal[np.newaxis, :],
        sfreq=sfreq,
        ch_names=["BOLD global mean"],
        ch_types=["bold"],
        info_extra={"tr": tr, "shape": vol.shape},
    )


# ── Welch PSD (pure numpy) ────────────────────────────────────────────────────

def _welch_psd(
    signal: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method for power spectral density. Returns (freqs, psd)."""
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n = len(signal)
    window = np.hanning(nperseg)
    win_sum_sq = np.sum(window ** 2)

    segments = []
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start: start + nperseg] * window
        spec = np.fft.rfft(seg)
        segments.append(np.abs(spec) ** 2)

    if not segments:
        freqs = np.fft.rfftfreq(nperseg, 1.0 / sfreq)
        return freqs, np.zeros_like(freqs)

    psd = np.mean(segments, axis=0) / (sfreq * win_sum_sq)
    psd[1:-1] *= 2  # one-sided spectrum scaling
    freqs = np.fft.rfftfreq(nperseg, 1.0 / sfreq)
    return freqs, psd


def _stft(
    signal: np.ndarray,
    sfreq: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform. Returns (freqs, times, power_db)."""
    if noverlap is None:
        noverlap = nperseg * 3 // 4

    step = nperseg - noverlap
    window = np.hanning(nperseg)
    n = len(signal)

    specs = []
    t_centers = []
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start: start + nperseg] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        specs.append(spec)
        t_centers.append((start + nperseg / 2) / sfreq)

    freqs = np.fft.rfftfreq(nperseg, 1.0 / sfreq)
    t_arr = np.array(t_centers)
    power = np.array(specs).T  # (freqs, times)
    power_db = 10 * np.log10(np.maximum(power, 1e-30))
    return freqs, t_arr, power_db


# ── TimeSeriesViewer ──────────────────────────────────────────────────────────

class TimeSeriesViewer:
    """Interactive plotly-based viewer for EEG, MEG, BOLD fMRI, or any time series.

    Parameters
    ----------
    source:
        Path to an EEG/MEG raw file (any MNE-supported format), a 4D NIfTI
        for BOLD signal extraction, a numpy array (channels × samples), or an
        MNE Raw object.
    sfreq:
        Sampling frequency in Hz.  Required only when source is a numpy array.
    modality:
        ``"eeg"``, ``"meg"``, ``"bold"``, or ``"signal"`` (generic).
    """

    def __init__(
        self,
        source: Any,
        *,
        sfreq: float | None = None,
        modality: str | None = None,
    ) -> None:
        self._bundle = self._load(source, sfreq=sfreq)
        self.modality = modality or self._bundle.ch_types[0] if self._bundle.ch_types else "signal"

    def _load(self, source: Any, sfreq: float | None) -> _SignalBundle:
        mne = _try_mne()

        # MNE Raw object
        if mne and hasattr(source, "get_data") and hasattr(source, "info"):
            data = source.get_data().astype(np.float32)
            return _SignalBundle(
                data=data,
                sfreq=source.info["sfreq"],
                ch_names=source.info["ch_names"],
            )

        # numpy array
        if isinstance(source, np.ndarray):
            if sfreq is None:
                raise ValueError("sfreq required when source is a numpy array")
            return _from_ndarray(source, sfreq)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        suffix = path.suffix.lower()
        nifti_exts = {".nii", ".gz", ".mgz"}
        eeg_exts = {".fif", ".edf", ".bdf", ".set", ".cnt", ".vhdr", ".gdf", ".egi", ".mff"}

        if suffix in nifti_exts or (suffix == ".gz" and ".nii" in path.name):
            return _from_bold_nifti(path)
        elif suffix in eeg_exts:
            return _load_raw_mne(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        return self._bundle.n_channels

    @property
    def n_samples(self) -> int:
        return self._bundle.n_samples

    @property
    def sfreq(self) -> float:
        return self._bundle.sfreq

    @property
    def duration_s(self) -> float:
        return self._bundle.duration_s

    @property
    def channel_names(self) -> list[str]:
        return self._bundle.ch_names

    # ── Visualizations ────────────────────────────────────────────────────────

    def butterfly(
        self,
        *,
        tmin: float = 0.0,
        tmax: float | None = None,
        channels: list[int | str] | None = None,
        show_envelope: bool = True,
        title: str = "",
        max_channels: int = 64,
    ):
        """Butterfly plot: all channels overlaid, with amplitude envelope.

        Returns a plotly Figure.
        """
        go, sp = _require_plotly()

        bundle = self._bundle
        tmax_s = tmax if tmax is not None else bundle.duration_s
        t0 = int(tmin * bundle.sfreq)
        t1 = min(int(tmax_s * bundle.sfreq), bundle.n_samples)
        times = bundle.times[t0:t1]

        ch_indices = self._resolve_channels(channels, max_channels)
        data_slice = bundle.data[ch_indices, t0:t1]

        # Normalise each channel to zero mean, unit std for display
        mu = data_slice.mean(axis=1, keepdims=True)
        std = data_slice.std(axis=1, keepdims=True) + 1e-10
        normed = (data_slice - mu) / std

        fig = go.Figure()
        alpha = max(0.08, min(0.4, 1.0 / math.sqrt(len(ch_indices))))
        rgba = f"rgba(100,180,255,{alpha:.2f})"

        for i in range(len(ch_indices)):
            fig.add_trace(go.Scatter(
                x=times, y=normed[i].tolist(),
                mode="lines",
                line=dict(color=rgba, width=0.8),
                showlegend=False,
                hoverinfo="skip",
            ))

        if show_envelope and len(ch_indices) > 1:
            env_max = normed.max(axis=0)
            env_min = normed.min(axis=0)
            fig.add_trace(go.Scatter(
                x=np.concatenate([times, times[::-1]]).tolist(),
                y=np.concatenate([env_max, env_min[::-1]]).tolist(),
                fill="toself",
                fillcolor="rgba(255,200,50,0.15)",
                line=dict(color="rgba(255,200,50,0.6)", width=1),
                name="Envelope",
            ))

        fig.update_layout(
            title=title or f"{bundle.n_channels}-channel butterfly ({bundle.sfreq:.0f} Hz)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (z-scored)",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            height=400,
        )
        return fig

    def psd(
        self,
        *,
        fmin: float = 0.5,
        fmax: float | None = None,
        channels: list[int | str] | None = None,
        nperseg: int = 512,
        log_scale: bool = True,
        title: str = "",
        max_channels: int = 32,
    ):
        """Power spectral density (Welch method).

        Returns a plotly Figure with frequency on X and power (dB) on Y.
        """
        go, _ = _require_plotly()

        bundle = self._bundle
        ch_indices = self._resolve_channels(channels, max_channels)
        fmax_use = fmax if fmax is not None else bundle.sfreq / 2.0

        fig = go.Figure()
        colors = _channel_colors(len(ch_indices))

        for color_idx, ch_i in enumerate(ch_indices):
            freqs, psd = _welch_psd(bundle.data[ch_i], bundle.sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax_use)
            f_plot = freqs[mask]
            p_plot = psd[mask]
            if log_scale:
                p_plot = 10 * np.log10(np.maximum(p_plot, 1e-30))

            fig.add_trace(go.Scatter(
                x=f_plot.tolist(),
                y=p_plot.tolist(),
                mode="lines",
                name=bundle.ch_names[ch_i],
                line=dict(color=colors[color_idx], width=1),
                opacity=0.8,
            ))

        fig.update_layout(
            title=title or f"Power Spectral Density ({fmin:.1f}–{fmax_use:.1f} Hz)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB/Hz)" if log_scale else "Power (V²/Hz)",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            height=380,
        )
        return fig

    def spectrogram(
        self,
        channel: int | str = 0,
        *,
        fmin: float = 0.0,
        fmax: float | None = None,
        nperseg: int = 256,
        colormap: str = "plasma",
        title: str = "",
    ):
        """Time-frequency power spectrogram (STFT) for a single channel.

        Returns a plotly Figure (Heatmap).
        """
        go, _ = _require_plotly()

        bundle = self._bundle
        ch_i = self._resolve_channel(channel)
        signal = bundle.data[ch_i]
        fmax_use = fmax if fmax is not None else bundle.sfreq / 2.0

        freqs, t_arr, power_db = _stft(signal, bundle.sfreq, nperseg=nperseg)
        mask_f = (freqs >= fmin) & (freqs <= fmax_use)

        # Map to plotly-compatible colorscale name
        cscale = {"plasma": "Plasma", "hot": "Hot", "gray": "Gray"}.get(colormap, "Plasma")

        fig = go.Figure(go.Heatmap(
            x=t_arr.tolist(),
            y=freqs[mask_f].tolist(),
            z=power_db[mask_f, :].tolist(),
            colorscale=cscale,
            colorbar=dict(title="dB", len=0.8),
        ))

        ch_label = bundle.ch_names[ch_i]
        fig.update_layout(
            title=title or f"Spectrogram — {ch_label}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            height=360,
        )
        return fig

    def epoched(
        self,
        events: np.ndarray | None = None,
        event_id: dict[str, int] | None = None,
        *,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: tuple[float | None, float | None] = (None, 0.0),
        channels: list[int | str] | None = None,
        title: str = "",
    ):
        """Event-locked average (ERP/ERF) with ±SEM shading.

        If events is None and MNE events are available in the bundle they are used.
        Requires at least 2 distinct events. Returns a plotly Figure.
        """
        go, _ = _require_plotly()

        bundle = self._bundle
        _ev = events if events is not None else bundle.events
        if _ev is None or len(_ev) == 0:
            log.warning("epoched(): no events found, returning flat trace")
            return self.butterfly(title=title or "No events found")

        _eid = event_id or bundle.event_id
        ch_indices = self._resolve_channels(channels, max_channels=6)

        sfreq = bundle.sfreq
        n_pre = int(abs(tmin) * sfreq)
        n_post = int(tmax * sfreq)
        epoch_len = n_pre + n_post
        times_ep = np.linspace(tmin, tmax, epoch_len)

        cond_colors = ["#6af", "#f96", "#6f9", "#f6f", "#ff6", "#6ff"]
        fig = go.Figure()

        for cond_idx, (cond_name, cond_code) in enumerate(_eid.items()):
            ev_mask = _ev[:, 2] == cond_code
            onsets = _ev[ev_mask, 0]
            if len(onsets) == 0:
                continue

            epochs = []
            for onset in onsets:
                start = onset - n_pre
                end = onset + n_post
                if start < 0 or end > bundle.n_samples:
                    continue
                ep = bundle.data[ch_indices, start:end].mean(axis=0)
                epochs.append(ep)

            if not epochs:
                continue

            ep_arr = np.stack(epochs)   # (n_epochs, epoch_len)
            mean_ep = ep_arr.mean(axis=0)
            sem_ep = ep_arr.std(axis=0) / math.sqrt(len(epochs))

            # Apply baseline
            bl_start, bl_end = baseline
            bl_mask = np.ones(epoch_len, dtype=bool)
            if bl_start is not None:
                bl_mask &= times_ep >= bl_start
            if bl_end is not None:
                bl_mask &= times_ep <= bl_end
            if bl_mask.any():
                mean_ep -= mean_ep[bl_mask].mean()

            color = cond_colors[cond_idx % len(cond_colors)]
            rgba_fill = color.replace("#", "rgba(") + ",0.2)"
            if rgba_fill.startswith("rgba("):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                rgba_fill = f"rgba({r},{g},{b},0.2)"

            upper = (mean_ep + sem_ep).tolist()
            lower = (mean_ep - sem_ep).tolist()

            fig.add_trace(go.Scatter(
                x=np.concatenate([times_ep, times_ep[::-1]]).tolist(),
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=rgba_fill,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=times_ep.tolist(), y=mean_ep.tolist(),
                mode="lines",
                name=f"{cond_name} (n={len(epochs)})",
                line=dict(color=color, width=2),
            ))

        # Zero line and baseline window
        fig.add_vline(x=0, line=dict(color="#888", dash="dash", width=1))
        fig.update_layout(
            title=title or "Event-locked average",
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            height=380,
        )
        return fig

    def dashboard(self, *, title: str = "", channel: int | str = 0) -> str:
        """Build a self-contained HTML dashboard combining all views.

        Returns an HTML string (self-contained, no server required).
        """
        try:
            import plotly.io as pio
        except ImportError:
            raise ImportError("dashboard() requires plotly: pip install plotly")

        ch_i = self._resolve_channel(channel)
        plots: list[str] = []

        try:
            fig = self.butterfly(title="Signal Overview")
            plots.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="butterfly"))
        except Exception as exc:
            log.debug("butterfly failed: %s", exc)

        try:
            fig = self.psd(title="Power Spectral Density")
            plots.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="psd"))
        except Exception as exc:
            log.debug("psd failed: %s", exc)

        try:
            fig = self.spectrogram(channel=ch_i, title=f"Spectrogram — ch {ch_i}")
            plots.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="spec"))
        except Exception as exc:
            log.debug("spectrogram failed: %s", exc)

        bundle = self._bundle
        meta = {
            "Channels": bundle.n_channels,
            "Duration": f"{bundle.duration_s:.1f} s",
            "Sampling freq": f"{bundle.sfreq:.0f} Hz",
            "Samples": bundle.n_samples,
            **bundle.info_extra,
        }
        meta_rows = "".join(
            f'<tr><td style="color:#888">{k}</td><td>{v}</td></tr>'
            for k, v in meta.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title or 'Time Series Dashboard'}</title>
<style>
  body {{ background:#111; color:#ccc; font-family:sans-serif; margin:20px; }}
  h1 {{ color:#6af; font-size:1.4em; margin-bottom:4px; }}
  .meta-table {{ border-collapse:collapse; margin-bottom:24px; font-size:0.85em; }}
  .meta-table td {{ padding:3px 16px 3px 0; vertical-align:top; }}
  .plot-section {{ margin-bottom:32px; }}
</style>
</head>
<body>
<h1>{title or 'Time Series Dashboard'}</h1>
<table class="meta-table">{meta_rows}</table>
"""
        for plot_html in plots:
            html += f'<div class="plot-section">{plot_html}</div>\n'

        html += "</body></html>"
        return html

    def to_html(self, output: Path | str, **kwargs) -> Path:
        out_path = Path(output)
        out_path.write_text(self.dashboard(**kwargs), encoding="utf-8")
        return out_path

    def show(self) -> None:
        import tempfile, webbrowser
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(self.dashboard())
            tmp_path = f.name
        webbrowser.open(f"file://{tmp_path}")

    # ── Channel resolution ────────────────────────────────────────────────────

    def _resolve_channel(self, channel: int | str) -> int:
        if isinstance(channel, int):
            return channel
        try:
            return self._bundle.ch_names.index(channel)
        except ValueError:
            raise ValueError(f"Channel not found: {channel!r}")

    def _resolve_channels(
        self, channels: list[int | str] | None, max_channels: int
    ) -> list[int]:
        if channels is None:
            n = min(self._bundle.n_channels, max_channels)
            return list(range(n))
        resolved = []
        for c in channels:
            resolved.append(self._resolve_channel(c))
        return resolved

    def __repr__(self) -> str:
        return (
            f"TimeSeriesViewer(modality={self.modality!r}, "
            f"channels={self.n_channels}, duration={self.duration_s:.1f}s, "
            f"sfreq={self.sfreq:.0f}Hz)"
        )


# ── Color helpers ─────────────────────────────────────────────────────────────

def _channel_colors(n: int) -> list[str]:
    """Return n visually distinct hex color strings."""
    palette = [
        "#6af", "#f96", "#6f9", "#f6f", "#ff6", "#6ff",
        "#fa6", "#a6f", "#6fa", "#f6a", "#af6", "#6af",
    ]
    if n <= len(palette):
        return palette[:n]
    # Generate more via HSV rotation
    extra = []
    for i in range(n - len(palette)):
        h = (i / (n - len(palette))) * 0.9
        r, g, b = _hsv_to_rgb(h, 0.8, 0.9)
        extra.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return palette + extra


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    hi = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][hi]
