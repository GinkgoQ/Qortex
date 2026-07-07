"""Group-comparison power spectral density figure (mean ± SEM band).

Reuses the existing Welch PSD implementation in ``timeseries.py`` — each
trial/channel's spectrum is computed independently before averaging, so the
uncertainty band reflects real across-trial variability rather than an
assumed interval.
"""

from __future__ import annotations

import numpy as np


def psd_band_comparison(
    conditions: dict[str, list[np.ndarray]],
    sfreq: float,
    *,
    fmin: float = 0.5,
    fmax: float | None = None,
    nperseg: int = 512,
    log_scale: bool = True,
    title: str = "",
):
    """Mean PSD ± SEM band per condition, across repeated trials/channels.

    Parameters
    ----------
    conditions:
        Mapping of condition label -> list of 1D signal arrays (repeated
        trials or channels for that condition, all at ``sfreq``). Each
        array's PSD is computed independently before averaging.
    sfreq:
        Sampling rate shared across all signals.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "psd_band_comparison() requires matplotlib and seaborn: "
            "pip install matplotlib seaborn"
        ) from exc

    from qortex.visualize.timeseries import _welch_psd

    if not conditions:
        raise ValueError("conditions must contain at least one condition with signals")

    fmax_use = fmax if fmax is not None else sfreq / 2.0

    sns.set_theme(style="whitegrid", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    palette = sns.color_palette("deep", n_colors=len(conditions))

    for color, (label, signals) in zip(palette, conditions.items()):
        if not signals:
            continue
        freqs = None
        psds = []
        for sig in signals:
            f, p = _welch_psd(np.asarray(sig, dtype=np.float64), sfreq, nperseg=nperseg)
            freqs = f
            psds.append(p)
        psd_arr = np.stack(psds, axis=0)
        if log_scale:
            psd_arr = 10 * np.log10(np.maximum(psd_arr, 1e-30))

        mean_psd = psd_arr.mean(axis=0)
        sem_psd = psd_arr.std(axis=0, ddof=1) / np.sqrt(len(psds)) if len(psds) > 1 else np.zeros_like(mean_psd)

        mask = (freqs >= fmin) & (freqs <= fmax_use)
        ax.plot(freqs[mask], mean_psd[mask], color=color, linewidth=1.6, label=f"{label} (n={len(psds)})")
        ax.fill_between(freqs[mask], (mean_psd - sem_psd)[mask], (mean_psd + sem_psd)[mask],
                         color=color, alpha=0.25, linewidth=0)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB/Hz)" if log_scale else "Power (V²/Hz)")
    ax.set_title(title or f"Power spectral density (mean ± SEM), {fmin:.1f}–{fmax_use:.1f} Hz",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


__all__ = ["psd_band_comparison"]
