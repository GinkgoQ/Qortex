"""Colormaps, windowing presets, and auto-windowing for neuroimaging data.

Windowing (center/width) is the standard convention in radiology:
    vmin = center - width/2
    vmax = center + width/2

All values are in native image units (Hounsfield for CT, arbitrary for MRI).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ── Window presets ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WindowPreset:
    """Windowing parameters for one clinical/research view."""
    name: str
    colormap: str = "gray"
    center: float | None = None    # intensity center; None → percentile mode
    width: float | None = None     # intensity width
    pct_low: float = 1.0           # percentile for vmin when center/width absent
    pct_high: float = 99.0         # percentile for vmax

    @property
    def vmin(self) -> float | None:
        return (self.center - self.width / 2) if self.center is not None else None

    @property
    def vmax(self) -> float | None:
        return (self.center + self.width / 2) if self.center is not None else None


# CT presets (Hounsfield units)
CT_PRESETS: dict[str, WindowPreset] = {
    "brain":       WindowPreset("brain",       center=40,    width=80),
    "subdural":    WindowPreset("subdural",    center=100,   width=300),
    "stroke":      WindowPreset("stroke",      center=40,    width=40),
    "bone":        WindowPreset("bone",        center=500,   width=2000),
    "soft_tissue": WindowPreset("soft_tissue", center=60,    width=400),
    "lung":        WindowPreset("lung",        center=-600,  width=1500),
    "liver":       WindowPreset("liver",       center=70,    width=150),
    "angio":       WindowPreset("angio",       center=300,   width=600),
    "abdomen":     WindowPreset("abdomen",     center=40,    width=350),
    "pelvis":      WindowPreset("pelvis",      center=40,    width=400),
}

# MR presets (percentile-based — intensity is scanner-dependent)
MR_PRESETS: dict[str, WindowPreset] = {
    "t1w":   WindowPreset("t1w",   pct_low=0.5, pct_high=99.5),
    "t2w":   WindowPreset("t2w",   pct_low=0.5, pct_high=99.5),
    "flair": WindowPreset("flair", pct_low=0.5, pct_high=99.5),
    "dwi":   WindowPreset("dwi",   pct_low=1.0, pct_high=99.0),
    "adc":   WindowPreset("adc",   pct_low=1.0, pct_high=99.0),
    "asl":   WindowPreset("asl",   colormap="plasma", pct_low=1.0, pct_high=99.0),
    "swi":   WindowPreset("swi",   pct_low=1.0, pct_high=99.5),
}

# fMRI presets
FMRI_PRESETS: dict[str, WindowPreset] = {
    "bold":  WindowPreset("bold",  pct_low=1.0, pct_high=99.0),
    "stat":  WindowPreset("stat",  colormap="hot", pct_low=0.0, pct_high=99.5),
    "zmap":  WindowPreset("zmap",  colormap="RdBu_r", pct_low=0.0, pct_high=99.5),
    "tmap":  WindowPreset("tmap",  colormap="RdBu_r", pct_low=0.0, pct_high=99.5),
}

# PET presets
PET_PRESETS: dict[str, WindowPreset] = {
    "suv":      WindowPreset("suv",      colormap="hot",    pct_low=0.0, pct_high=99.0),
    "fdg":      WindowPreset("fdg",      colormap="hot",    pct_low=0.0, pct_high=99.0),
    "amyloid":  WindowPreset("amyloid",  colormap="plasma", pct_low=0.0, pct_high=99.0),
}

_ALL_PRESETS = {**CT_PRESETS, **MR_PRESETS, **FMRI_PRESETS, **PET_PRESETS}


# ── Auto-windowing ────────────────────────────────────────────────────────────

def auto_window(
    array: np.ndarray,
    modality: str = "mri",
    *,
    suffix: str = "",
    preset: str | WindowPreset | None = None,
) -> tuple[float, float]:
    """Compute (vmin, vmax) for optimal display of a neuroimaging volume.

    Parameters
    ----------
    array:
        The 3D (or 4D, collapsed) image data.
    modality:
        Detected image modality: "ct", "mri", "fmri", "pet", "dwi", …
    suffix:
        BIDS suffix (e.g. "T1w", "bold", "dwi") for more specific selection.
    preset:
        Override: named preset string or WindowPreset dataclass.
    """
    if isinstance(preset, str):
        preset = _ALL_PRESETS.get(preset.lower())

    if isinstance(preset, WindowPreset):
        if preset.center is not None and preset.width is not None:
            return float(preset.vmin), float(preset.vmax)
        # fall through to percentile mode with preset params
        pct_low, pct_high = preset.pct_low, preset.pct_high
    else:
        pct_low, pct_high = _default_percentiles(modality, suffix)

    # Work on finite, non-background voxels
    flat = array.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0

    # For MRI/fMRI, exclude dark background (near-zero voxels from air/skull-stripping)
    if modality in {"mri", "fmri", "dwi"}:
        threshold = float(np.percentile(flat, 2.0))
        tissue = flat[flat > threshold]
        if tissue.size > 100:
            flat = tissue

    vmin = float(np.percentile(flat, pct_low))
    vmax = float(np.percentile(flat, pct_high))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def apply_window(
    array: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Clip + normalize array to [0, 1] using vmin/vmax."""
    span = vmax - vmin
    if span == 0:
        span = 1.0
    return np.clip((array.astype(np.float32) - vmin) / span, 0.0, 1.0)


def _default_percentiles(modality: str, suffix: str) -> tuple[float, float]:
    suffix_l = suffix.lower()
    if modality == "ct":
        return 2.0, 98.0
    if modality == "pet":
        return 0.0, 99.0
    if "bold" in suffix_l or "cbv" in suffix_l:
        return 1.0, 99.0
    if "stat" in suffix_l or "z" in suffix_l or "t" in suffix_l:
        return 0.0, 99.5
    return 0.5, 99.5


def colormap_for_modality(modality: str, suffix: str = "") -> str:
    """Return a suitable plotly/matplotlib colorscale name for this modality."""
    suffix_l = suffix.lower()
    if modality == "pet":
        return "hot"
    if "stat" in suffix_l or "zmap" in suffix_l or "tmap" in suffix_l:
        return "RdBu_r"
    if "bold" in suffix_l or modality == "fmri":
        return "gray"
    if modality in {"mri", "ct", "dwi"}:
        return "gray"
    return "gray"


# ── Colormap LUTs (pure numpy, no matplotlib dependency) ─────────────────────

def _make_gray_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    t = np.linspace(0, 255, 256, dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = t
    return lut


def _make_hot_lut() -> np.ndarray:
    """Black → red → yellow → white."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    t = np.arange(256, dtype=np.float32) / 255.0
    lut[:, 0] = np.clip(t * 3.0, 0, 1) * 255
    lut[:, 1] = np.clip(t * 3.0 - 1.0, 0, 1) * 255
    lut[:, 2] = np.clip(t * 3.0 - 2.0, 0, 1) * 255
    return lut


def _make_plasma_lut() -> np.ndarray:
    """Perceptually uniform plasma: dark purple → pink → yellow."""
    # 8-keypoint approximation of plasma
    r = np.array([0.05, 0.30, 0.56, 0.78, 0.93, 0.99, 0.97, 0.94], dtype=np.float32)
    g = np.array([0.03, 0.06, 0.08, 0.28, 0.59, 0.76, 0.88, 0.97], dtype=np.float32)
    b = np.array([0.53, 0.54, 0.55, 0.45, 0.26, 0.19, 0.14, 0.13], dtype=np.float32)
    x = np.linspace(0, 1, len(r))
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.interp(t, x, r) * 255
    lut[:, 1] = np.interp(t, x, g) * 255
    lut[:, 2] = np.interp(t, x, b) * 255
    return lut


def _make_rdbu_r_lut() -> np.ndarray:
    """Diverging blue–white–red (reversed: red at high values)."""
    r = np.array([0.02, 0.26, 0.57, 1.0, 1.0, 0.94, 0.70], dtype=np.float32)
    g = np.array([0.19, 0.47, 0.77, 1.0, 0.59, 0.22, 0.09], dtype=np.float32)
    b = np.array([0.74, 0.88, 0.97, 1.0, 0.59, 0.17, 0.09], dtype=np.float32)
    x = np.linspace(0, 1, len(r))
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.interp(t, x, r) * 255
    lut[:, 1] = np.interp(t, x, g) * 255
    lut[:, 2] = np.interp(t, x, b) * 255
    return lut


# Cache
_LUTS: dict[str, np.ndarray] = {}


def get_lut(name: str) -> np.ndarray:
    """Return (256, 3) uint8 LUT for the named colormap."""
    if name not in _LUTS:
        builders = {
            "gray":   _make_gray_lut,
            "grey":   _make_gray_lut,
            "hot":    _make_hot_lut,
            "plasma": _make_plasma_lut,
            "rdbu_r": _make_rdbu_r_lut,
            "RdBu_r": _make_rdbu_r_lut,
        }
        fn = builders.get(name) or builders.get(name.lower()) or _make_gray_lut
        _LUTS[name] = fn()
    return _LUTS[name]
