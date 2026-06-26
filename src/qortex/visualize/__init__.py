"""qortex.visualize — modality-aware visual inspection layer.

Architecture
------------
Every rendering path goes through three typed stages:

    inspect(source)  →  VisualAsset   (what is this file?)
    asset.plan()     →  VisualPlan    (what will be rendered?)
    asset.render()   →  VisualResult  (html + png + provenance)

The universal entry point:

    visualize(source)  →  VisualResult

routes automatically to the right viewer based on the asset's detected intent,
family, and modality — no manual dispatch needed.

Supported sources
-----------------
Volume imaging (3D / 4D):
    NIfTI (.nii, .nii.gz) — anatomical MRI, fMRI BOLD, DWI, PET, stat maps
    DICOM series directory — CT, MR, PET; multi-series study browser
    FreeSurfer (.mgz, .mgh) — anatomical volumes
    NumPy arrays / nibabel images (duck-typed)

Overlays:
    overlay_mask(base, mask)       — binary mask overlay
    overlay_labelmap(base, labels) — multi-atlas segmentation
    overlay_stat(base, stat_map)   — z/t statistical map with threshold
    overlay_pet(base, pet)         — PET SUVR on anatomy

Electrophysiology:
    EEG/MEG/iEEG — via MNE (.fif, .edf, .bdf, .set, .vhdr, …)
    BOLD global mean signal from 4D NIfTI

DICOM:
    browse_dicom(dir) → HTML study/series browser

Quick usage
-----------
    from qortex import visualize

    # Auto-dispatch — always returns VisualResult
    result = visualize.visualize("sub-01_T1w.nii.gz")
    result.show()                   # open in browser
    result.to_html("output.html")   # write file + JSON sidecar

    # Inspect only (no rendering, very fast)
    asset = visualize.inspect("sub-01_task-rest_bold.nii.gz")
    print(asset.summary())
    print(asset.plan().describe())

    # Overlay API
    result = visualize.overlay_stat("T1w.nii.gz", "zmap.nii.gz", threshold=2.3)
    result = visualize.overlay_mask("T1w.nii.gz", "brain_mask.nii.gz")
    result = visualize.overlay_labelmap("T1w.nii.gz", "aparc+aseg.nii.gz")
    result = visualize.overlay_pet("T1w.nii.gz", "pet_suv.nii.gz")

    # DICOM
    result = visualize.browse_dicom("/path/to/dicom_study/")

    # Explicit viewers
    viewer = visualize.volume("bold.nii.gz", window="auto")
    ts     = visualize.timeseries("raw.edf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    # Core language
    "VisualAsset", "VisualPlan", "VisualResult", "VisualWarning",
    "INTENT_ANATOMICAL", "INTENT_BOLD", "INTENT_CT", "INTENT_DWI",
    "INTENT_PET", "INTENT_STAT_MAP", "INTENT_MASK", "INTENT_LABELMAP",
    "INTENT_RAW_SIGNAL", "INTENT_SURFACE", "INTENT_SERIES_BROWSER",
    # Universal API
    "inspect", "visualize", "browse_dicom",
    # Overlay API
    "overlay_mask", "overlay_labelmap", "overlay_stat", "overlay_pet",
    # ML overlay comparison
    "overlay_contour", "overlay_edges", "compare_masks",
    # DWI
    "DWIViewer", "dwi_summary",
    # Explicit viewers
    "volume", "timeseries",
    # Viewer classes (lazy)
    "VolumeViewer", "TimeSeriesViewer",
    # Colormap utilities
    "auto_window", "get_lut",
    "CT_PRESETS", "MR_PRESETS", "FMRI_PRESETS", "PET_PRESETS",
    # DICOM
    "list_dicom_series", "load_dicom_series", "DicomSeriesBrowser",
]

# ── Lazy imports — keep module load time minimal ──────────────────────────────

def _asset_module():
    from qortex.visualize import _asset
    return _asset

def _dispatch_module():
    from qortex.visualize import _dispatch
    return _dispatch

def _overlay_module():
    from qortex.visualize import overlay
    return overlay

def _dicom_module():
    from qortex.visualize import dicom
    return dicom

def _volume_cls():
    from qortex.visualize.volume import VolumeViewer
    return VolumeViewer

def _ts_cls():
    from qortex.visualize.timeseries import TimeSeriesViewer
    return TimeSeriesViewer


# ── Core language re-exports (lazy) ──────────────────────────────────────────

def __getattr__(name: str):
    # Asset language constants and classes
    _asset_names = {
        "VisualAsset", "VisualPlan", "VisualResult", "VisualWarning",
        "INTENT_ANATOMICAL", "INTENT_BOLD", "INTENT_CT", "INTENT_DWI",
        "INTENT_PET", "INTENT_STAT_MAP", "INTENT_MASK", "INTENT_LABELMAP",
        "INTENT_RAW_SIGNAL", "INTENT_SURFACE", "INTENT_SERIES_BROWSER",
    }
    if name in _asset_names:
        from qortex.visualize import _asset as a
        return getattr(a, name)

    # Colormap utilities
    _color_names = {"auto_window", "get_lut", "CT_PRESETS", "MR_PRESETS",
                    "FMRI_PRESETS", "PET_PRESETS"}
    if name in _color_names:
        from qortex.visualize import _colors
        return getattr(_colors, name)

    # Viewer classes
    if name == "VolumeViewer":
        return _volume_cls()
    if name == "TimeSeriesViewer":
        return _ts_cls()
    if name == "DWIViewer":
        return _dwi_cls()

    # DICOM helpers
    if name in {"list_dicom_series", "load_dicom_series", "DicomSeriesBrowser"}:
        from qortex.visualize import dicom as _d
        return getattr(_d, name)

    raise AttributeError(f"module 'qortex.visualize' has no attribute {name!r}")


# ── Universal API ─────────────────────────────────────────────────────────────

def inspect(source: Any) -> "VisualAsset":
    """Inspect any neuroimaging source and return a VisualAsset.

    Fast: reads only file headers, never loads pixel data.

    Parameters
    ----------
    source:
        Path (str or Path), nibabel image, MNE Raw, or numpy array.

    Returns
    -------
    VisualAsset with detected intent, modality, shape, warnings, and a
    recommended rendering strategy.

    Examples
    --------
    >>> asset = visualize.inspect("sub-01_T1w.nii.gz")
    >>> print(asset.summary())
    >>> print(asset.plan().describe())
    """
    from qortex.visualize._dispatch import inspect_file
    return inspect_file(source)


def visualize(
    source: Any,
    *,
    mode: str = "auto",
    overlay: Any = None,
    window: str | tuple | None = "auto",
    colormap: str | None = None,
    threshold: float = 2.3,
    alpha: float = 0.65,
    title: str = "",
) -> "VisualResult":
    """Universal renderer: detect modality, choose strategy, return VisualResult.

    Parameters
    ----------
    source:
        Path (str or Path), nibabel image, MNE Raw, or numpy array.
    mode:
        Rendering mode: ``"auto"`` (default), ``"interactive_html"``,
        ``"static"``, ``"thumbnail"``, ``"summary"``.
    overlay:
        Optional path/array to overlay on the volume (stat map, mask, etc.).
        When given, routes to the appropriate overlay function.
    window:
        Window preset or (vmin, vmax) tuple (volumetric only).
    colormap:
        Override colormap name.
    threshold:
        Threshold for stat map overlays.
    alpha:
        Overlay opacity.
    title:
        Override the auto-generated HTML title.

    Returns
    -------
    VisualResult with .html, .show(), .to_html(), .to_png() methods.

    Examples
    --------
    >>> result = visualize.visualize("sub-01_T1w.nii.gz")
    >>> result.show()
    >>> result.to_html("output.html")

    >>> result = visualize.visualize("T1w.nii.gz", overlay="zmap.nii.gz", threshold=2.3)
    """
    from qortex.visualize._dispatch import inspect_file, render_asset

    asset = inspect_file(source)

    # Overlay routing
    if overlay is not None:
        from qortex.visualize.overlay import overlay_stat, overlay_mask, overlay_labelmap, overlay_pet
        ov_asset = inspect_file(overlay)
        if ov_asset.intent in {"statistical_map"}:
            return overlay_stat(source, overlay, threshold=threshold, alpha=alpha,
                                colormap=colormap or "RdBu_r",
                                title=title or "Statistical Map Overlay")
        elif ov_asset.intent == "pet_volume":
            return overlay_pet(source, overlay, alpha=alpha,
                               title=title or "PET Overlay")
        elif ov_asset.intent == "labelmap":
            return overlay_labelmap(source, overlay, alpha=alpha,
                                    title=title or "Segmentation Overlay")
        else:
            return overlay_mask(source, overlay, alpha=alpha,
                                title=title or "Mask Overlay")

    return render_asset(asset, mode=mode, window=window, colormap=colormap, title=title)


def browse_dicom(directory: Path | str) -> "VisualResult":
    """Build an interactive HTML study/series browser for a DICOM directory.

    Returns a VisualResult whose .html is a self-contained study browser
    with a sortable series table and series-detail panel (panel 7 style).
    """
    from qortex.visualize.dicom import DicomSeriesBrowser
    from qortex.visualize._dispatch import inspect_file, plan_from_asset
    from qortex.visualize._asset import VisualResult

    browser = DicomSeriesBrowser(directory)
    html = browser.to_html()
    path = Path(directory)
    asset = inspect_file(path)
    plan = plan_from_asset(asset, "interactive_html")
    return VisualResult(
        asset=asset, plan=plan, html=html,
        provenance={"renderer": "DicomSeriesBrowser", "path": str(path)},
    )


# ── Overlay API ───────────────────────────────────────────────────────────────

def overlay_mask(base: Any, mask: Any, **kwargs) -> "VisualResult":
    """Overlay a binary mask on an anatomical image."""
    from qortex.visualize.overlay import overlay_mask as _fn
    return _fn(base, mask, **kwargs)


def overlay_labelmap(base: Any, labels: Any, **kwargs) -> "VisualResult":
    """Overlay a multi-label atlas/segmentation on an anatomical image."""
    from qortex.visualize.overlay import overlay_labelmap as _fn
    return _fn(base, labels, **kwargs)


def overlay_stat(base: Any, stat_map: Any, **kwargs) -> "VisualResult":
    """Overlay a thresholded statistical map (z/t) on an anatomical image."""
    from qortex.visualize.overlay import overlay_stat as _fn
    return _fn(base, stat_map, **kwargs)


def overlay_pet(base: Any, pet: Any, **kwargs) -> "VisualResult":
    """Overlay a PET SUVR map on an anatomical background."""
    from qortex.visualize.overlay import overlay_pet as _fn
    return _fn(base, pet, **kwargs)


def overlay_contour(base: Any, mask: Any, **kwargs) -> "VisualResult":
    """Overlay the 1-voxel-thick contour of a binary mask on an anatomical image."""
    from qortex.visualize.overlay import overlay_contour as _fn
    return _fn(base, mask, **kwargs)


def overlay_edges(base: Any, mask: Any, **kwargs) -> "VisualResult":
    """Overlay gradient-magnitude edges of a mask on an anatomical image."""
    from qortex.visualize.overlay import overlay_edges as _fn
    return _fn(base, mask, **kwargs)


def compare_masks(base: Any, pred: Any, truth: Any, **kwargs) -> "VisualResult":
    """TP/FP/FN diagnostic overlay comparing predicted vs ground-truth masks.

    Green = True Positive, Red = False Positive, Blue = False Negative.
    Dice similarity is computed from sampled slices and shown in the report.
    """
    from qortex.visualize.overlay import compare_masks as _fn
    return _fn(base, pred, truth, **kwargs)


def dwi_summary(dwi_path: Any, bval_path: Any = None, bvec_path: Any = None, **kwargs) -> Any:
    """4-panel DWI QC summary: b0, high-b, b-value histogram, gradient sphere."""
    from qortex.visualize.dwi import dwi_summary as _fn
    return _fn(dwi_path, bval_path=bval_path, bvec_path=bvec_path, **kwargs)


def _dwi_cls():
    from qortex.visualize.dwi import DWIViewer
    return DWIViewer


# ── Explicit viewers ──────────────────────────────────────────────────────────

def volume(source: Any, **kwargs) -> "VolumeViewer":
    """Create a VolumeViewer for 3D/4D neuroimaging data."""
    return _volume_cls()(source, **kwargs)


def timeseries(source: Any, **kwargs) -> "TimeSeriesViewer":
    """Create a TimeSeriesViewer for EEG/MEG/BOLD signal data."""
    return _ts_cls()(source, **kwargs)


# ── Deprecated alias -----------------------------------------------------------
# open() conflicts with Python built-in; prefer inspect() + render() or visualize()
def open(source: Any, **kwargs) -> Any:
    """Deprecated: use visualize() or inspect(). Will be removed in a future version."""
    import warnings
    warnings.warn(
        "qortex.visualize.open() is deprecated. Use visualize.visualize() or visualize.inspect().",
        DeprecationWarning, stacklevel=2,
    )
    return visualize(source, **kwargs)
