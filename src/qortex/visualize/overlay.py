"""Overlay system: mask, segmentation, statistical maps.

All overlays start with a geometry check.  By default, overlaying two images
with different affines raises OverlayGeometryError.  Explicit resample=True
enables nearest-neighbour (for labels) or linear (for scalars) resampling.

Overlay types
-------------
overlay_mask(base, mask)         — binary mask, transparent where 0
overlay_labelmap(base, labels)   — multi-label atlas/segmentation with colours
overlay_stat(base, stat_map)     — thresholded diverging statistical map
overlay_pet(base, pet)           — PET SUVR on anatomical background

All return VisualResult so callers have a consistent interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from qortex.visualize._asset import (
    INTENT_LABELMAP, INTENT_MASK, INTENT_PET, INTENT_STAT_MAP,
    VisualAsset, VisualPlan, VisualResult, VisualWarning,
    MODE_INTERACTIVE, MODE_STATIC, _warn,
)
from qortex.visualize._colors import apply_window, auto_window, get_lut
from qortex.visualize._html import array_to_b64png, build_interactive_html, render_axis_slices
from qortex.visualize._dispatch import inspect_file, plan_from_asset

log = logging.getLogger(__name__)


class OverlayGeometryError(ValueError):
    """Raised when two images have incompatible geometry for overlay."""


# ── Discrete label palette ────────────────────────────────────────────────────

_LABEL_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0 = background → transparent
    (255, 0, 0),      # 1 red
    (0, 255, 0),      # 2 green
    (0, 120, 255),    # 3 blue
    (255, 200, 0),    # 4 yellow
    (255, 0, 255),    # 5 magenta
    (0, 255, 255),    # 6 cyan
    (255, 128, 0),    # 7 orange
    (128, 0, 255),    # 8 purple
    (0, 200, 128),    # 9 teal
    (200, 100, 0),    # 10 brown
    (200, 200, 200),  # 11 light grey
    (80, 80, 255),    # 12 periwinkle
    (255, 80, 80),    # 13 salmon
    (80, 255, 80),    # 14 lime
    (255, 255, 80),   # 15 pale yellow
]


def _label_color(label: int) -> tuple[int, int, int]:
    if label == 0:
        return (0, 0, 0)
    return _LABEL_COLORS[label % len(_LABEL_COLORS)]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _affines_close(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return True  # cannot check — assume OK
    return bool(np.allclose(a, b, atol=1e-3))


def _shapes_match_3d(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape[:3] == b.shape[:3]


def _check_geometry(
    base: Any, overlay: Any,
    base_affine: np.ndarray | None, overlay_affine: np.ndarray | None,
    resample: bool,
    allow_affine_mismatch: bool = False,
    interp_order: int = 0,
) -> tuple[Any, Any]:
    """Verify geometry compatibility between base and overlay.

    Accepts lazy (_LazyNIfTI) or eager (np.ndarray) handles.  When shapes
    differ and ``resample=True``, both handles are materialised and nibabel
    resampling is applied, returning plain arrays.  Otherwise the original
    handles (possibly still lazy) are returned unchanged so that callers can
    continue to read slices on demand.
    """
    base_shape   = _get_shape3(base)
    overlay_shape = _get_shape3(overlay)

    if base_shape != overlay_shape[:3]:
        if not resample:
            raise OverlayGeometryError(
                f"Shape mismatch: base {base_shape} vs overlay {overlay_shape[:3]}. "
                "Pass resample=True to enable resampling."
            )
        # Resampling needs full arrays in RAM — materialise lazily
        try:
            import nibabel.processing as nbp
            import nibabel as nib
            base_arr = _force_load(base)
            ov_arr   = _force_load(overlay)
            aff    = base_affine    if base_affine    is not None else np.eye(4)
            ov_aff = overlay_affine if overlay_affine is not None else np.eye(4)
            base_img = nib.Nifti1Image(base_arr, aff)
            ov_img   = nib.Nifti1Image(ov_arr.astype(np.float32), ov_aff)
            resampled = nbp.resample_from_to(ov_img, base_img, order=interp_order)
            overlay = np.asarray(resampled.dataobj, dtype=np.float32)
            log.info("Resampled overlay → %s", overlay.shape[:3])
        except ImportError:
            raise ImportError("Resampling requires nibabel: pip install nibabel")

    if not _affines_close(base_affine, overlay_affine):
        if allow_affine_mismatch:
            log.warning("Affine mismatch — overlay may be misaligned.")
        else:
            raise OverlayGeometryError(
                "Affine mismatch: base and overlay have different world-space geometry. "
                "Pass allow_affine_mismatch=True to suppress (images may be misaligned), "
                "or resample=True to resample into base space."
            )

    return base, overlay


# ── Blend helpers ─────────────────────────────────────────────────────────────

def _blend_slice(
    base_slice: np.ndarray,  # (H, W) float in [0, 1]
    ov_slice: np.ndarray,    # (H, W) float or int
    ov_lut: np.ndarray,      # (256, 3) or None → use label colours
    alpha: float,
    threshold: float = 0.0,
    mode: str = "scalar",    # "scalar" | "label" | "binary"
) -> np.ndarray:
    """Alpha-blend overlay onto base slice.  Returns (H, W, 3) uint8."""
    base_lut = get_lut("gray")
    base_idx = (np.clip(base_slice, 0, 1) * 255).astype(np.uint8)
    base_rgb = base_lut[base_idx].astype(np.float32)

    if mode == "binary":
        mask = ov_slice.astype(bool)
        ov_color = np.array([255, 80, 80], dtype=np.float32)  # red
        result = base_rgb.copy()
        result[mask] = (1 - alpha) * base_rgb[mask] + alpha * ov_color
    elif mode == "label":
        result = base_rgb.copy()
        unique_labels = np.unique(ov_slice.astype(int))
        for lbl in unique_labels:
            if lbl == 0:
                continue
            mask = ov_slice.astype(int) == lbl
            color = np.array(_label_color(lbl), dtype=np.float32)
            result[mask] = (1 - alpha) * base_rgb[mask] + alpha * color
    else:  # scalar stat map
        above_thresh = np.abs(ov_slice) >= threshold
        normed_ov = np.clip(ov_slice / (np.abs(ov_slice).max() + 1e-10) * 0.5 + 0.5, 0, 1)
        ov_idx = (normed_ov * 255).astype(np.uint8)
        ov_rgb = ov_lut[ov_idx].astype(np.float32)
        result = base_rgb.copy()
        result[above_thresh] = (1 - alpha) * base_rgb[above_thresh] + alpha * ov_rgb[above_thresh]

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Overlay renderers ─────────────────────────────────────────────────────────

def _rgb_slice_to_b64png(blended: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB array as base64 PNG without Pillow."""
    import base64, struct, zlib
    H, W = blended.shape[:2]
    rows = [b"\x00" + blended[r].tobytes() for r in range(H)]
    raw = b"".join(rows)
    compressed = zlib.compress(raw, 6)

    def _chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _build_overlay_html(
    base_handle: Any,
    ov_handle: Any,
    title: str,
    vmin: float, vmax: float,
    alpha: float,
    mode: str,
    threshold: float,
    colormap: str,
    metadata_str: str,
    voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Build interactive HTML viewer with blended overlay, reading slices lazily.

    Both ``base_handle`` and ``ov_handle`` may be ``_LazyNIfTI`` instances (for
    large files only the slices needed for display are ever read from disk) or
    plain numpy arrays.
    """
    ov_lut = get_lut(colormap) if mode == "scalar" else None
    nx, ny, nz = _get_shape3(base_handle)
    MAX_SLICES = 80

    def _b64_blended_slice(axis: int, idx: int) -> str:
        base_slc = _get_slice(base_handle, axis, idx).T[::-1, :]
        ov_slc   = _get_slice(ov_handle,   axis, idx).T[::-1, :]
        base_normed = apply_window(base_slc, vmin, vmax)
        blended = _blend_slice(base_normed, ov_slc, ov_lut, alpha, threshold, mode)
        return _rgb_slice_to_b64png(blended)

    def _render_axis(axis: int, n_total: int) -> tuple[list[str], list[int]]:
        idxs = (
            np.round(np.linspace(0, n_total - 1, MAX_SLICES)).astype(int).tolist()
            if n_total > MAX_SLICES
            else list(range(n_total))
        )
        return [_b64_blended_slice(axis, i) for i in idxs], idxs

    slices_x, si_x = _render_axis(0, nx)
    slices_y, si_y = _render_axis(1, ny)
    slices_z, si_z = _render_axis(2, nz)

    return build_interactive_html(
        title=title,
        dataset_info=metadata_str,
        modality="overlay",
        shape=(nx, ny, nz),
        voxel_sizes=voxel_sizes,
        vmin=vmin, vmax=vmax, window_str="blended",
        slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
        si_x=si_x, si_y=si_y, si_z=si_z,
        cx=nx // 2, cy=ny // 2, cz=nz // 2,
        n_volumes=1, tr=None, slices_t=None,
    )


def _load_vol_lazy(source: Any) -> tuple[Any, tuple[int, ...], np.ndarray | None]:
    """Open source with the minimum possible I/O.

    Returns
    -------
    handle : _LazyNIfTI | np.ndarray
        For path-based NIfTI: a ``_LazyNIfTI`` — no pixel data is read.
        For numpy arrays or nibabel images: a pre-loaded array.
    shape3 : tuple[int, ...]
        Spatial shape (nx, ny, nz).
    affine : np.ndarray | None
        World-space affine, if available.
    """
    if isinstance(source, np.ndarray):
        arr = (source if source.ndim == 3 else source.mean(axis=-1)).astype(np.float32)
        return arr, arr.shape[:3], None

    if hasattr(source, "get_fdata") and hasattr(source, "affine"):
        data = np.asarray(source.get_fdata(dtype=np.float32))
        arr = data if data.ndim == 3 else data.mean(axis=-1)
        return arr, arr.shape[:3], source.affine

    path = Path(source)
    try:
        from qortex.visualize.volume import _LazyNIfTI
        lazy = _LazyNIfTI(path)
        return lazy, lazy.shape[:3], lazy.affine
    except ImportError:
        # nibabel unavailable — try loading eagerly as last resort
        try:
            import nibabel as nib
            img = nib.load(str(path))
            data = img.get_fdata(dtype=np.float32)
            arr = data if data.ndim == 3 else data.mean(axis=-1)
            return arr, arr.shape[:3], img.affine
        except ImportError:
            raise ImportError("Overlay rendering requires nibabel: pip install nibabel")


def _get_slice(handle: Any, axis: int, idx: int) -> np.ndarray:
    """Read one 2D slice from a lazy or eager volume handle."""
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(handle, _LazyNIfTI):
        return handle.slice_along(axis, idx)
    return np.take(handle, idx, axis=axis).astype(np.float32)


def _force_load(handle: Any) -> np.ndarray:
    """Materialise a lazy handle into a full numpy array (for resampling)."""
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(handle, _LazyNIfTI):
        return handle.mean_volume()
    return handle.astype(np.float32) if isinstance(handle, np.ndarray) else np.asarray(handle, dtype=np.float32)


def _get_shape3(handle: Any) -> tuple[int, ...]:
    from qortex.visualize.volume import _LazyNIfTI
    return handle.shape[:3] if isinstance(handle, _LazyNIfTI) else handle.shape[:3]


# ── Public overlay API ────────────────────────────────────────────────────────

def _overlay_voxel_sizes(base_handle: Any) -> tuple[float, float, float]:
    """Extract voxel sizes from a lazy or eager base handle."""
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(base_handle, _LazyNIfTI):
        return base_handle.zooms[:3]
    return (1.0, 1.0, 1.0)


def _overlay_auto_window(base_handle: Any) -> tuple[float, float]:
    """Estimate intensity window from a lazy or eager base handle."""
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(base_handle, _LazyNIfTI):
        return base_handle.sample_window("mri")
    arr = base_handle if isinstance(base_handle, np.ndarray) else np.asarray(base_handle)
    return auto_window(arr, "mri")


def overlay_mask(
    base: Any,
    mask: Any,
    *,
    alpha: float = 0.5,
    title: str = "Mask Overlay",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay a binary mask on an anatomical image.

    Parameters
    ----------
    base:   Path, nibabel image, or numpy array for the anatomical background.
    mask:   Path, nibabel image, or numpy array for the binary mask.
    alpha:  Overlay opacity (0 = invisible, 1 = opaque).
    resample:
        Resample mask to base geometry when shapes differ (requires nibabel).
    allow_affine_mismatch:
        Suppress the OverlayGeometryError when affines differ.  Images may be
        misaligned.
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    mask_h, mask_shape, mask_aff = _load_vol_lazy(mask)
    base_h, mask_h = _check_geometry(
        base_h, mask_h, base_aff, mask_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_h, mask_h, title, vmin, vmax, alpha,
        mode="binary", threshold=0.5, colormap="hot",
        metadata_str="Binary mask overlay",
        voxel_sizes=vox,
    )
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "mask_overlay", "alpha": alpha})


def overlay_labelmap(
    base: Any,
    labels: Any,
    *,
    alpha: float = 0.45,
    title: str = "Segmentation Overlay",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay a multi-label segmentation/atlas on an anatomical image.

    Each unique non-zero label receives a distinct colour from the built-in
    16-colour discrete palette.  Labels > 15 wrap around the palette.
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    label_h, label_shape, label_aff = _load_vol_lazy(labels)
    base_h, label_h = _check_geometry(
        base_h, label_h, base_aff, label_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)

    # Determine unique labels for provenance (materialise only if lazy)
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(label_h, _LazyNIfTI):
        label_sample = label_h.mean_volume()
        unique_labels = sorted(int(v) for v in np.unique(label_sample) if v != 0)
    else:
        unique_labels = sorted(int(v) for v in np.unique(label_h) if v != 0)

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_h, label_h, title, vmin, vmax, alpha,
        mode="label", threshold=0.0, colormap="plasma",
        metadata_str=f"Labels: {unique_labels[:10]}{'…' if len(unique_labels)>10 else ''}",
        voxel_sizes=vox,
    )
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "labelmap_overlay", "n_labels": len(unique_labels)})


def overlay_stat(
    base: Any,
    stat_map: Any,
    *,
    threshold: float = 2.3,
    alpha: float = 0.7,
    colormap: str = "RdBu_r",
    title: str = "Statistical Map",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay a thresholded z/t-map on an anatomical image.

    Voxels with |z| < ``threshold`` are transparent.  Above-threshold voxels
    are coloured by the diverging colormap (``RdBu_r`` by default).

    Parameters
    ----------
    threshold:  |z| or |t| value below which voxels are not shown.
    colormap:   Diverging (``"RdBu_r"``) or unilateral (``"hot"``) colormap.
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    stat_h, stat_shape, stat_aff = _load_vol_lazy(stat_map)
    base_h, stat_h = _check_geometry(
        base_h, stat_h, base_aff, stat_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=1,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)

    # Suprathreshold count for provenance — materialise stat only if needed
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(stat_h, _LazyNIfTI):
        stat_arr = stat_h.mean_volume()
    else:
        stat_arr = stat_h
    n_clusters = int((np.abs(stat_arr) >= threshold).sum())

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_h, stat_h, title, vmin, vmax, alpha,
        mode="scalar", threshold=threshold, colormap=colormap,
        metadata_str=f"Threshold |z|≥{threshold:.1f} · {n_clusters:,} suprathreshold voxels",
        voxel_sizes=vox,
    )
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "stat_overlay", "threshold": threshold,
                                    "n_suprathreshold": n_clusters})


def overlay_pet(
    base: Any,
    pet: Any,
    *,
    alpha: float = 0.65,
    colormap: str = "hot",
    threshold_pct: float = 10.0,
    title: str = "PET Overlay (SUVR)",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay a PET SUVR map on an anatomical background.

    Voxels below ``threshold_pct`` percentile of the PET volume are
    transparent so the anatomical background shows through at low-uptake
    regions.
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    pet_h, pet_shape, pet_aff = _load_vol_lazy(pet)
    base_h, pet_h = _check_geometry(
        base_h, pet_h, base_aff, pet_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=1,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)

    # Compute percentile-based threshold — needs PET values
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(pet_h, _LazyNIfTI):
        pet_sample = pet_h.sample_window("mri")
        threshold = pet_sample[0]  # vmin of PET as a rough proxy
    else:
        pet_arr = pet_h
        threshold = float(
            np.percentile(pet_arr[pet_arr > 0], threshold_pct)
        ) if (pet_arr > 0).any() else 0.0

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_h, pet_h, title, vmin, vmax, alpha,
        mode="scalar", threshold=threshold, colormap=colormap,
        metadata_str=f"PET SUV threshold={threshold:.2f} (p{threshold_pct:.0f})",
        voxel_sizes=vox,
    )
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "pet_overlay", "threshold": threshold})
