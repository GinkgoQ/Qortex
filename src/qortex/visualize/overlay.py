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

    # Discover unique labels from 3 strategic slices (25/50/75% of z-axis).
    # Covers most atlas structures without materialising the full volume.
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(label_h, _LazyNIfTI):
        n_z = label_h.shape[2]
        label_sample = np.concatenate([
            label_h.slice_along(2, int(n_z * q)).ravel()
            for q in (0.25, 0.50, 0.75)
        ])
        unique_labels = sorted(int(v) for v in np.unique(label_sample.astype(np.int32)) if v != 0)
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

    # Estimate suprathreshold count from ~10% of axial slices and scale to volume.
    # Avoids materialising the full stat map while still providing a useful count.
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(stat_h, _LazyNIfTI):
        n_z = stat_h.shape[2]
        sample_idxs = np.round(np.linspace(0, n_z - 1, max(3, n_z // 10))).astype(int)
        sample_supra = sum(
            int((np.abs(stat_h.slice_along(2, int(i)).ravel()) >= threshold).sum())
            for i in sample_idxs
        )
        n_clusters = int(sample_supra * n_z / len(sample_idxs))
    else:
        n_clusters = int((np.abs(stat_h) >= threshold).sum())

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

    # Compute percentile threshold from positive PET voxels sampled across ~10%
    # of axial slices — exact percentile without loading the full volume.
    from qortex.visualize.volume import _LazyNIfTI
    if isinstance(pet_h, _LazyNIfTI):
        n_z = pet_h.shape[2]
        sample_idxs = np.round(np.linspace(0, n_z - 1, max(5, n_z // 10))).astype(int)
        pos_samples = []
        for idx in sample_idxs:
            slc = pet_h.slice_along(2, int(idx)).ravel()
            pos_samples.append(slc[slc > 0])
        flat_pos = np.concatenate(pos_samples) if any(s.size > 0 for s in pos_samples) else np.array([0.0])
        threshold = float(np.percentile(flat_pos, threshold_pct)) if flat_pos.size > 0 else 0.0
    else:
        pet_arr = np.asarray(pet_h)
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


# ── ML-oriented overlay utilities ─────────────────────────────────────────────

def _binary_contour_2d(mask_2d: np.ndarray) -> np.ndarray:
    """Return the 1-pixel-wide contour of a binary mask.

    Pure numpy: contour = mask AND NOT morphological-erosion.
    The erosion uses 4-connectivity (no diagonal neighbors) to produce
    crisp, unambiguous boundary pixels.
    """
    m = mask_2d.astype(bool)
    eroded = (
        m
        & np.roll(m,  1, axis=0)
        & np.roll(m, -1, axis=0)
        & np.roll(m,  1, axis=1)
        & np.roll(m, -1, axis=1)
    )
    return m & ~eroded


def overlay_contour(
    base: Any,
    mask: Any,
    *,
    alpha: float = 0.9,
    color: tuple[int, int, int] = (255, 80, 80),
    title: str = "Mask Contour Overlay",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay the contour (boundary) of a binary mask on an anatomical image.

    Unlike ``overlay_mask()`` which fills the masked region, this draws only
    the 1-voxel-thick boundary — useful for checking registration accuracy and
    mask placement without obscuring the underlying anatomy.

    Parameters
    ----------
    base:   Path, nibabel image, or numpy array for the anatomical background.
    mask:   Path, nibabel image, or numpy array for the binary mask.
    alpha:  Contour line opacity (0–1).
    color:  RGB tuple for the contour colour.  Default red (255, 80, 80).
    resample:
        Resample mask to base geometry when shapes differ.
    allow_affine_mismatch:
        Suppress the OverlayGeometryError when affines differ.
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    mask_h, mask_shape, mask_aff = _load_vol_lazy(mask)
    base_h, mask_h = _check_geometry(
        base_h, mask_h, base_aff, mask_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)
    from qortex.visualize._html import array_to_b64png, build_interactive_html, apply_window
    from qortex.visualize._colors import apply_window as _aw

    nx, ny, nz = _get_shape3(base_h)
    MAX_SLICES = 80

    def _b64_contour_slice(axis: int, idx: int) -> str:
        base_slc = _get_slice(base_h, axis, idx).T[::-1, :]
        mask_slc = _get_slice(mask_h, axis, idx).T[::-1, :]
        base_norm = _aw(base_slc, vmin, vmax)     # (H, W) in [0, 1]
        contour = _binary_contour_2d(mask_slc)

        from qortex.visualize.overlay import _blend_slice
        # Use the label color mode but with a pre-built contour mask
        base_lut = get_lut("gray")
        base_idx = (np.clip(base_norm, 0, 1) * 255).astype(np.uint8)
        rgb = base_lut[base_idx].astype(np.float32)
        ov_color = np.array(color, dtype=np.float32)
        rgb[contour] = (1 - alpha) * rgb[contour] + alpha * ov_color
        return _rgb_slice_to_b64png(np.clip(rgb, 0, 255).astype(np.uint8))

    def _render_axis(axis: int, n: int):
        idxs = (np.round(np.linspace(0, n - 1, MAX_SLICES)).astype(int).tolist()
                if n > MAX_SLICES else list(range(n)))
        return [_b64_contour_slice(axis, i) for i in idxs], idxs

    slices_x, si_x = _render_axis(0, nx)
    slices_y, si_y = _render_axis(1, ny)
    slices_z, si_z = _render_axis(2, nz)

    html = build_interactive_html(
        title=title, dataset_info="Contour overlay",
        modality="overlay", shape=(nx, ny, nz), voxel_sizes=vox,
        vmin=vmin, vmax=vmax, window_str="contour",
        slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
        si_x=si_x, si_y=si_y, si_z=si_z,
        cx=nx // 2, cy=ny // 2, cz=nz // 2,
        n_volumes=1, tr=None, slices_t=None,
    )

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "contour_overlay", "alpha": alpha, "color": color})


def overlay_edges(
    base: Any,
    mask: Any,
    *,
    alpha: float = 0.8,
    color: tuple[int, int, int] = (80, 200, 255),
    title: str = "Edge Overlay",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Overlay the gradient-magnitude edges of a mask on an anatomical image.

    Uses a 2D finite-difference gradient (np.gradient) to detect edges —
    crisper than contour-erosion for smooth/probabilistic masks.  Threshold
    is set at 30 % of the per-slice gradient maximum so faint edges are
    captured without noise amplification.

    Parameters
    ----------
    base:   Anatomical background.
    mask:   Binary or probabilistic mask (edges are extracted via gradient).
    alpha:  Edge overlay opacity.
    color:  RGB tuple for edge colour.  Default cyan (80, 200, 255).
    """
    base_h, base_shape, base_aff = _load_vol_lazy(base)
    mask_h, mask_shape, mask_aff = _load_vol_lazy(mask)
    base_h, mask_h = _check_geometry(
        base_h, mask_h, base_aff, mask_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)
    from qortex.visualize._html import build_interactive_html
    from qortex.visualize._colors import apply_window as _aw

    nx, ny, nz = _get_shape3(base_h)
    MAX_SLICES = 80

    def _b64_edge_slice(axis: int, idx: int) -> str:
        base_slc = _get_slice(base_h, axis, idx).T[::-1, :]
        mask_slc = _get_slice(mask_h, axis, idx).T[::-1, :].astype(np.float32)
        base_norm = _aw(base_slc, vmin, vmax)

        dy, dx = np.gradient(mask_slc)
        grad_mag = np.sqrt(dx ** 2 + dy ** 2)
        thresh = float(grad_mag.max()) * 0.30
        edges = grad_mag > thresh if thresh > 1e-6 else np.zeros_like(grad_mag, dtype=bool)

        base_lut = get_lut("gray")
        base_idx_arr = (np.clip(base_norm, 0, 1) * 255).astype(np.uint8)
        rgb = base_lut[base_idx_arr].astype(np.float32)
        ov_color = np.array(color, dtype=np.float32)
        rgb[edges] = (1 - alpha) * rgb[edges] + alpha * ov_color
        return _rgb_slice_to_b64png(np.clip(rgb, 0, 255).astype(np.uint8))

    def _render_axis(axis: int, n: int):
        idxs = (np.round(np.linspace(0, n - 1, MAX_SLICES)).astype(int).tolist()
                if n > MAX_SLICES else list(range(n)))
        return [_b64_edge_slice(axis, i) for i in idxs], idxs

    slices_x, si_x = _render_axis(0, nx)
    slices_y, si_y = _render_axis(1, ny)
    slices_z, si_z = _render_axis(2, nz)

    html = build_interactive_html(
        title=title, dataset_info="Edge overlay (gradient magnitude)",
        modality="overlay", shape=(nx, ny, nz), voxel_sizes=vox,
        vmin=vmin, vmax=vmax, window_str="edges",
        slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
        si_x=si_x, si_y=si_y, si_z=si_z,
        cx=nx // 2, cy=ny // 2, cz=nz // 2,
        n_volumes=1, tr=None, slices_t=None,
    )

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "edge_overlay", "alpha": alpha})


def compare_masks(
    base: Any,
    pred: Any,
    truth: Any,
    *,
    alpha: float = 0.65,
    title: str = "Mask Comparison (TP / FP / FN)",
    resample: bool = False,
    allow_affine_mismatch: bool = False,
) -> VisualResult:
    """Compare a predicted binary mask against a ground-truth mask.

    Three-class diagnostic overlay — the standard tool for ML segmentation QC:

    * **Green**  — True Positive  (pred=1, truth=1) — correct detection
    * **Red**    — False Positive (pred=1, truth=0) — over-segmentation
    * **Blue**   — False Negative (pred=0, truth=1) — under-segmentation
    * Transparent — True Negative (pred=0, truth=0) — background unchanged

    Dice similarity and voxel counts are embedded in the HTML report header.
    All three volumes are read lazily — only the displayed slices are loaded.

    Parameters
    ----------
    base:   Anatomical background (NIfTI path, nibabel image, or numpy array).
    pred:   Predicted binary mask (same geometry as base).
    truth:  Ground-truth binary mask (same geometry as base).
    alpha:  Overlay opacity for TP/FP/FN voxels (0–1).
    resample:
        Resample pred and truth to base geometry when shapes differ.
    allow_affine_mismatch:
        Suppress OverlayGeometryError for affine mismatches.
    """
    base_h, _, base_aff = _load_vol_lazy(base)
    pred_h, _, pred_aff = _load_vol_lazy(pred)
    truth_h, _, truth_aff = _load_vol_lazy(truth)

    base_h, pred_h = _check_geometry(
        base_h, pred_h, base_aff, pred_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    base_h, truth_h = _check_geometry(
        base_h, truth_h, base_aff, truth_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )

    vmin, vmax = _overlay_auto_window(base_h)
    vox = _overlay_voxel_sizes(base_h)
    from qortex.visualize._html import build_interactive_html
    from qortex.visualize._colors import apply_window as _aw

    # Compute Dice from sampled slices (avoids full volume materialisation)
    from qortex.visualize.volume import _LazyNIfTI
    nx, ny, nz = _get_shape3(base_h)
    n_z_sample = max(5, nz // 8)
    sample_z = np.round(np.linspace(0, nz - 1, n_z_sample)).astype(int)
    tp_total = fp_total = fn_total = 0
    for z in sample_z:
        p = _get_slice(pred_h, 2, int(z)).ravel() > 0.5
        t = _get_slice(truth_h, 2, int(z)).ravel() > 0.5
        tp_total += int((p & t).sum())
        fp_total += int((p & ~t).sum())
        fn_total += int((~p & t).sum())
    denom = 2 * tp_total + fp_total + fn_total
    dice_approx = (2 * tp_total / denom) if denom > 0 else 1.0

    MAX_SLICES = 80

    # RGB colors
    _TP = np.array([30, 200, 80], dtype=np.float32)    # green
    _FP = np.array([240, 60, 60], dtype=np.float32)    # red
    _FN = np.array([60, 100, 240], dtype=np.float32)   # blue

    def _b64_compare_slice(axis: int, idx: int) -> str:
        base_slc = _get_slice(base_h, axis, idx).T[::-1, :]
        pred_slc = (_get_slice(pred_h, axis, idx).T[::-1, :] > 0.5)
        truth_slc = (_get_slice(truth_h, axis, idx).T[::-1, :] > 0.5)

        base_norm = _aw(base_slc, vmin, vmax)
        base_lut = get_lut("gray")
        base_idx_arr = (np.clip(base_norm, 0, 1) * 255).astype(np.uint8)
        rgb = base_lut[base_idx_arr].astype(np.float32)

        tp = pred_slc & truth_slc
        fp = pred_slc & ~truth_slc
        fn = ~pred_slc & truth_slc

        for mask_2d, color in [(tp, _TP), (fp, _FP), (fn, _FN)]:
            rgb[mask_2d] = (1 - alpha) * rgb[mask_2d] + alpha * color

        return _rgb_slice_to_b64png(np.clip(rgb, 0, 255).astype(np.uint8))

    def _render_axis(axis: int, n: int):
        idxs = (np.round(np.linspace(0, n - 1, MAX_SLICES)).astype(int).tolist()
                if n > MAX_SLICES else list(range(n)))
        return [_b64_compare_slice(axis, i) for i in idxs], idxs

    slices_x, si_x = _render_axis(0, nx)
    slices_y, si_y = _render_axis(1, ny)
    slices_z, si_z = _render_axis(2, nz)

    dice_str = f"{dice_approx:.3f}" if not (dice_approx != dice_approx) else "N/A"
    legend = (
        "🟢 TP (correct) &nbsp;🔴 FP (over-seg) &nbsp;🔵 FN (under-seg)"
        f" &nbsp;·&nbsp; Dice ≈ {dice_str}"
        f" &nbsp;(sampled from {n_z_sample} slices)"
    )

    html = build_interactive_html(
        title=title, dataset_info=legend,
        modality="overlay", shape=(nx, ny, nz), voxel_sizes=vox,
        vmin=vmin, vmax=vmax, window_str="comparison",
        slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
        si_x=si_x, si_y=si_y, si_z=si_z,
        cx=nx // 2, cy=ny // 2, cz=nz // 2,
        n_volumes=1, tr=None, slices_t=None,
    )

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    return VisualResult(
        asset=asset, plan=plan, html=html,
        provenance={
            "type": "compare_masks",
            "dice_approx": dice_approx,
            "tp_sample": tp_total,
            "fp_sample": fp_total,
            "fn_sample": fn_total,
        },
    )
