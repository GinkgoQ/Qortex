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
    base: np.ndarray, overlay: np.ndarray,
    base_affine: np.ndarray | None, overlay_affine: np.ndarray | None,
    resample: bool,
    allow_affine_mismatch: bool = False,
    interp_order: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify / resample overlay to match base geometry."""
    if not _shapes_match_3d(base, overlay):
        if not resample:
            raise OverlayGeometryError(
                f"Shape mismatch: base {base.shape[:3]} vs overlay {overlay.shape[:3]}. "
                "Pass resample=True to enable resampling."
            )
        try:
            import nibabel.processing as nbp
            import nibabel as nib
            # Build minimal nibabel images for processing
            aff = base_affine if base_affine is not None else np.eye(4)
            ov_aff = overlay_affine if overlay_affine is not None else np.eye(4)
            base_img = nib.Nifti1Image(base, aff)
            ov_img = nib.Nifti1Image(overlay.astype(np.float32), ov_aff)
            resampled = nbp.resample_from_to(ov_img, base_img, order=interp_order)
            overlay = np.asarray(resampled.dataobj)
            log.info("Resampled overlay from %s to %s", overlay.shape, base.shape[:3])
        except ImportError:
            raise ImportError("Resampling requires nibabel: pip install nibabel")

    if not _affines_close(base_affine, overlay_affine):
        if allow_affine_mismatch:
            log.warning("Affine mismatch between base and overlay — alignment may be incorrect.")
        else:
            raise OverlayGeometryError(
                "Affine mismatch: base and overlay have different world-space geometry. "
                "Pass allow_affine_mismatch=True to override (may produce misaligned images), "
                "or resample=True to resample overlay into base space."
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

def _build_overlay_html(
    base_vol: np.ndarray,
    ov_vol: np.ndarray,
    title: str,
    vmin: float, vmax: float,
    alpha: float,
    mode: str,
    threshold: float,
    colormap: str,
    metadata_str: str,
) -> str:
    """Build interactive HTML viewer with blended overlay pre-rendered."""
    import base64

    ov_lut = get_lut(colormap) if mode == "scalar" else None
    nx, ny, nz = base_vol.shape[:3]
    MAX_SLICES = 80

    def _b64_slice(axis: int, idx: int) -> str:
        base_slc = np.take(base_vol, idx, axis=axis).T[::-1, :].astype(np.float32)
        ov_slc = np.take(ov_vol, idx, axis=axis).T[::-1, :].astype(np.float32)
        base_normed = apply_window(base_slc, vmin, vmax)
        blended = _blend_slice(base_normed, ov_slc, ov_lut, alpha, threshold, mode)
        # Encode blended RGB array as PNG
        import io, struct, zlib
        H, W, _ = blended.shape

        def _row_filter(row_rgb: np.ndarray) -> bytes:
            return b"\x00" + row_rgb.tobytes()

        rows = [_row_filter(blended[r]) for r in range(H)]
        raw = b"".join(rows)
        compressed = zlib.compress(raw, 6)

        def _chunk(name: bytes, data: bytes) -> bytes:
            length = len(data)
            crc = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", length) + name + data + struct.pack(">I", crc)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", compressed)
            + _chunk(b"IEND", b"")
        )
        return base64.b64encode(png).decode()

    def _render_axis(axis: int, n_total: int) -> tuple[list[str], list[int]]:
        if n_total > MAX_SLICES:
            idxs = np.round(np.linspace(0, n_total - 1, MAX_SLICES)).astype(int).tolist()
        else:
            idxs = list(range(n_total))
        return [_b64_slice(axis, i) for i in idxs], idxs

    slices_x, si_x = _render_axis(0, nx)
    slices_y, si_y = _render_axis(1, ny)
    slices_z, si_z = _render_axis(2, nz)

    return build_interactive_html(
        title=title,
        dataset_info=metadata_str,
        modality="overlay",
        shape=(nx, ny, nz),
        voxel_sizes=(1.0, 1.0, 1.0),
        vmin=vmin, vmax=vmax, window_str="auto",
        slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
        si_x=si_x, si_y=si_y, si_z=si_z,
        cx=nx // 2, cy=ny // 2, cz=nz // 2,
        n_volumes=1, tr=None, slices_t=None,
    )


def _load_vol(source: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a volume from path / numpy / nibabel and return (data3d, affine)."""
    if isinstance(source, np.ndarray):
        vol = source if source.ndim == 3 else source.mean(axis=-1)
        return vol.astype(np.float32), None

    if hasattr(source, "get_fdata"):
        data = np.asarray(source.get_fdata(dtype=np.float32))
        vol = data if data.ndim == 3 else data.mean(axis=-1)
        return vol, source.affine

    path = Path(source)
    try:
        import nibabel as nib
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        vol = data if data.ndim == 3 else data.mean(axis=-1)
        return vol, img.affine
    except ImportError:
        raise ImportError("Loading NIfTI for overlay requires nibabel: pip install nibabel")


# ── Public overlay API ────────────────────────────────────────────────────────

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
    title:  HTML title string.
    resample:
        If True, resample mask to base geometry when shapes differ.
    allow_affine_mismatch:
        If True, allow overlay when affines differ (may produce misaligned images).
    """
    base_vol, base_aff = _load_vol(base)
    mask_vol, mask_aff = _load_vol(mask)
    base_vol, mask_vol = _check_geometry(
        base_vol, mask_vol, base_aff, mask_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = auto_window(base_vol, "mri")

    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_vol, mask_vol, title, vmin, vmax, alpha,
        mode="binary", threshold=0.5, colormap="hot",
        metadata_str="Binary mask overlay",
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

    Each unique non-zero label receives a distinct colour.
    """
    base_vol, base_aff = _load_vol(base)
    label_vol, label_aff = _load_vol(labels)
    base_vol, label_vol = _check_geometry(
        base_vol, label_vol, base_aff, label_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=0,
    )
    vmin, vmax = auto_window(base_vol, "mri")

    unique_labels = sorted(int(v) for v in np.unique(label_vol) if v != 0)
    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_vol, label_vol, title, vmin, vmax, alpha,
        mode="label", threshold=0.0, colormap="plasma",
        metadata_str=f"Labels: {unique_labels[:10]}{'…' if len(unique_labels)>10 else ''}",
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

    Voxels with |z| < threshold are transparent.  Above-threshold voxels
    are coloured by the diverging colormap.

    Parameters
    ----------
    threshold:  |z| or |t| value below which voxels are not shown.
    colormap:   ``"RdBu_r"`` (diverging) or ``"hot"`` (unilateral).
    """
    base_vol, base_aff = _load_vol(base)
    stat_vol, stat_aff = _load_vol(stat_map)
    base_vol, stat_vol = _check_geometry(
        base_vol, stat_vol, base_aff, stat_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=1,
    )
    vmin, vmax = auto_window(base_vol, "mri")

    n_clusters = int((np.abs(stat_vol) >= threshold).sum())
    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_vol, stat_vol, title, vmin, vmax, alpha,
        mode="scalar", threshold=threshold, colormap=colormap,
        metadata_str=f"Threshold |z|≥{threshold:.1f} · {n_clusters} suprathreshold voxels",
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

    Voxels below threshold_pct percentile of the PET volume are transparent.
    """
    base_vol, base_aff = _load_vol(base)
    pet_vol, pet_aff = _load_vol(pet)
    base_vol, pet_vol = _check_geometry(
        base_vol, pet_vol, base_aff, pet_aff, resample,
        allow_affine_mismatch=allow_affine_mismatch, interp_order=1,
    )
    vmin, vmax = auto_window(base_vol, "mri")

    threshold = float(np.percentile(pet_vol[pet_vol > 0], threshold_pct)) if (pet_vol > 0).any() else 0.0
    asset = inspect_file(base if not isinstance(base, np.ndarray) else base)
    plan = plan_from_asset(asset, "interactive_html")
    html = _build_overlay_html(
        base_vol, pet_vol, title, vmin, vmax, alpha,
        mode="scalar", threshold=threshold, colormap=colormap,
        metadata_str=f"PET SUV threshold={threshold:.2f} (p{threshold_pct:.0f})",
    )
    return VisualResult(asset=asset, plan=plan, html=html,
                        provenance={"type": "pet_overlay", "threshold": threshold})
