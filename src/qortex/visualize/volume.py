"""VolumeViewer — interactive orthogonal slice viewer for 3D/4D neuroimaging.

Supported sources
-----------------
* NIfTI (.nii, .nii.gz) — via nibabel (optional); fallback reads raw header only
* DICOM series directory or single .dcm file — via pydicom (optional)
* Already-loaded nibabel image object
* Qortex ImageRecord

Design
------
VolumeViewer is lazy: the image is memory-mapped by nibabel and slices are
extracted on demand.  No full volume is ever loaded into RAM unless the caller
explicitly calls ``.data()`` or ``.mean_volume()``.

The interactive HTML viewer pre-renders all slices along each axis as base64
PNGs (pure Python, no Pillow/matplotlib required) and embeds them in a
self-contained HTML page with JavaScript sliders for navigation.

For 4D fMRI the default view collapses to the mean volume; a TR slider is
added to navigate through individual time points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from qortex.visualize._colors import (
    CT_PRESETS, WindowPreset,
    auto_window, apply_window, colormap_for_modality, get_lut,
)
from qortex.visualize._html import (
    array_to_b64png, render_axis_slices, build_interactive_html,
)

log = logging.getLogger(__name__)

_NIFTI_EXTS = frozenset({".nii", ".mgz", ".mgh"})
_DICOM_EXTS = frozenset({".dcm", ".dicom", ".ima", ".img"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        raise ImportError(
            "VolumeViewer for NIfTI requires nibabel: "
            "pip install nibabel  (or pip install 'qortex[mri]')"
        )


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "VolumeViewer for DICOM requires pydicom: "
            "pip install pydicom  (or pip install 'qortex[dicom]')"
        )


def _detect_modality_from_path(path: Path) -> str:
    """Guess modality from filename / BIDS suffix."""
    name = path.name.lower()
    if any(x in name for x in ("bold", "cbv", "func")):
        return "fmri"
    if any(x in name for x in ("t1w", "t2w", "t2star", "flair", "pd", "anat")):
        return "mri"
    if any(x in name for x in ("dwi", "dti")):
        return "dwi"
    if "pet" in name or "fdg" in name:
        return "pet"
    return "mri"


def _voxel_sizes_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Extract voxel sizes (mm) from a NIfTI affine."""
    return tuple(float(v) for v in np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)))


# ── Lazy NIfTI accessor ───────────────────────────────────────────────────────

class _LazyNIfTI:
    """Memory-mapped NIfTI accessor. Reads slices on demand, no full-volume load."""

    def __init__(self, path):
        import nibabel as nib
        self._img = nib.load(str(path))
        self._proxy = self._img.dataobj  # nibabel ArrayProxy — zero RAM cost
        self.shape = self._img.shape
        self.affine = self._img.affine
        self.dtype = str(self._img.get_data_dtype())
        try:
            self.zooms = tuple(float(z) for z in self._img.header.get_zooms()[:3])
        except Exception:
            self.zooms = (1.0, 1.0, 1.0)

    def slice_along(self, axis: int, idx: int) -> np.ndarray:
        """Read exactly one 2D slice from disk."""
        return np.asarray(np.take(self._proxy, idx, axis=axis)).astype(np.float32)

    def mean_volume(self, max_frames: int = 50) -> np.ndarray:
        """Compute mean 3D volume from 4D data without loading all frames.

        Uses incremental accumulation — only one frame in memory at a time.
        For 3D data, reads the full volume (unavoidable).
        """
        shape = self._proxy.shape
        if len(shape) == 3:
            return np.asarray(self._proxy).astype(np.float32)
        n_t = shape[3]
        step = max(1, n_t // max_frames)
        frame_idxs = list(range(0, n_t, step))
        acc = np.zeros(shape[:3], dtype=np.float64)
        for t in frame_idxs:
            acc += np.asarray(self._proxy[..., t]).astype(np.float64)
        return (acc / len(frame_idxs)).astype(np.float32)

    def frame(self, t: int) -> np.ndarray:
        """Read one 3D frame from a 4D volume."""
        if len(self._proxy.shape) == 3:
            return np.asarray(self._proxy).astype(np.float32)
        return np.asarray(self._proxy[..., t]).astype(np.float32)

    def sample_window(self, modality: str = "mri") -> tuple[float, float]:
        """Estimate intensity window from a spatial subsample (not full volume).

        Samples ~10% of axial slices to compute robust percentiles.
        Never loads the full volume.
        """
        shape = self._proxy.shape
        n_z = shape[2]
        sample_idxs = np.round(np.linspace(0, n_z - 1, max(5, n_z // 10))).astype(int)
        samples = []
        for idx in sample_idxs:
            slc = np.asarray(self._proxy[:, :, int(idx)]).ravel().astype(np.float32)
            samples.append(slc)
        flat = np.concatenate(samples)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return 0.0, 1.0
        if modality in {"mri", "fmri", "dwi"}:
            threshold = float(np.percentile(flat, 2.0))
            tissue = flat[flat > threshold]
            if tissue.size > 100:
                flat = tissue
        pct_lo = 0.5 if modality == "ct" else 1.0
        pct_hi = 99.5
        vmin = float(np.percentile(flat, pct_lo))
        vmax = float(np.percentile(flat, pct_hi))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def voxel_timeseries(self, x: int, y: int, z: int) -> np.ndarray:
        """Read the full timeseries for a single voxel from disk.

        Uses nibabel's direct proxy indexing so only the voxel's data stripe
        is read — no 3D frame is ever constructed.  For 3D images returns a
        length-1 array.
        """
        if len(self._proxy.shape) == 3:
            return np.array([float(self._proxy[x, y, z])], dtype=np.float32)
        return np.asarray(self._proxy[x, y, z, :]).astype(np.float32)


# ── VolumeViewer ──────────────────────────────────────────────────────────────

class VolumeViewer:
    """Interactive viewer for 3D/4D neuroimaging volumes.

    Supports NIfTI (.nii, .nii.gz), DICOM (files or directories), and
    already-loaded nibabel images or Qortex ImageRecord objects.

    Parameters
    ----------
    source:
        Path to a NIfTI file, DICOM file, DICOM directory, or a nibabel
        image object, or a Qortex ``ImageRecord``.
    modality:
        Override the detected modality: ``"ct"``, ``"mri"``, ``"fmri"``,
        ``"pet"``, ``"dwi"``.
    window:
        Window preset name (``"brain"``, ``"bone"``, …), a ``(vmin, vmax)``
        tuple, or ``"auto"`` (default).
    colormap:
        Colormap: ``"gray"``, ``"hot"``, ``"plasma"``, ``"RdBu_r"``.
    """

    def __init__(
        self,
        source: Any,
        *,
        modality: str | None = None,
        window: str | tuple[float, float] | None = "auto",
        colormap: str | None = None,
    ) -> None:
        self._vol: np.ndarray | None = None
        self._lazy: _LazyNIfTI | None = None
        self._affine: np.ndarray = np.eye(4)
        self._meta: dict = {}
        self._overlay: np.ndarray | None = None
        self._overlay_params: dict = {}

        self._load_source(source)

        self.modality = modality or self._meta.get("modality", "mri")
        self._resolve_window(window)
        self.colormap = colormap or colormap_for_modality(self.modality)

    # ── Source loading ────────────────────────────────────────────────────

    def _load_source(self, source: Any) -> None:
        # ImageRecord (Qortex)
        try:
            from qortex.core.entities import ImageRecord
            if isinstance(source, ImageRecord):
                img = source.img
                self._vol = np.asarray(img.get_fdata(dtype=np.float32))
                self._affine = img.affine
                self._meta = {
                    "shape": self._vol.shape,
                    "zooms": source.voxel_size,
                    "tr": source.tr,
                    "n_volumes": source.n_volumes,
                }
                self.modality = source.file.modality or "mri"
                return
        except ImportError:
            pass

        # nibabel image object (duck-typed)
        if hasattr(source, "get_fdata") and hasattr(source, "affine"):
            self._vol = np.asarray(source.get_fdata(dtype=np.float32))
            self._affine = source.affine
            self._meta = {"shape": self._vol.shape}
            return

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        name_lower = path.name.lower()

        if suffix in {".dcm", ".dicom", ".ima"} or path.is_dir():
            from qortex.visualize.dicom import load_dicom_series, list_dicom_series
            if path.is_dir():
                series_list = list_dicom_series(path)
            else:
                series_list = list_dicom_series(path.parent)
            target = series_list[0] if series_list else None
            if target:
                volume, meta = load_dicom_series(target, apply_rescale=True)
                self._vol = volume
                self._affine = np.eye(4)
                self._meta = meta
                if "ct" in str(target.modality).lower():
                    self._meta["modality"] = "ct"
                else:
                    self._meta["modality"] = "mri"
            else:
                raise FileNotFoundError(f"No DICOM series found in {path}")
        elif suffix in _NIFTI_EXTS or name_lower.endswith(".nii.gz"):
            try:
                self._lazy = _LazyNIfTI(path)
                self._vol = None  # not loaded yet
                self._affine = self._lazy.affine
                self._meta = {
                    "shape": self._lazy.shape,
                    "zooms": self._lazy.zooms,
                    "modality": _detect_modality_from_path(path),
                }
                self.modality = self._meta["modality"]
            except ImportError:
                raise
            except Exception as exc:
                raise ValueError(f"Cannot load NIfTI {path}: {exc}") from exc
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _resolve_window(self, window: str | tuple | None) -> None:
        """Set self._vmin, self._vmax from the window spec."""
        if isinstance(window, (tuple, list)) and len(window) == 2:
            self._vmin, self._vmax = float(window[0]), float(window[1])
            return

        if isinstance(window, str) and window != "auto":
            preset = CT_PRESETS.get(window) or {}
            if isinstance(preset, WindowPreset) and preset.center is not None:
                self._vmin = float(preset.vmin)
                self._vmax = float(preset.vmax)
                return

        # auto / DICOM embedded window
        wc = self._meta.get("window_center")
        ww = self._meta.get("window_width")
        if self.modality == "ct" and wc is not None and ww is not None:
            self._vmin = wc - ww / 2
            self._vmax = wc + ww / 2
        elif self._lazy is not None:
            self._vmin, self._vmax = self._lazy.sample_window(self.modality)
        else:
            vol3d = self._vol3d()
            self._vmin, self._vmax = auto_window(vol3d, self.modality)

    def _vol3d(self) -> np.ndarray:
        """Return the 3D view: for 4D data, use the temporal mean."""
        if self._lazy is not None:
            return self._lazy.mean_volume()
        if self._vol is not None:
            if self._vol.ndim == 4:
                return self._vol.mean(axis=-1)
            return self._vol
        raise RuntimeError("No volume data")

    @property
    def shape(self) -> tuple:
        if self._lazy is not None:
            return self._lazy.shape
        return self._vol.shape if self._vol is not None else ()

    @property
    def voxel_sizes(self) -> tuple[float, float, float]:
        zooms = self._meta.get("zooms")
        if zooms and len(zooms) >= 3:
            return tuple(float(z) for z in zooms[:3])
        return _voxel_sizes_from_affine(self._affine)

    @property
    def n_volumes(self) -> int:
        if self._lazy is not None:
            shape = self._lazy.shape
            return shape[3] if len(shape) == 4 else 1
        if self._vol is not None and self._vol.ndim == 4:
            return self._vol.shape[3]
        return 1

    @property
    def tr(self) -> float | None:
        return self._meta.get("tr")

    # ── Overlay ───────────────────────────────────────────────────────────

    def overlay(
        self,
        stat_map: Any,
        *,
        threshold: float = 2.0,
        colormap: str = "hot",
        alpha: float = 0.6,
    ) -> "VolumeViewer":
        """Add a statistical map overlay (z-map, t-map, or any volume)."""
        if isinstance(stat_map, (str, Path)):
            lazy = _LazyNIfTI(Path(stat_map))
            stat_arr = lazy.mean_volume()
        elif isinstance(stat_map, np.ndarray):
            stat_arr = stat_map
        else:
            raise TypeError(f"Unsupported stat_map type: {type(stat_map)}")

        self._overlay = stat_arr
        self._overlay_params = {
            "threshold": threshold,
            "colormap": colormap,
            "alpha": alpha,
            "vmin": float(np.percentile(stat_arr[np.isfinite(stat_arr)], 1)),
            "vmax": float(np.percentile(stat_arr[np.isfinite(stat_arr)], 99.5)),
        }
        return self

    def mean_volume(self) -> "VolumeViewer":
        """Return a new VolumeViewer containing only the temporal mean (for 4D)."""
        new = VolumeViewer.__new__(VolumeViewer)
        new._vol = self._vol3d()
        new._lazy = None
        new._affine = self._affine
        new._meta = {**self._meta, "n_volumes": 1}
        new.modality = self.modality
        new._vmin = self._vmin
        new._vmax = self._vmax
        new.colormap = self.colormap
        new._overlay = self._overlay
        new._overlay_params = self._overlay_params
        return new

    # ── Plotly figures ────────────────────────────────────────────────────

    def ortho(
        self,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
        *,
        t: int | None = None,
        title: str = "",
    ):
        """Return a 3-panel plotly Figure showing orthogonal slices.

        For nibabel-backed sources reads exactly 3 slices from disk — the full
        volume is never loaded into RAM.  For 4D data the display timepoint
        defaults to the midpoint and can be overridden with ``t``.
        Voxel-size-aware aspect ratios are applied automatically.

        Parameters
        ----------
        x, y, z : int, optional
            Slice indices along each axis.  Defaults to the volume centre.
        t : int, optional
            Timepoint index for 4D data.  Defaults to n_volumes // 2.
        title : str, optional
            Override the auto-generated title.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("ortho() requires plotly: pip install plotly")

        # ── Shape (no data load for lazy sources) ─────────────────────────
        if self._lazy is not None:
            shape3 = self._lazy.shape[:3]
        elif self._vol is not None:
            shape3 = self._vol.shape[:3]
        else:
            raise RuntimeError("No volume data loaded")

        nx, ny, nz = shape3
        cx = x if x is not None else nx // 2
        cy = y if y is not None else ny // 2
        cz = z if z is not None else nz // 2

        # ── Read exactly 3 slices ─────────────────────────────────────────
        if self._lazy is not None:
            proxy = self._lazy._proxy
            n_vols = self.n_volumes
            if n_vols > 1:
                t_idx = int(t) if t is not None else n_vols // 2
                t_idx = max(0, min(t_idx, n_vols - 1))
                # Direct 4D proxy indexing — each reads one frame slice
                slc_ax  = np.asarray(proxy[:, :, cz, t_idx]).astype(np.float32)
                slc_cor = np.asarray(proxy[:, cy, :, t_idx]).astype(np.float32)
                slc_sag = np.asarray(proxy[cx, :, :, t_idx]).astype(np.float32)
            else:
                slc_ax  = self._lazy.slice_along(2, cz)   # (nx, ny)
                slc_cor = self._lazy.slice_along(1, cy)   # (nx, nz)
                slc_sag = self._lazy.slice_along(0, cx)   # (ny, nz)
        else:
            vol3d   = self._vol3d()
            slc_ax  = vol3d[:, :, cz]
            slc_cor = vol3d[:, cy, :]
            slc_sag = vol3d[cx, :, :]

        def _norm(arr: np.ndarray) -> np.ndarray:
            """Normalise to [0,1], transpose for display (row = y or z, col = x or y)."""
            disp = arr.T[::-1, :]
            return np.clip(
                (disp - self._vmin) / max(self._vmax - self._vmin, 1e-8), 0.0, 1.0
            )

        # ── Subplot labels ────────────────────────────────────────────────
        ax_label  = f"Axial   z={cz}"
        cor_label = f"Coronal  y={cy}"
        sag_label = f"Sagittal  x={cx}"

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(ax_label, cor_label, sag_label),
            horizontal_spacing=0.04,
        )

        common = dict(colorscale="gray", zmin=0, zmax=1, showscale=False,
                      hovertemplate="%{z:.3f}<extra></extra>")
        fig.add_trace(go.Heatmap(z=_norm(slc_ax),  **common, name="axial"),   row=1, col=1)
        fig.add_trace(go.Heatmap(z=_norm(slc_cor), **common, name="coronal"), row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=_norm(slc_sag), colorscale="gray", zmin=0, zmax=1,
                       showscale=True, hovertemplate="%{z:.3f}<extra></extra>",
                       colorbar=dict(len=0.7, thickness=12, x=1.01,
                                     title=dict(text="Norm.", side="right")),
                       name="sagittal"),
            row=1, col=3,
        )

        # ── Physical aspect-ratio correction ──────────────────────────────
        # Each subplot's y-axis is anchored to its x-axis with scaleratio = dy/dx
        # so that one screen pixel represents the same physical distance in both dims.
        # Axial  (xy-plane): x-cols=x (dx), y-rows=y (dy)  → scaleratio = dy/dx
        # Coronal (xz-plane): x-cols=x (dx), y-rows=z (dz)  → scaleratio = dz/dx
        # Sagittal(yz-plane): x-cols=y (dy), y-rows=z (dz)  → scaleratio = dz/dy
        dx, dy, dz = self.voxel_sizes[:3]
        fig.update_yaxes(scaleanchor="x",  scaleratio=dy/dx, row=1, col=1)
        fig.update_yaxes(scaleanchor="x2", scaleratio=dz/dx, row=1, col=2)
        fig.update_yaxes(scaleanchor="x3", scaleratio=dz/dy, row=1, col=3)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        # ── Informative auto-title ─────────────────────────────────────────
        if not title:
            fname = ""
            if self._lazy is not None:
                try:
                    fname = " — " + Path(str(self._lazy._img.get_filename())).name
                except Exception:
                    pass
            shape_str = "×".join(str(s) for s in shape3)
            vox_str   = "×".join(f"{v:.2f}" for v in (dx, dy, dz))
            vol_hint  = f" · t={t_idx}" if (self.n_volumes > 1 and t is not None) else ""
            title = f"{self.modality.upper()}{fname}  [{shape_str}]  {vox_str} mm{vol_hint}"

        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color="#ccc")),
            paper_bgcolor="#111", plot_bgcolor="#111",
            font_color="#aaa",
            margin=dict(l=5, r=45, t=65, b=5),
            height=430,
        )
        return fig

    def lightbox(
        self,
        axis: int = 2,
        *,
        n_slices: int = 25,
        n_cols: int = 5,
        step: int | None = None,
        title: str = "",
    ):
        """Return a plotly Figure with a grid of evenly spaced slices.

        Reads each slice directly from the nibabel ArrayProxy when available —
        never materialises the full volume.  Aspect ratios are corrected per
        panel using the image voxel sizes.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("lightbox() requires plotly: pip install plotly")

        # Determine n_total without loading data
        if self._lazy is not None:
            n_total = self._lazy.shape[axis]
        elif self._vol is not None:
            n_total = self._vol.shape[axis]
        else:
            raise RuntimeError("No volume data loaded")

        if step is not None:
            indices = list(range(0, n_total, step))[:n_slices]
        else:
            indices = np.round(np.linspace(0, n_total - 1, n_slices)).astype(int).tolist()

        n_rows = max(1, (len(indices) + n_cols - 1) // n_cols)
        axis_labels = {0: "x=", 1: "y=", 2: "z="}

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{axis_labels[axis]}{i}" for i in indices],
            horizontal_spacing=0.01, vertical_spacing=0.04,
        )

        # Aspect ratio per panel
        dx, dy, dz = self.voxel_sizes[:3]
        ar = {0: dz / dy, 1: dz / dx, 2: dy / dx}.get(axis, 1.0)

        for k, idx in enumerate(indices):
            row, col = divmod(k, n_cols)

            # Read one slice lazily from disk when possible
            if self._lazy is not None:
                slc = self._lazy.slice_along(axis, int(idx))
            else:
                vol3d = self._vol3d()
                slc = np.take(vol3d, int(idx), axis=axis)

            normed = apply_window(slc.T[::-1, :], self._vmin, self._vmax)
            fig.add_trace(
                go.Heatmap(
                    z=normed, colorscale="gray", zmin=0, zmax=1,
                    showscale=False, hoverinfo="skip", name=f"s{idx}",
                ),
                row=row + 1, col=col + 1,
            )

        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

        # Apply aspect correction to every panel (same ratio for all since same axis)
        for r in range(1, n_rows + 1):
            for c in range(1, n_cols + 1):
                try:
                    fig.update_yaxes(scaleratio=ar, row=r, col=c)
                except Exception:
                    pass

        axis_name = ("Sagittal", "Coronal", "Axial")[axis]
        if self._lazy is not None:
            shape3 = self._lazy.shape[:3]
        else:
            shape3 = (self._vol.shape[:3] if self._vol is not None else (0, 0, 0))
        shape_str = "×".join(str(s) for s in shape3)

        fig.update_layout(
            title=title or f"{self.modality.upper()} — {axis_name} lightbox  [{shape_str}]",
            paper_bgcolor="#111", plot_bgcolor="#111",
            font_color="#aaa",
            margin=dict(l=5, r=5, t=45, b=5),
            height=max(200, n_rows * 170),
        )
        return fig

    def timeseries_at(
        self,
        x: int,
        y: int,
        z: int,
        *,
        roi_radius: int = 0,
        title: str = "",
    ):
        """Plot BOLD/fMRI signal at voxel (x,y,z) over time."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("timeseries_at() requires plotly: pip install plotly")

        if self._vol is None and self._lazy is None:
            raise RuntimeError("Volume not loaded")

        if self.n_volumes > 1:
            if self._lazy is not None:
                signal = self._lazy.voxel_timeseries(x, y, z)
            else:
                if roi_radius > 0:
                    xs = slice(max(0, x - roi_radius), min(self._vol.shape[0], x + roi_radius + 1))
                    ys = slice(max(0, y - roi_radius), min(self._vol.shape[1], y + roi_radius + 1))
                    zs = slice(max(0, z - roi_radius), min(self._vol.shape[2], z + roi_radius + 1))
                    signal = self._vol[xs, ys, zs, :].mean(axis=(0, 1, 2))
                else:
                    signal = self._vol[x, y, z, :]
        else:
            vol3d = self._vol3d()
            signal = np.array([float(vol3d[x, y, z])])

        tr = self.tr or 1.0
        times = np.arange(len(signal)) * tr
        xlabel = "Time (s)" if self.tr else "Volume index"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=signal.tolist(),
            mode="lines", line=dict(color="#6af", width=1.5),
            name=f"vox ({x},{y},{z})",
        ))
        if roi_radius > 0:
            fig.update_traces(name=f"ROI r={roi_radius} around ({x},{y},{z})")

        fig.update_layout(
            title=title or f"BOLD signal — voxel ({x},{y},{z})",
            xaxis_title=xlabel, yaxis_title="Signal intensity",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            height=300,
        )
        return fig

    # ── Lazy slice rendering ──────────────────────────────────────────────

    def _render_lazy_axis(
        self,
        lazy: _LazyNIfTI,
        axis: int,
        max_slices: int,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[list[str], list[int]]:
        """Render slices along one axis from the lazy proxy."""
        n = lazy.shape[axis]
        if n > max_slices:
            idxs = np.round(np.linspace(0, n - 1, max_slices)).astype(int).tolist()
        else:
            idxs = list(range(n))
        _vmin = vmin if vmin is not None else self._vmin
        _vmax = vmax if vmax is not None else self._vmax
        b64s = []
        for i in idxs:
            slc = lazy.slice_along(axis, i)
            b64s.append(array_to_b64png(slc.T, _vmin, _vmax, self.colormap))
        return b64s, idxs

    # ── Interactive HTML ──────────────────────────────────────────────────

    def interactive_html(
        self,
        output: Path | str | None = None,
        *,
        title: str = "",
        max_slices_per_axis: int = 100,
        include_time_slider: bool = True,
        n_time_frames: int = 20,
    ) -> str:
        """Build a fully interactive orthogonal viewer as a standalone HTML page."""
        vox = self.voxel_sizes

        # Pre-render slices for all three axes
        if self._lazy is not None:
            slices_x, si_x = self._render_lazy_axis(self._lazy, 0, max_slices_per_axis)
            slices_y, si_y = self._render_lazy_axis(self._lazy, 1, max_slices_per_axis)
            slices_z, si_z = self._render_lazy_axis(self._lazy, 2, max_slices_per_axis)
            vol3d_shape = self._lazy.shape[:3]
        else:
            vol3d = self._vol3d()
            vol3d_shape = vol3d.shape

            def _render(axis: int) -> tuple[list[str], list[int]]:
                n = vol3d.shape[axis]
                if n > max_slices_per_axis:
                    idxs = np.round(np.linspace(0, n - 1, max_slices_per_axis)).astype(int).tolist()
                else:
                    idxs = list(range(n))
                slices_b64 = render_axis_slices(
                    vol3d, axis, self._vmin, self._vmax, self.colormap,
                    voxel_sizes=vox, max_slices=max_slices_per_axis,
                )
                return slices_b64, idxs

            slices_x, si_x = _render(0)
            slices_y, si_y = _render(1)
            slices_z, si_z = _render(2)

        nx, ny, nz = vol3d_shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        # Time slider pre-render (4D fMRI)
        slices_t: list[str] | None = None
        si_t: list[int] | None = None
        if include_time_slider and self.n_volumes > 1:
            t_idxs = np.round(
                np.linspace(0, self.n_volumes - 1, min(n_time_frames, self.n_volumes))
            ).astype(int).tolist()
            si_t = t_idxs
            if self._lazy is not None:
                slices_t = []
                for t in t_idxs:
                    frame = self._lazy.frame(t)
                    slices_t.append(
                        array_to_b64png(frame[:, :, cz].T, self._vmin, self._vmax, self.colormap)
                    )
            elif self._vol is not None:
                slices_t = [
                    array_to_b64png(
                        self._vol[:, :, cz, t].T,
                        self._vmin, self._vmax, self.colormap,
                    )
                    for t in t_idxs
                ]

        # Pre-render CT window stacks for interactive CT windowing
        ct_window_stacks: dict | None = None
        if self.modality == "ct" and self._lazy is not None:
            ct_window_stacks = {}
            for preset_name in ("brain", "soft_tissue", "bone", "lung"):
                preset = CT_PRESETS.get(preset_name)
                if not preset or preset.vmin is None:
                    continue
                wmin, wmax = float(preset.vmin), float(preset.vmax)
                sx, six = self._render_lazy_axis(self._lazy, 0, max_slices_per_axis, wmin, wmax)
                sy, siy = self._render_lazy_axis(self._lazy, 1, max_slices_per_axis, wmin, wmax)
                sz, siz = self._render_lazy_axis(self._lazy, 2, max_slices_per_axis, wmin, wmax)
                ct_window_stacks[preset_name] = {
                    "slices_x": sx, "si_x": six,
                    "slices_y": sy, "si_y": siy,
                    "slices_z": sz, "si_z": siz,
                }

        vmin_str = f"{self._vmin:.0f}"
        vmax_str = f"{self._vmax:.0f}"
        window_str = f"[{vmin_str}, {vmax_str}]"
        dataset_info = self._meta.get("series_description", "")
        modality = self._meta.get("modality", self.modality)

        html = build_interactive_html(
            title=title or f"{modality.upper()} Volume",
            dataset_info=dataset_info,
            modality=modality,
            shape=vol3d_shape,
            voxel_sizes=vox,
            vmin=self._vmin, vmax=self._vmax,
            window_str=window_str,
            slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
            si_x=si_x, si_y=si_y, si_z=si_z,
            cx=cx, cy=cy, cz=cz,
            n_volumes=self.n_volumes,
            tr=self.tr,
            slices_t=slices_t,
            si_t=si_t,
            ct_window_stacks=ct_window_stacks,
        )

        if output is not None:
            out_path = Path(output)
            out_path.write_text(html, encoding="utf-8")
            log.info("Wrote interactive viewer to %s", out_path)

        return html

    def to_html(self, output: Path | str, **kwargs) -> Path:
        """Write interactive HTML viewer to file. Returns the output Path."""
        out_path = Path(output)
        self.interactive_html(out_path, **kwargs)
        return out_path

    def show(self) -> None:
        """Open the interactive viewer in the default web browser."""
        import tempfile, webbrowser
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(self.interactive_html())
            tmp_path = f.name
        webbrowser.open(f"file://{tmp_path}")

    def __repr__(self) -> str:
        shape_str = " × ".join(str(s) for s in self.shape)
        return (
            f"VolumeViewer(modality={self.modality!r}, shape={shape_str}, "
            f"window=[{self._vmin:.0f}, {self._vmax:.0f}])"
        )
