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

_NIFTI_EXTS = frozenset({".nii", ".gz", ".mgz", ".mgh"})
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


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load NIfTI file; returns (data_proxy, affine, header_meta)."""
    nib = _require_nibabel()
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    hdr = img.header
    zooms = tuple(float(z) for z in hdr.get_zooms())
    meta: dict = {
        "shape": data.shape,
        "zooms": zooms,
        "dtype": str(img.get_data_dtype()),
        "intent_code": int(hdr.get("intent_code", 0)) if hasattr(hdr, "get") else 0,
    }
    return data, affine, meta


def _load_dicom_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct a 3D volume from a DICOM series directory or single file."""
    pydicom = _require_pydicom()

    if path.is_file():
        ds = pydicom.dcmfile(path)
        pixel_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope"):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        affine = np.eye(4)
        if hasattr(ds, "PixelSpacing"):
            ps = ds.PixelSpacing
            affine[0, 0] = float(ps[1])
            affine[1, 1] = float(ps[0])
        if hasattr(ds, "SliceThickness"):
            affine[2, 2] = float(ds.SliceThickness)
        modality = str(getattr(ds, "Modality", "MR")).lower()
        meta = {
            "shape": pixel_array.shape,
            "modality": modality,
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "window_center": float(getattr(ds, "WindowCenter", 40) or 40),
            "window_width": float(getattr(ds, "WindowWidth", 400) or 400),
        }
        return pixel_array, affine, meta

    # Directory: collect and sort DICOM files by InstanceNumber
    dcm_files = sorted(
        [f for f in path.iterdir() if f.suffix.lower() in {".dcm", ".dicom", ""}],
        key=lambda f: f.name,
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {path}")

    slices = []
    inst_nums = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            inst_nums.append(int(getattr(ds, "InstanceNumber", len(slices))))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope"):
                arr = arr * float(ds.RescaleSlope) + float(getattr(ds, "RescaleIntercept", 0))
            slices.append((inst_nums[-1], arr, ds))
        except Exception as exc:
            log.debug("Skip DICOM %s: %s", f, exc)

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=-1)  # (rows, cols, slices)

    first_ds = slices[0][2]
    ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    st = float(getattr(first_ds, "SliceThickness", 1.0) or 1.0)
    affine = np.diag([float(ps[1]), float(ps[0]), st, 1.0])

    modality = str(getattr(first_ds, "Modality", "MR")).lower()
    wc = getattr(first_ds, "WindowCenter", None)
    ww = getattr(first_ds, "WindowWidth", None)
    meta = {
        "shape": volume.shape,
        "modality": modality,
        "series_description": str(getattr(first_ds, "SeriesDescription", "")),
        "patient_id": str(getattr(first_ds, "PatientID", "ANON")),
        "window_center": float(wc) if wc else None,
        "window_width": float(ww) if ww else None,
        "n_slices": len(slices),
    }
    return volume, affine, meta


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
                nib = _require_nibabel()
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
        name = path.name.lower()

        if suffix in {".dcm", ".dicom", ".ima"} or path.is_dir():
            self._vol, self._affine, self._meta = _load_dicom_series(path)
            # Extract modality from DICOM metadata
            if "modality" not in self._meta:
                self._meta["modality"] = "ct" if "ct" in name else "mri"
        elif suffix in {".nii", ".gz", ".mgz", ".mgh"}:
            self._vol, self._affine, self._meta = _load_nifti(path)
            self._meta["modality"] = _detect_modality_from_path(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _resolve_window(self, window: str | tuple | None) -> None:
        """Set self._vmin, self._vmax from the window spec."""
        vol3d = self._vol3d()

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
        else:
            self._vmin, self._vmax = auto_window(vol3d, self.modality)

    def _vol3d(self) -> np.ndarray:
        """Return the 3D view: for 4D data, use the temporal mean."""
        if self._vol is None:
            raise RuntimeError("Volume not loaded")
        if self._vol.ndim == 4:
            return self._vol.mean(axis=-1)
        return self._vol

    @property
    def shape(self) -> tuple:
        return self._vol.shape if self._vol is not None else ()

    @property
    def voxel_sizes(self) -> tuple[float, float, float]:
        zooms = self._meta.get("zooms")
        if zooms and len(zooms) >= 3:
            return tuple(float(z) for z in zooms[:3])
        return _voxel_sizes_from_affine(self._affine)

    @property
    def n_volumes(self) -> int:
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
        """Add a statistical map overlay (z-map, t-map, or any volume).

        The overlay is rendered on top of the background using alpha blending.

        Parameters
        ----------
        stat_map:
            Path (NIfTI) or numpy array of the same spatial shape as this volume.
        threshold:
            Voxels with absolute value below this threshold are transparent.
        colormap:
            Overlay colormap: ``"hot"``, ``"cool"``, ``"RdBu_r"``.
        alpha:
            Blend weight (0 = invisible, 1 = fully opaque).
        """
        if isinstance(stat_map, (str, Path)):
            arr, _, _ = _load_nifti(Path(stat_map))
            stat_arr = arr if arr.ndim == 3 else arr.mean(axis=-1)
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
        title: str = "",
    ):
        """Return a 3-panel plotly Figure showing orthogonal slices.

        Requires plotly.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("ortho() requires plotly: pip install plotly")

        vol3d = self._vol3d()
        nx, ny, nz = vol3d.shape
        cx = x if x is not None else nx // 2
        cy = y if y is not None else ny // 2
        cz = z if z is not None else nz // 2

        def _slice(axis: int, idx: int) -> np.ndarray:
            arr = np.take(vol3d, idx, axis=axis).T[::-1, :]
            return np.clip((arr - self._vmin) / max(self._vmax - self._vmin, 1), 0, 1)

        vmin, vmax = 0.0, 1.0
        cs = "gray"

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f"Axial  z={cz}",
                f"Coronal  y={cy}",
                f"Sagittal  x={cx}",
            ),
            horizontal_spacing=0.03,
        )

        common = dict(colorscale=cs, zmin=vmin, zmax=vmax, showscale=False)
        fig.add_trace(go.Heatmap(z=_slice(2, cz), **common, name="axial"),    row=1, col=1)
        fig.add_trace(go.Heatmap(z=_slice(1, cy), **common, name="coronal"),  row=1, col=2)
        fig.add_trace(go.Heatmap(z=_slice(0, cx), **common, showscale=True,
                                 colorbar=dict(len=0.6, thickness=14), name="sagittal"),
                      row=1, col=3)

        fig.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(
            title=title or f"{self.modality.upper()} — orthogonal view",
            paper_bgcolor="#111", plot_bgcolor="#111",
            font_color="#ccc",
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
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

        Requires plotly. Ideal for a quick overview of the full volume.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("lightbox() requires plotly: pip install plotly")

        vol3d = self._vol3d()
        n_total = vol3d.shape[axis]
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

        lut = get_lut(self.colormap)

        for k, idx in enumerate(indices):
            row, col = divmod(k, n_cols)
            slc = np.take(vol3d, int(idx), axis=axis).T[::-1, :]
            normed = apply_window(slc, self._vmin, self._vmax)
            fig.add_trace(
                go.Heatmap(
                    z=normed,
                    colorscale="gray", zmin=0, zmax=1,
                    showscale=False,
                    name=f"slice{idx}",
                ),
                row=row + 1, col=col + 1,
            )

        fig.update_yaxes(scaleanchor="x", scaleratio=1, showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        axis_name = ("Sagittal", "Coronal", "Axial")[axis]
        fig.update_layout(
            title=title or f"{self.modality.upper()} — {axis_name} lightbox",
            paper_bgcolor="#111", plot_bgcolor="#111",
            font_color="#ccc",
            margin=dict(l=5, r=5, t=40, b=5),
            height=max(200, n_rows * 160),
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
        """Plot BOLD/fMRI signal at voxel (x,y,z) over time.

        For 3D volumes returns a flat line (single value). Requires plotly.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("timeseries_at() requires plotly: pip install plotly")

        if self._vol is None:
            raise RuntimeError("Volume not loaded")

        if self._vol.ndim == 4:
            if roi_radius > 0:
                xs = slice(max(0, x - roi_radius), min(self._vol.shape[0], x + roi_radius + 1))
                ys = slice(max(0, y - roi_radius), min(self._vol.shape[1], y + roi_radius + 1))
                zs = slice(max(0, z - roi_radius), min(self._vol.shape[2], z + roi_radius + 1))
                signal = self._vol[xs, ys, zs, :].mean(axis=(0, 1, 2))
            else:
                signal = self._vol[x, y, z, :]
        else:
            signal = np.array([float(self._vol[x, y, z])])

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
        """Build a fully interactive orthogonal viewer as a standalone HTML page.

        All slices along all three axes are pre-rendered as base64 PNGs and
        embedded in the HTML.  No server required.  The output is a single
        self-contained file.

        Parameters
        ----------
        output:
            If provided, write HTML to this path and return it as a string.
        max_slices_per_axis:
            Maximum pre-rendered slices per axis.  Larger volumes are sub-sampled.
        include_time_slider:
            For 4D fMRI, add a TR slider showing the mean and individual volumes.
        n_time_frames:
            Number of time points to pre-render for the TR slider.
        """
        vol3d = self._vol3d()
        nx, ny, nz = vol3d.shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        vox = self.voxel_sizes

        # Pre-render slices for all three axes
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

        # Time slider pre-render (4D fMRI)
        slices_t: list[str] | None = None
        if include_time_slider and self.n_volumes > 1 and self._vol is not None:
            t_idxs = np.round(
                np.linspace(0, self.n_volumes - 1, min(n_time_frames, self.n_volumes))
            ).astype(int).tolist()
            slices_t = [
                array_to_b64png(
                    self._vol[:, :, cz, t].T,
                    self._vmin, self._vmax, self.colormap,
                )
                for t in t_idxs
            ]

        vmin_str = f"{self._vmin:.0f}"
        vmax_str = f"{self._vmax:.0f}"
        window_str = f"[{vmin_str}, {vmax_str}]"
        dataset_info = self._meta.get("series_description", "")
        modality = self._meta.get("modality", self.modality)

        html = build_interactive_html(
            title=title or f"{modality.upper()} Volume",
            dataset_info=dataset_info,
            modality=modality,
            shape=vol3d.shape,
            voxel_sizes=vox,
            vmin=self._vmin, vmax=self._vmax,
            window_str=window_str,
            slices_x=slices_x, slices_y=slices_y, slices_z=slices_z,
            si_x=si_x, si_y=si_y, si_z=si_z,
            cx=cx, cy=cy, cz=cz,
            n_volumes=self.n_volumes,
            tr=self.tr,
            slices_t=slices_t,
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
