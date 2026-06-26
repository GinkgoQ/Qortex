"""DWI-specific visualization: b0 preview, diffusion direction, gradient sphere.

Usage
-----
    from qortex.visualize.dwi import DWIViewer, dwi_summary

    viewer = DWIViewer("sub-01_dwi.nii.gz", bval_path="sub-01.bval", bvec_path="sub-01.bvec")
    fig = viewer.dwi_summary()   # 4-panel figure
    fig.show()

    # Or standalone
    fig = dwi_summary("sub-01_dwi.nii.gz")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── b-value / b-vector loaders ────────────────────────────────────────────────

def _load_bvals(path: Path) -> np.ndarray:
    """Load FSL-style .bval file → 1D array of b-values."""
    text = path.read_text().strip()
    return np.array([float(v) for v in text.split()], dtype=np.float32)


def _load_bvecs(path: Path) -> np.ndarray:
    """Load FSL-style .bvec file → (3, n_volumes) array of gradient directions."""
    lines = path.read_text().strip().splitlines()
    rows = [[float(v) for v in line.split()] for line in lines if line.strip()]
    arr = np.array(rows, dtype=np.float32)
    if arr.shape[0] == 3:
        return arr          # (3, n_vols)
    if arr.shape[1] == 3:
        return arr.T        # (n_vols, 3) → transpose
    raise ValueError(f"Unexpected bvec shape: {arr.shape}")


def _find_companions(dwi_path: Path, bval_path: Path | None, bvec_path: Path | None):
    """Auto-detect bval/bvec companions next to the DWI file."""
    stem = dwi_path.name
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    parent = dwi_path.parent

    def _find(hint: Path | None, ext: str) -> Path | None:
        if hint is not None and Path(hint).exists():
            return Path(hint)
        candidate = parent / f"{stem}{ext}"
        return candidate if candidate.exists() else None

    return _find(bval_path, ".bval"), _find(bvec_path, ".bvec")


# ── DWIViewer ─────────────────────────────────────────────────────────────────

class DWIViewer:
    """Lazy DWI-specific viewer.

    Reads the minimum data needed for each panel:
    - b0 image: volumes where bval < 50
    - High-b image: volumes where bval > 800
    - b-value histogram: just the bval file (no image data)
    - Gradient sphere: just the bvec file (no image data)

    Parameters
    ----------
    dwi_path:
        Path to the 4D DWI NIfTI (.nii or .nii.gz).
    bval_path:
        Path to the FSL .bval file.  Auto-detected from the NIfTI name if None.
    bvec_path:
        Path to the FSL .bvec file.  Auto-detected from the NIfTI name if None.
    """

    def __init__(
        self,
        dwi_path: Any,
        bval_path: Path | str | None = None,
        bvec_path: Path | str | None = None,
    ) -> None:
        self._path = Path(dwi_path)
        bval_p, bvec_p = _find_companions(
            self._path,
            Path(bval_path) if bval_path else None,
            Path(bvec_path) if bvec_path else None,
        )
        self._bvals: np.ndarray | None = _load_bvals(bval_p) if bval_p else None
        self._bvecs: np.ndarray | None = _load_bvecs(bvec_p) if bvec_p else None

        from qortex.visualize.volume import _LazyNIfTI
        self._lazy = _LazyNIfTI(self._path)
        self._shape = self._lazy.shape  # (nx, ny, nz, n_vols)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_volumes(self) -> int:
        return self._shape[3] if len(self._shape) == 4 else 1

    @property
    def b0_indices(self) -> list[int]:
        """Volume indices where b-value < 50 (b0 images)."""
        if self._bvals is None:
            return [0]
        return [i for i, b in enumerate(self._bvals) if b < 50]

    @property
    def high_b_indices(self) -> list[int]:
        """Volume indices with the highest b-value (diffusion-weighted images)."""
        if self._bvals is None:
            return [self.n_volumes - 1]
        bmax = float(self._bvals.max())
        thresh = max(800.0, bmax * 0.8)
        idxs = [i for i, b in enumerate(self._bvals) if b >= thresh]
        return idxs if idxs else [int(np.argmax(self._bvals))]

    @property
    def shells(self) -> dict[int, int]:
        """Map rounded b-value shell → number of directions."""
        if self._bvals is None:
            return {}
        rounded = (np.round(self._bvals / 50.0) * 50).astype(int)
        shells: dict[int, int] = {}
        for b in rounded:
            shells[int(b)] = shells.get(int(b), 0) + 1
        return dict(sorted(shells.items()))

    # ── Panel rendering ───────────────────────────────────────────────────

    def b0(self, *, title: str = "") -> Any:
        """Orthogonal view of the mean b0 image."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("b0() requires plotly: pip install plotly")

        idxs = self.b0_indices
        shape3 = self._shape[:3]
        nx, ny, nz = shape3
        cx, cy, cz = nx // 2, ny // 2, nz // 2

        # Mean b0 from all b0 volumes
        b0_vol = np.zeros(shape3, dtype=np.float64)
        for t in idxs:
            b0_vol += np.asarray(self._lazy._proxy[..., t]).astype(np.float64)
        b0_vol /= max(1, len(idxs))
        b0_vol = b0_vol.astype(np.float32)

        vmax = float(np.percentile(b0_vol[b0_vol > 0], 98)) if (b0_vol > 0).any() else 1.0
        vmin = float(np.percentile(b0_vol[b0_vol > 0], 1.0)) if (b0_vol > 0).any() else 0.0

        def _norm(slc):
            return np.clip((slc - vmin) / max(vmax - vmin, 1e-8), 0, 1)

        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=(f"Axial z={cz}", f"Coronal y={cy}", f"Sagittal x={cx}"),
                             horizontal_spacing=0.04)
        fig.add_trace(go.Heatmap(z=_norm(b0_vol[:, :, cz].T[::-1, :]), colorscale="Gray",
                                  zmin=0, zmax=1, showscale=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=_norm(b0_vol[:, cy, :].T[::-1, :]), colorscale="Gray",
                                  zmin=0, zmax=1, showscale=False, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=_norm(b0_vol[cx, :, :].T[::-1, :]), colorscale="Gray",
                                  zmin=0, zmax=1, showscale=True,
                                  colorbar=dict(len=0.7, thickness=10, title="Norm."),
                                  hoverinfo="skip"), row=1, col=3)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(
            title=title or f"b0 Image  (mean of {len(idxs)} b0 volumes)",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#aaa",
            margin=dict(l=5, r=45, t=60, b=5), height=350,
        )
        return fig

    def high_b(self, *, title: str = "") -> Any:
        """Center-slice view of the mean high-b diffusion-weighted image."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("high_b() requires plotly: pip install plotly")

        idxs = self.high_b_indices
        shape3 = self._shape[:3]
        nx, ny, nz = shape3
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        bmax_str = f"{int(self._bvals[idxs[0]])}" if self._bvals is not None else "max-b"

        dwi_vol = np.zeros(shape3, dtype=np.float64)
        for t in idxs[:4]:  # average at most 4 directions to keep it fast
            dwi_vol += np.asarray(self._lazy._proxy[..., t]).astype(np.float64)
        dwi_vol /= max(1, min(len(idxs), 4))
        dwi_vol = dwi_vol.astype(np.float32)

        vmax = float(np.percentile(dwi_vol[dwi_vol > 0], 99)) if (dwi_vol > 0).any() else 1.0
        vmin = 0.0

        def _norm(slc):
            return np.clip((slc - vmin) / max(vmax - vmin, 1e-8), 0, 1)

        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=(f"Axial z={cz}", f"Coronal y={cy}", f"Sagittal x={cx}"),
                             horizontal_spacing=0.04)
        for col, slc in enumerate([
            dwi_vol[:, :, cz].T[::-1, :],
            dwi_vol[:, cy, :].T[::-1, :],
            dwi_vol[cx, :, :].T[::-1, :],
        ], start=1):
            fig.add_trace(go.Heatmap(z=_norm(slc), colorscale="Gray",
                                      zmin=0, zmax=1, showscale=(col == 3),
                                      colorbar=dict(len=0.7, thickness=10) if col == 3 else None,
                                      hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(
            title=title or f"High-b DWI  (b={bmax_str}, mean of {min(len(idxs),4)} directions)",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#aaa",
            margin=dict(l=5, r=45, t=60, b=5), height=350,
        )
        return fig

    def bval_histogram(self, *, title: str = "") -> Any:
        """Bar chart of b-value shell distribution — reads only the .bval file."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("bval_histogram() requires plotly: pip install plotly")

        if self._bvals is None:
            raise RuntimeError("No .bval file found; cannot plot b-value histogram.")

        shells = self.shells
        fig = go.Figure(go.Bar(
            x=[str(b) for b in shells.keys()],
            y=list(shells.values()),
            marker_color="#6af",
            text=list(shells.values()),
            textposition="outside",
        ))
        fig.update_layout(
            title=title or f"b-value Shell Distribution  ({self.n_volumes} volumes)",
            xaxis_title="b-value (s/mm²)", yaxis_title="# directions",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
            bargap=0.25, height=300,
        )
        return fig

    def gradient_sphere(self, *, title: str = "") -> Any:
        """3D scatter plot of gradient directions on the unit sphere.

        b0 volumes (bval < 50) appear as grey markers at the origin.
        Each non-zero shell gets a distinct color.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("gradient_sphere() requires plotly: pip install plotly")

        if self._bvecs is None:
            raise RuntimeError("No .bvec file found; cannot render gradient sphere.")

        bvecs = self._bvecs      # (3, n_vols)
        bvals = self._bvals if self._bvals is not None else np.zeros(bvecs.shape[1])

        # Normalize each direction to unit sphere
        norms = np.linalg.norm(bvecs, axis=0)
        norms[norms < 1e-6] = 1.0
        dirs = bvecs / norms     # (3, n_vols)

        # Shell coloring
        rounded = (np.round(bvals / 50.0) * 50).astype(int)
        unique_shells = sorted(set(rounded.tolist()))
        palette = ["#888", "#6af", "#fa6", "#6fa", "#f6a", "#a6f", "#ff6", "#6ff"]
        shell_color = {b: palette[i % len(palette)] for i, b in enumerate(unique_shells)}

        fig = go.Figure()
        for shell_b in unique_shells:
            mask = rounded == shell_b
            x, y, z = dirs[0, mask], dirs[1, mask], dirs[2, mask]
            label = "b0" if shell_b < 50 else f"b={shell_b}"
            fig.add_trace(go.Scatter3d(
                x=np.concatenate([x, -x]).tolist(),
                y=np.concatenate([y, -y]).tolist(),
                z=np.concatenate([z, -z]).tolist(),
                mode="markers",
                name=label,
                marker=dict(
                    size=5 if shell_b >= 50 else 4,
                    color=shell_color[shell_b],
                    opacity=0.85 if shell_b >= 50 else 0.5,
                ),
            ))

        # Unit sphere wireframe (latitude/longitude lines)
        u = np.linspace(0, 2 * np.pi, 60)
        for lv in np.linspace(0, np.pi, 7):
            fig.add_trace(go.Scatter3d(
                x=(np.cos(u) * np.sin(lv)).tolist(),
                y=(np.sin(u) * np.sin(lv)).tolist(),
                z=([np.cos(lv)] * len(u)),
                mode="lines", showlegend=False,
                line=dict(color="#333", width=1),
            ))

        fig.update_layout(
            title=title or f"Gradient Directions  ({bvecs.shape[1]} volumes, {len(unique_shells)} shells)",
            scene=dict(
                xaxis=dict(showticklabels=False, title="", backgroundcolor="#111"),
                yaxis=dict(showticklabels=False, title="", backgroundcolor="#111"),
                zaxis=dict(showticklabels=False, title="", backgroundcolor="#111"),
                bgcolor="#111",
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            ),
            paper_bgcolor="#111", font_color="#aaa",
            legend=dict(font=dict(color="#aaa"), bgcolor="#1a1a1a"),
            height=420,
        )
        return fig

    def dwi_summary(self, *, title: str = "") -> Any:
        """2×2 DWI QC summary figure.

        Panels:
        1. Mean b0 — axial center slice
        2. High-b DWI — axial center slice
        3. b-value histogram
        4. Gradient sphere
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("dwi_summary() requires plotly: pip install plotly")

        shape3 = self._shape[:3]
        cz = shape3[2] // 2

        # b0 slice
        b0_idxs = self.b0_indices
        b0_vol = np.zeros(shape3, dtype=np.float64)
        for t in b0_idxs:
            b0_vol += np.asarray(self._lazy._proxy[..., t]).astype(np.float64)
        b0_vol /= max(1, len(b0_idxs))
        b0_slc = b0_vol[:, :, cz].T[::-1, :].astype(np.float32)
        pos_b0 = b0_slc[b0_slc > 0]
        b0_vmin = float(np.percentile(pos_b0, 1.0)) if pos_b0.size else 0.0
        b0_vmax = float(np.percentile(pos_b0, 99.0)) if pos_b0.size else 1.0

        # High-b slice
        hb_idxs = self.high_b_indices
        dwi_vol = np.zeros(shape3, dtype=np.float64)
        for t in hb_idxs[:3]:
            dwi_vol += np.asarray(self._lazy._proxy[..., t]).astype(np.float64)
        dwi_vol /= max(1, min(len(hb_idxs), 3))
        dwi_slc = dwi_vol[:, :, cz].T[::-1, :].astype(np.float32)
        pos_dwi = dwi_slc[dwi_slc > 0]
        dwi_vmax = float(np.percentile(pos_dwi, 99.0)) if pos_dwi.size else 1.0

        def _norm(slc, vmin, vmax):
            return np.clip((slc - vmin) / max(vmax - vmin, 1e-8), 0, 1)

        bmax_str = f"b={int(self._bvals[hb_idxs[0]])}" if self._bvals is not None else "high-b"
        has_bvals = self._bvals is not None
        has_bvecs = self._bvecs is not None

        # Build subplot grid — all panels use xy axes; the gradient sphere panel
        # uses a 2D azimuthal-equidistant projection (go.Scatter), NOT go.Scatter3d,
        # so it does not need a "scene" subplot type.
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ]
        subtitles = (
            f"Mean b0 ({len(b0_idxs)} vol{'s' if len(b0_idxs)>1 else ''})",
            f"High-b DWI ({bmax_str})",
            "b-value Distribution" if has_bvals else "No .bval file",
            "Gradient Sphere" if has_bvecs else "No .bvec file",
        )
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subtitles,
            specs=specs,
            horizontal_spacing=0.06, vertical_spacing=0.14,
        )

        # b0 image
        fig.add_trace(go.Heatmap(z=_norm(b0_slc, b0_vmin, b0_vmax), colorscale="Gray",
                                  zmin=0, zmax=1, showscale=False, hoverinfo="skip"),
                       row=1, col=1)

        # High-b image
        fig.add_trace(go.Heatmap(z=_norm(dwi_slc, 0.0, dwi_vmax), colorscale="Gray",
                                  zmin=0, zmax=1, showscale=False, hoverinfo="skip"),
                       row=1, col=2)

        # b-value histogram
        if has_bvals:
            shells = self.shells
            fig.add_trace(go.Bar(x=[str(b) for b in shells.keys()],
                                  y=list(shells.values()),
                                  marker_color="#6af", showlegend=False),
                           row=2, col=1)
            fig.update_xaxes(title_text="b-value", row=2, col=1, color="#888")
            fig.update_yaxes(title_text="# dirs", row=2, col=1, color="#888")

        # Gradient sphere (2D projection when embedded in subplot)
        if has_bvecs and self._bvals is not None:
            bvecs = self._bvecs
            bvals = self._bvals
            norms = np.linalg.norm(bvecs, axis=0)
            norms[norms < 1e-6] = 1.0
            dirs = bvecs / norms
            rounded = (np.round(bvals / 50.0) * 50).astype(int)
            unique_shells = sorted(set(rounded.tolist()))
            palette = ["#888", "#6af", "#fa6", "#6fa", "#f6a"]
            for i, shell_b in enumerate(unique_shells):
                mask = rounded == shell_b
                theta = np.arccos(np.clip(dirs[2, mask], -1, 1))
                phi = np.arctan2(dirs[1, mask], dirs[0, mask])
                # Azimuthal equidistant projection → 2D scatter
                x_proj = (theta / np.pi) * np.cos(phi)
                y_proj = (theta / np.pi) * np.sin(phi)
                label = "b0" if shell_b < 50 else f"b={shell_b}"
                fig.add_trace(go.Scatter(
                    x=x_proj.tolist(), y=y_proj.tolist(),
                    mode="markers", name=label,
                    marker=dict(size=6, color=palette[i % len(palette)], opacity=0.8),
                ), row=2, col=2)

        fig.update_xaxes(showticklabels=False, showgrid=False, row=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1)
        fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)

        shape_str = "×".join(str(s) for s in self._shape)
        shells_str = ", ".join(f"b={k}({v})" for k, v in self.shells.items()) if has_bvals else "unknown shells"
        fig.update_layout(
            title=title or f"DWI QC Summary  [{shape_str}]  {shells_str}",
            paper_bgcolor="#111", plot_bgcolor="#111", font_color="#888",
            margin=dict(l=5, r=20, t=80, b=20),
            height=560,
            legend=dict(font=dict(color="#aaa"), bgcolor="#1a1a1a"),
        )
        return fig

    def contact_sheet(
        self,
        shell: int | str | None = None,
        *,
        n_slices: int = 9,
        n_cols: int = 3,
        title: str = "",
    ) -> Any:
        """Axial DWI montage for slice-dropout and distortion QC.

        Parameters
        ----------
        shell:
            ``None`` or ``"b0"`` uses b0 volumes. An integer selects the nearest
            rounded b-value shell. ``"high"`` uses the highest-b shell.
        n_slices:
            Number of evenly spaced axial slices to show.
        n_cols:
            Number of montage columns.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("contact_sheet() requires plotly: pip install plotly")

        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
        if n_cols <= 0:
            raise ValueError("n_cols must be positive")

        vol = self._mean_shell_volume(shell)
        nz = vol.shape[2]
        indices = np.round(np.linspace(0, nz - 1, min(n_slices, nz))).astype(int).tolist()
        n_rows = max(1, (len(indices) + n_cols - 1) // n_cols)

        positive = vol[vol > 0]
        vmin = float(np.percentile(positive, 1.0)) if positive.size else float(np.percentile(vol, 1.0))
        vmax = float(np.percentile(positive, 99.0)) if positive.size else float(np.percentile(vol, 99.0))
        if vmin == vmax:
            vmax = vmin + 1.0

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"z={idx}" for idx in indices],
            horizontal_spacing=0.01,
            vertical_spacing=0.05,
        )
        for k, z in enumerate(indices):
            row, col = divmod(k, n_cols)
            slc = vol[:, :, z].T[::-1, :]
            norm = np.clip((slc - vmin) / max(vmax - vmin, 1e-8), 0, 1)
            fig.add_trace(
                go.Heatmap(
                    z=norm,
                    colorscale="Gray",
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=row + 1,
                col=col + 1,
            )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        shell_label = self._shell_label(shell)
        fig.update_layout(
            title=title or f"DWI contact sheet — {shell_label}",
            paper_bgcolor="#111",
            plot_bgcolor="#111",
            font_color="#aaa",
            margin=dict(l=5, r=5, t=55, b=5),
            height=max(260, n_rows * 170),
        )
        return fig

    def _shell_indices(self, shell: int | str | None) -> list[int]:
        if shell is None or str(shell).lower() == "b0":
            return self.b0_indices
        if str(shell).lower() in {"high", "high-b", "max"}:
            return self.high_b_indices
        if self._bvals is None:
            return [0]
        target = int(shell)
        rounded = (np.round(self._bvals / 50.0) * 50).astype(int)
        unique = np.array(sorted(set(rounded.tolist())), dtype=int)
        nearest = int(unique[np.argmin(np.abs(unique - target))])
        idxs = [i for i, b in enumerate(rounded) if int(b) == nearest]
        return idxs or [int(np.argmin(np.abs(rounded - target)))]

    def _mean_shell_volume(self, shell: int | str | None) -> np.ndarray:
        idxs = self._shell_indices(shell)
        shape3 = self._shape[:3]
        if len(self._shape) == 3:
            return np.asarray(self._lazy._proxy).astype(np.float32)
        acc = np.zeros(shape3, dtype=np.float64)
        for t in idxs:
            acc += np.asarray(self._lazy._proxy[..., int(t)]).astype(np.float64)
        return (acc / max(1, len(idxs))).astype(np.float32)

    def _shell_label(self, shell: int | str | None) -> str:
        if shell is None or str(shell).lower() == "b0":
            return f"b0 ({len(self.b0_indices)} vol{'s' if len(self.b0_indices) != 1 else ''})"
        if str(shell).lower() in {"high", "high-b", "max"}:
            return f"high-b ({len(self.high_b_indices)} vol{'s' if len(self.high_b_indices) != 1 else ''})"
        return f"b≈{shell}"

    def __repr__(self) -> str:
        shape_str = "×".join(str(s) for s in self._shape)
        shells_str = str(self.shells) if self._bvals is not None else "no bval"
        return f"DWIViewer(shape={shape_str}, shells={shells_str})"


# ── Convenience function ──────────────────────────────────────────────────────

def dwi_summary(
    dwi_path: Any,
    bval_path: Path | str | None = None,
    bvec_path: Path | str | None = None,
    *,
    title: str = "",
) -> Any:
    """Return a 4-panel DWI QC summary Plotly figure.

    Reads the minimum data needed:
    - b-value / b-vector files for histogram and gradient sphere (no NIfTI I/O)
    - Mean b0 and mean high-b slices (one 3D frame each)

    Parameters
    ----------
    dwi_path:
        Path to the 4D DWI NIfTI.
    bval_path:
        Path to the .bval file.  Auto-detected from ``dwi_path`` if None.
    bvec_path:
        Path to the .bvec file.  Auto-detected from ``dwi_path`` if None.
    title:
        Override the auto-generated title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    return DWIViewer(dwi_path, bval_path=bval_path, bvec_path=bvec_path).dwi_summary(title=title)
