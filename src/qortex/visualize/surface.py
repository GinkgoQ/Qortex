"""Surface, GIFTI, and CIFTI visual summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def surface_summary(path: Path | str, *, title: str = "") -> Any:
    """Return a Plotly summary figure for a GIFTI surface or CIFTI matrix.

    GIFTI meshes are rendered as a 3D cortical surface when coordinate and
    triangle arrays are present.  CIFTI dense matrices are summarized as a
    downsampled heatmap so users can inspect dimensionality and signal scale
    without loading an oversized full-resolution browser.
    """
    try:
        import nibabel as nib
        import plotly.graph_objects  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("surface_summary() requires nibabel and plotly") from exc

    src = Path(path)
    img = nib.load(str(src))
    name = img.__class__.__name__.lower()
    if "gifti" in name:
        return _gifti_summary(img, src, title=title)
    if "cifti" in name:
        return _cifti_summary(img, src, title=title)
    raise ValueError(f"Unsupported surface file type for {src}")


def _gifti_summary(img: Any, path: Path, *, title: str = "") -> Any:
    import plotly.graph_objects as go

    coords = None
    faces = None
    scalars: list[np.ndarray] = []

    for arr in getattr(img, "darrays", []):
        data = np.asarray(arr.data)
        if data.ndim != 2:
            if data.ndim == 1:
                scalars.append(data.astype(float, copy=False))
            continue
        if data.shape[1] == 3 and np.issubdtype(data.dtype, np.floating) and coords is None:
            coords = data.astype(float, copy=False)
        elif data.shape[1] == 3 and np.issubdtype(data.dtype, np.integer) and faces is None:
            faces = data.astype(np.int64, copy=False)
        elif data.shape[0] == 1 or data.shape[1] == 1:
            scalars.append(data.ravel().astype(float, copy=False))

    fig = go.Figure()
    if coords is not None and faces is not None:
        intensity = None
        for values in scalars:
            if values.size == coords.shape[0]:
                intensity = values
                break
        mesh_kwargs: dict[str, Any] = {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "i": faces[:, 0],
            "j": faces[:, 1],
            "k": faces[:, 2],
            "opacity": 1.0,
            "name": "surface",
            "showscale": intensity is not None,
        }
        if intensity is not None:
            mesh_kwargs["intensity"] = intensity
            mesh_kwargs["colorscale"] = "Viridis"
        else:
            mesh_kwargs["color"] = "#9aa4b2"
        fig.add_trace(go.Mesh3d(**mesh_kwargs))
        subtitle = f"{coords.shape[0]:,} vertices · {faces.shape[0]:,} faces"
    else:
        rows = [
            f"arrays: {len(getattr(img, 'darrays', []))}",
            f"coords: {'present' if coords is not None else 'missing'}",
            f"faces: {'present' if faces is not None else 'missing'}",
        ]
        fig.add_annotation(text="<br>".join(rows), showarrow=False, font=dict(size=14))
        subtitle = "mesh geometry incomplete"

    fig.update_layout(
        title=title or f"Surface QC — {path.name}<br><sup>{subtitle}</sup>",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#ddd",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            bgcolor="#111",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=560,
    )
    return fig


def _cifti_summary(img: Any, path: Path, *, title: str = "") -> Any:
    import plotly.graph_objects as go

    data = np.asarray(img.dataobj)
    if data.ndim == 1:
        matrix = data[np.newaxis, :]
    else:
        matrix = data.reshape(data.shape[0], -1)

    max_rows = min(200, matrix.shape[0])
    max_cols = min(400, matrix.shape[1])
    row_idx = np.linspace(0, matrix.shape[0] - 1, max_rows).astype(int)
    col_idx = np.linspace(0, matrix.shape[1] - 1, max_cols).astype(int)
    sampled = matrix[np.ix_(row_idx, col_idx)]
    finite = sampled[np.isfinite(sampled)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=sampled,
            zmin=float(vmin),
            zmax=float(vmax),
            colorscale="Viridis",
            colorbar=dict(title="value"),
        )
    )
    fig.update_layout(
        title=title or f"CIFTI QC — {path.name}<br><sup>shape {tuple(data.shape)} · sampled {sampled.shape}</sup>",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#ddd",
        xaxis_title="sampled greyordinates",
        yaxis_title="sampled rows / time",
        height=520,
    )
    return fig
