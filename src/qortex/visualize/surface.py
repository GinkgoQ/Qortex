"""Surface, GIFTI, and CIFTI visual inspection.

This module intentionally separates structural inspection from rendering.  A
GIFTI/CIFTI file is first summarized into typed array/axis records, then the
summary renderer uses those records to choose a mesh, scalar map, label map, or
matrix preview.  It is still a lightweight QC viewer, not a replacement for
Connectome Workbench, but it now exposes the details needed for real review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceArrayInfo:
    """One data array inside a GIFTI file."""

    index: int
    role: str
    intent: str
    shape: tuple[int, ...]
    dtype: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class SurfaceInfo:
    """Header-level surface summary for GIFTI/CIFTI files."""

    path: Path
    family: str
    shape: tuple[int, ...]
    hemisphere: str | None = None
    arrays: list[SurfaceArrayInfo] = field(default_factory=list)
    axes: list[dict[str, Any]] = field(default_factory=list)
    n_vertices: int | None = None
    n_faces: int | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    labels: dict[int, str] = field(default_factory=dict)


def inspect_surface(path: Path | str) -> SurfaceInfo:
    """Inspect a GIFTI/CIFTI file without building a rendered figure."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("inspect_surface() requires nibabel") from exc

    src = Path(path)
    img = nib.load(str(src))
    name = img.__class__.__name__.lower()
    if "gifti" in name:
        return _inspect_gifti(img, src)
    if "cifti" in name:
        return _inspect_cifti(img, src)
    raise ValueError(f"Unsupported surface file type for {src}")


def surface_summary(path: Path | str, *, title: str = "") -> Any:
    """Return a Plotly QC figure for a GIFTI surface or CIFTI dense matrix."""
    try:
        import nibabel as nib
        import plotly.graph_objects  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("surface_summary() requires nibabel and plotly") from exc

    src = Path(path)
    img = nib.load(str(src))
    info = inspect_surface(src)
    if info.family == "gifti":
        return _gifti_summary(img, info, title=title)
    if info.family == "cifti":
        return _cifti_summary(img, info, title=title)
    raise ValueError(f"Unsupported surface file type for {src}")


def find_hemisphere_pair(path: Path | str) -> Path | None:
    """Return the likely opposite-hemisphere GIFTI/CIFTI companion if present."""
    src = Path(path)
    name = src.name
    replacements = [
        ("hemi-L", "hemi-R"),
        ("hemi-R", "hemi-L"),
        (".L.", ".R."),
        (".R.", ".L."),
        ("_L.", "_R."),
        ("_R.", "_L."),
        ("left", "right"),
        ("right", "left"),
    ]
    for old, new in replacements:
        if old in name:
            candidate = src.with_name(name.replace(old, new, 1))
            if candidate.exists():
                return candidate
    return None


def _inspect_gifti(img: Any, path: Path) -> SurfaceInfo:
    arrays: list[SurfaceArrayInfo] = []
    coords = None
    faces = None
    labels: dict[int, str] = {}

    for i, arr in enumerate(getattr(img, "darrays", [])):
        data = np.asarray(arr.data)
        role = _gifti_array_role(arr, data)
        finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.number) else np.asarray([])
        arrays.append(
            SurfaceArrayInfo(
                index=i,
                role=role,
                intent=_intent_label(arr),
                shape=tuple(int(s) for s in data.shape),
                dtype=str(data.dtype),
                min_value=float(np.min(finite)) if finite.size else None,
                max_value=float(np.max(finite)) if finite.size else None,
            )
        )
        if role == "coordinates" and coords is None:
            coords = data.astype(float, copy=False)
        elif role == "triangles" and faces is None:
            faces = data.astype(np.int64, copy=False)

    labeltable = getattr(img, "labeltable", None)
    if labeltable is not None:
        for item in getattr(labeltable, "labels", []) or []:
            key = int(getattr(item, "key", 0))
            label = str(getattr(item, "label", "") or key)
            labels[key] = label

    bounds = None
    if coords is not None and coords.ndim == 2 and coords.shape[1] == 3:
        bounds = {
            axis: (float(coords[:, j].min()), float(coords[:, j].max()))
            for j, axis in enumerate(("x", "y", "z"))
        }

    return SurfaceInfo(
        path=path,
        family="gifti",
        shape=tuple(),
        hemisphere=_guess_hemisphere(path),
        arrays=arrays,
        n_vertices=int(coords.shape[0]) if coords is not None and coords.ndim == 2 else None,
        n_faces=int(faces.shape[0]) if faces is not None and faces.ndim == 2 else None,
        bounds=bounds,
        labels=labels,
    )


def _inspect_cifti(img: Any, path: Path) -> SurfaceInfo:
    axes: list[dict[str, Any]] = []
    try:
        for i in range(len(img.shape)):
            axis = img.header.get_axis(i)
            row: dict[str, Any] = {
                "index": i,
                "type": axis.__class__.__name__,
                "size": int(len(axis)),
            }
            if hasattr(axis, "name"):
                row["name"] = str(axis.name)
            if hasattr(axis, "volume_shape"):
                row["volume_shape"] = tuple(int(v) for v in axis.volume_shape) if axis.volume_shape else None
            if hasattr(axis, "iter_structures"):
                structures = []
                for structure in axis.iter_structures():
                    structures.append(str(structure[0]))
                row["structures"] = structures[:16]
                row["n_structures"] = len(structures)
            axes.append(row)
    except Exception:
        axes = [{"index": i, "type": "unknown", "size": int(s)} for i, s in enumerate(img.shape)]

    return SurfaceInfo(
        path=path,
        family="cifti",
        shape=tuple(int(s) for s in img.shape),
        hemisphere=_guess_hemisphere(path),
        axes=axes,
    )


def _gifti_array_role(arr: Any, data: np.ndarray) -> str:
    label = _intent_label(arr).lower()
    if "pointset" in label:
        return "coordinates"
    if "triangle" in label:
        return "triangles"
    if "label" in label:
        return "labels"
    if data.ndim == 2 and data.shape[1] == 3 and np.issubdtype(data.dtype, np.floating):
        return "coordinates"
    if data.ndim == 2 and data.shape[1] == 3 and np.issubdtype(data.dtype, np.integer):
        return "triangles"
    if data.ndim == 1 and np.issubdtype(data.dtype, np.integer):
        return "labels"
    if data.ndim in {1, 2} and np.issubdtype(data.dtype, np.number):
        return "scalars"
    return "metadata"


def _intent_label(arr: Any) -> str:
    intent = getattr(arr, "intent", None)
    try:
        from nibabel.nifti1 import intent_codes
        return str(intent_codes.label[int(intent)])
    except Exception:
        return str(intent or "unknown")


def _guess_hemisphere(path: Path) -> str | None:
    name = path.name.lower()
    if "hemi-l" in name or ".l." in name or "_l." in name or "left" in name:
        return "L"
    if "hemi-r" in name or ".r." in name or "_r." in name or "right" in name:
        return "R"
    return None


def _gifti_arrays(img: Any) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray], list[np.ndarray]]:
    coords = None
    faces = None
    scalars: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for arr in getattr(img, "darrays", []):
        data = np.asarray(arr.data)
        role = _gifti_array_role(arr, data)
        if role == "coordinates" and coords is None:
            coords = data.astype(float, copy=False)
        elif role == "triangles" and faces is None:
            faces = data.astype(np.int64, copy=False)
        elif role == "labels":
            labels.append(data.ravel())
        elif role == "scalars":
            scalars.append(data.ravel().astype(float, copy=False))
    return coords, faces, scalars, labels


def _gifti_summary(img: Any, info: SurfaceInfo, *, title: str = "") -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    coords, faces, scalars, labels = _gifti_arrays(img)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "table"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.03,
    )

    subtitle = _surface_subtitle(info)
    if coords is not None and faces is not None:
        intensity = _first_vertex_values(scalars + labels, coords.shape[0])
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
            "hovertemplate": "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        }
        if intensity is not None:
            mesh_kwargs["intensity"] = intensity
            mesh_kwargs["colorscale"] = "Viridis"
            mesh_kwargs["hovertemplate"] += "<br>value=%{intensity:.3g}"
        else:
            mesh_kwargs["color"] = "#9aa4b2"
        fig.add_trace(go.Mesh3d(**mesh_kwargs), row=1, col=1)
    else:
        fig.add_annotation(text="GIFTI mesh geometry incomplete", showarrow=False, font=dict(size=14))

    fig.add_trace(_info_table(info), row=1, col=2)
    fig.update_layout(
        title=title or f"Surface QC — {info.path.name}<br><sup>{subtitle}</sup>",
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
        margin=dict(l=0, r=0, t=70, b=0),
        height=600,
    )
    return fig


def _cifti_summary(img: Any, info: SurfaceInfo, *, title: str = "") -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sampled = _sample_cifti_matrix(img, max_rows=200, max_cols=400)
    finite = sampled[np.isfinite(sampled)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "table"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.04,
    )
    fig.add_trace(
        go.Heatmap(
            z=sampled,
            zmin=float(vmin),
            zmax=float(vmax),
            colorscale="Viridis",
            colorbar=dict(title="value"),
            hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_info_table(info), row=1, col=2)
    fig.update_layout(
        title=title or f"CIFTI QC — {info.path.name}<br><sup>shape {info.shape} · sampled {sampled.shape}</sup>",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#ddd",
        xaxis_title="sampled greyordinates / columns",
        yaxis_title="sampled rows / time",
        height=560,
    )
    return fig


def _sample_cifti_matrix(img: Any, *, max_rows: int, max_cols: int) -> np.ndarray:
    shape = tuple(int(s) for s in img.shape)
    if len(shape) == 1:
        row_idx = np.array([0])
        col_idx = np.linspace(0, shape[0] - 1, min(max_cols, shape[0])).astype(int)
        try:
            return np.asarray(img.dataobj[col_idx], dtype=np.float32)[np.newaxis, :]
        except Exception:
            data = np.asarray(img.dataobj, dtype=np.float32)
            return data[col_idx][np.newaxis, :]

    n_rows = min(max_rows, shape[0])
    n_cols = min(max_cols, int(np.prod(shape[1:])))
    row_idx = np.linspace(0, shape[0] - 1, n_rows).astype(int)

    try:
        row_sample = np.asarray(img.dataobj[row_idx, ...], dtype=np.float32).reshape(n_rows, -1)
    except Exception:
        data = np.asarray(img.dataobj, dtype=np.float32).reshape(shape[0], -1)
        row_sample = data[row_idx]
    col_idx = np.linspace(0, row_sample.shape[1] - 1, n_cols).astype(int)
    return row_sample[:, col_idx]


def _first_vertex_values(values: list[np.ndarray], n_vertices: int) -> np.ndarray | None:
    for arr in values:
        if arr.size == n_vertices:
            return arr.astype(float, copy=False)
    return None


def _surface_subtitle(info: SurfaceInfo) -> str:
    parts = []
    if info.hemisphere:
        parts.append(f"hemi-{info.hemisphere}")
    if info.n_vertices is not None:
        parts.append(f"{info.n_vertices:,} vertices")
    if info.n_faces is not None:
        parts.append(f"{info.n_faces:,} faces")
    roles = _role_counts(info.arrays)
    if roles:
        parts.append(", ".join(f"{k}:{v}" for k, v in roles.items()))
    return " · ".join(parts) if parts else "surface summary"


def _role_counts(arrays: list[SurfaceArrayInfo]) -> dict[str, int]:
    out: dict[str, int] = {}
    for arr in arrays:
        out[arr.role] = out.get(arr.role, 0) + 1
    return dict(sorted(out.items()))


def _info_table(info: SurfaceInfo) -> Any:
    import plotly.graph_objects as go

    rows: list[tuple[str, str]] = [
        ("family", info.family),
        ("shape", str(info.shape or "")),
        ("hemisphere", info.hemisphere or "unknown"),
    ]
    if info.n_vertices is not None:
        rows.append(("vertices", f"{info.n_vertices:,}"))
    if info.n_faces is not None:
        rows.append(("faces", f"{info.n_faces:,}"))
    if info.bounds:
        bounds = " ".join(f"{axis}[{lo:.1f},{hi:.1f}]" for axis, (lo, hi) in info.bounds.items())
        rows.append(("bounds", bounds))
    if info.labels:
        rows.append(("labels", f"{len(info.labels)} label(s)"))
    if info.arrays:
        rows.append(("array roles", ", ".join(f"{k}:{v}" for k, v in _role_counts(info.arrays).items())))
        for arr in info.arrays[:8]:
            rng = ""
            if arr.min_value is not None and arr.max_value is not None:
                rng = f" [{arr.min_value:.3g}, {arr.max_value:.3g}]"
            rows.append((f"array {arr.index}", f"{arr.role} {arr.shape} {arr.dtype}{rng}"))
    if info.axes:
        for axis in info.axes[:4]:
            details = f"{axis.get('type')} size={axis.get('size')}"
            if axis.get("n_structures"):
                details += f" structures={axis.get('n_structures')}"
            rows.append((f"axis {axis.get('index')}", details))

    return go.Table(
        header=dict(values=["field", "value"], fill_color="#222", font_color="#ddd", align="left"),
        cells=dict(
            values=[[key for key, _ in rows], [value for _, value in rows]],
            fill_color="#111",
            font_color="#bbb",
            align="left",
            height=24,
        ),
    )
