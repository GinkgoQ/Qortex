"""User-facing handle for converted Qortex artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from qortex.core.entities import ArtifactManifest


class Artifact:
    """Open and use a converted Qortex artifact."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        manifest_path = self.path / "artifact_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No artifact_manifest.json found in {self.path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = ArtifactManifest(**data)

    @classmethod
    def open(cls, path: Path | str) -> "Artifact":
        return cls(path)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.manifest.artifact_id,
            "dataset_id": self.manifest.dataset_id,
            "snapshot": self.manifest.snapshot,
            "format": self.manifest.output_format,
            "n_samples": self.manifest.n_samples,
            "n_subjects": self.manifest.n_subjects,
            "splits": self.manifest.splits,
        }

    def torch(self, split: str | None = "train", *, iterable: bool = False):
        from qortex.train.torch import QortexIterableTorchDataset, QortexTorchDataset

        if self.manifest.output_format != "parquet":
            raise ValueError("Torch adapter currently expects a Parquet Qortex artifact.")
        if iterable:
            return QortexIterableTorchDataset(self.path, split=split)
        return QortexTorchDataset(self.path, split=split)

    def sklearn(self, split: str | None = None):
        from qortex.train.sklearn import SklearnAdapter

        if self.manifest.output_format != "parquet":
            raise ValueError("sklearn adapter currently expects a Parquet Qortex artifact.")
        return SklearnAdapter().from_dir(self.path, split=split)

    def visualize_sample(
        self,
        index: int = 0,
        *,
        split: str | None = "train",
        mode: str = "static",
    ):
        """Render one converted sample as a visual figure.

        Reads a single sample from the Parquet artifact without loading the
        whole split into memory.  Returns a plotly Figure showing:

        * Signal / time-series data → butterfly plot (EEG/BOLD)
        * Image data (2D/3D array) → center-slice orthogonal view

        Parameters
        ----------
        index:
            Zero-based sample index within the split.
        split:
            Which split to read from (``"train"``, ``"val"``, ``"test"``).
            ``None`` reads from the first available shard.
        mode:
            ``"static"`` (default) returns a plotly Figure; ``"html"`` returns
            an HTML string.

        Returns
        -------
        plotly.graph_objects.Figure  (mode="static")
        str                          (mode="html")
        """
        try:
            import plotly.graph_objects as go
            import polars  # noqa: F401
        except ImportError:
            raise ImportError("visualize_sample() requires plotly and polars: pip install plotly polars")

        # Find the first shard for the requested split
        split_dir = self.path / split if split else self.path
        shards = sorted(split_dir.glob("*.parquet")) if split_dir.exists() else []
        if not shards:
            # Fall back: scan the artifact root
            shards = sorted(self.path.glob("**/*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found in {self.path}")

        target_row = _read_artifact_row(shards, index)

        # Detect and render the data column
        data_col = _artifact_data_column(target_row, self.manifest)

        meta_rows = "".join(
            f'<tr><td style="color:#888;padding:2px 12px 2px 0">{k}</td>'
            f'<td style="color:#ccc">{v}</td></tr>'
            for k, v in target_row.items()
            if k != data_col and not isinstance(v, (list, bytes))
        )
        meta_html = f'<table style="font-size:0.82em;border-collapse:collapse;margin-bottom:12px">{meta_rows}</table>' if meta_rows else ""

        title = (
            f"Sample {index}"
            + (f" · label={target_row.get('label', '?')}" if "label" in target_row else "")
            + (f" · sub-{target_row.get('subject', '?')}" if "subject" in target_row else "")
        )

        if data_col is None or target_row[data_col] is None:
            fig = go.Figure()
            fig.add_annotation(text="No signal/image column found in sample",
                               showarrow=False, font=dict(size=14, color="#888"))
        elif isinstance(target_row[data_col], list):
            arr = _pl_to_array(target_row[data_col])
            if arr.ndim == 1:
                fig = _plot_signal_1d(arr, title)
            elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                fig = _plot_signal_2d(arr, title)
            elif arr.ndim >= 3:
                fig = _plot_volume_slice(arr, title)
            else:
                fig = _plot_signal_1d(arr.ravel(), title)
        else:
            fig = go.Figure()
            fig.add_annotation(text="Unsupported data format in sample",
                               showarrow=False, font=dict(size=14, color="#888"))

        fig.update_layout(paper_bgcolor="#111", plot_bgcolor="#111", font_color="#ccc",
                          height=380)

        if mode == "html":
            import plotly.io as pio
            body = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
            return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{background:#111;color:#ccc;font-family:sans-serif;margin:20px}}</style>
</head><body><h2 style="color:#6af">{title}</h2>{meta_html}{body}</body></html>"""
        return fig

    def visual_audit(
        self,
        *,
        split: str = "train",
        n: int = 16,
        seed: int | None = 0,
    ):
        """Render thumbnails for *n* random samples from the artifact.

        Returns a :class:`~qortex.visualize._audit.VisualAuditReport` whose
        ``.show()`` opens an HTML gallery.

        Parameters
        ----------
        split:
            Which split to sample from.  ``"all"`` scans all shards.
        n:
            Number of samples to render.
        """
        try:
            import plotly.io as pio
            import polars  # noqa: F401
        except ImportError:
            raise ImportError("visual_audit() requires plotly and polars: pip install plotly polars")

        from qortex.visualize._audit import VisualAuditReport, AuditEntry
        from qortex.visualize._asset import VisualAsset

        split_dir = self.path / split if split != "all" else self.path
        shards = sorted(split_dir.glob("**/*.parquet")) if split_dir.exists() else sorted(self.path.glob("**/*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found for split={split!r} in {self.path}")

        samples = _sample_artifact_rows(shards, n=n, seed=seed)

        entries: list[AuditEntry] = []
        n_rendered = 0
        n_failed = 0

        for i, row in enumerate(samples):
            path_label = _artifact_path_label(row, split=split, row_index=i)
            asset = VisualAsset(path=self.path / f"sample_{i}", family="array",
                                intent="artifact_sample", modality="signal")
            try:
                data_col = _artifact_data_column(row, self.manifest)
                thumb_b64 = None
                if data_col and row[data_col]:
                    arr = _pl_to_array(row[data_col])
                    asset.shape = tuple(int(s) for s in arr.shape)
                    asset.dtype = str(arr.dtype)
                    asset.metadata.update(_artifact_row_metadata(row, data_col=data_col, shape=arr.shape))
                    if arr.ndim >= 2:
                        fig = _plot_volume_slice(arr, f"Sample {i}") if arr.ndim >= 3 else _plot_signal_2d(arr, f"Sample {i}")
                        thumb_b64 = _figure_to_png_b64(fig, pio, width=300, height=180)
                    elif arr.ndim == 1:
                        fig = _plot_signal_1d(arr, f"Sample {i}")
                        thumb_b64 = _figure_to_png_b64(fig, pio, width=300, height=180)
                entries.append(AuditEntry(path_label=path_label, asset=asset, thumbnail_b64=thumb_b64))
                n_rendered += 1
            except Exception as exc:
                entries.append(AuditEntry(path_label=path_label, asset=asset, error=str(exc)))
                n_failed += 1

        return VisualAuditReport(
            dataset_id=f"{self.manifest.artifact_id} [{split}]",
            n_files_inspected=len(entries),
            n_rendered=n_rendered,
            n_failed=n_failed,
            entries=entries,
        )

    def compare_splits(
        self,
        *,
        n: int = 8,
        splits: list[str] | None = None,
        seed: int | None = 0,
    ):
        """Render *n* random samples from each split side-by-side for comparison.

        The returned :class:`~qortex.visualize._audit.VisualAuditReport` has one
        entry per sample, with ``path_label`` encoded as
        ``"{split}/sample_{i} label={lbl} sub={sub}"`` so the gallery is
        self-explanatory without requiring the user to understand shard layout.

        Uses the same lazy Parquet reading strategy as ``visual_audit()``: only
        the minimum rows are scanned per shard.

        Parameters
        ----------
        n:
            Samples to render per split.
        splits:
            Which splits to compare.  Defaults to all available splits from the
            manifest (``["train", "val", "test"]`` filtered by what exists on disk).
        """
        try:
            import plotly.io as pio
            import polars  # noqa: F401
        except ImportError:
            raise ImportError("compare_splits() requires plotly and polars: pip install plotly polars")

        from qortex.visualize._audit import VisualAuditReport, AuditEntry
        from qortex.visualize._asset import VisualAsset

        # Determine splits to compare
        available_splits = splits or list(self.manifest.splits.keys() if self.manifest.splits else [])
        if not available_splits:
            # Discover by directory scanning
            available_splits = [
                d.name for d in sorted(self.path.iterdir())
                if d.is_dir() and list(d.glob("*.parquet"))
            ]
        if not available_splits:
            raise FileNotFoundError(f"No split directories with Parquet shards found in {self.path}")

        all_entries: list[AuditEntry] = []
        n_rendered = 0
        n_failed = 0

        for split_name in available_splits:
            split_dir = self.path / split_name
            shards = sorted(split_dir.glob("*.parquet")) if split_dir.exists() else []
            if not shards:
                continue

            samples = _sample_artifact_rows(shards, n=n, seed=None if seed is None else seed + len(all_entries))

            for i, row in enumerate(samples):
                path_label = _artifact_path_label(row, split=split_name, row_index=i)
                asset = VisualAsset(
                    path=self.path / split_name / f"sample_{i}",
                    family="array",
                    intent="artifact_sample",
                    modality="signal",
                )
                try:
                    data_col = _artifact_data_column(row, self.manifest)
                    thumb_b64 = None
                    if data_col and row[data_col]:
                        arr = _pl_to_array(row[data_col])
                        asset.shape = tuple(int(s) for s in arr.shape)
                        asset.dtype = str(arr.dtype)
                        asset.metadata.update(_artifact_row_metadata(row, data_col=data_col, shape=arr.shape))
                        if arr.ndim >= 3:
                            fig = _plot_volume_slice(arr, f"{split_name}/{i}")
                        elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                            fig = _plot_signal_2d(arr, f"{split_name}/{i}")
                        else:
                            fig = _plot_signal_1d(arr.ravel(), f"{split_name}/{i}")
                        thumb_b64 = _figure_to_png_b64(fig, pio, width=280, height=160)
                    all_entries.append(AuditEntry(
                        path_label=path_label, asset=asset, thumbnail_b64=thumb_b64,
                    ))
                    n_rendered += 1
                except Exception as exc:
                    all_entries.append(AuditEntry(
                        path_label=path_label, asset=asset, error=str(exc),
                    ))
                    n_failed += 1

        splits_label = "/".join(available_splits)
        return VisualAuditReport(
            dataset_id=f"{self.manifest.artifact_id} [{splits_label}]",
            n_files_inspected=len(all_entries),
            n_rendered=n_rendered,
            n_failed=n_failed,
            entries=all_entries,
        )


# ── Helpers for visualize_sample / visual_audit ───────────────────────────────

_META_COLUMNS = {
    "label", "subject", "session", "task", "split", "source_path",
    "source_file", "dataset_id", "snapshot", "trial_type", "onset", "duration",
    "run", "row_index", "sample_index",
}


def _read_artifact_row(shards: list[Path], index: int) -> dict[str, Any]:
    import polars as pl

    if index < 0:
        raise IndexError("Sample index must be non-negative")
    row_cursor = 0
    for shard in shards:
        n_rows = _parquet_n_rows(shard)
        if row_cursor + n_rows > index:
            local_idx = index - row_cursor
            frame = pl.scan_parquet(shard).slice(local_idx, 1).collect()
            return frame.row(0, named=True)
        row_cursor += n_rows
    raise IndexError(f"Sample index {index} out of range ({row_cursor} samples)")


def _sample_artifact_rows(shards: list[Path], *, n: int, seed: int | None) -> list[dict[str, Any]]:
    import polars as pl

    if n <= 0:
        return []
    counts = [(shard, _parquet_n_rows(shard)) for shard in shards]
    total = sum(count for _, count in counts)
    if total == 0:
        return []
    rng = random.Random(seed)
    wanted = sorted(rng.sample(range(total), k=min(n, total)))
    out: list[dict[str, Any]] = []
    cursor = 0
    wanted_i = 0
    for shard, count in counts:
        if wanted_i >= len(wanted):
            break
        local_offsets = []
        while wanted_i < len(wanted) and cursor <= wanted[wanted_i] < cursor + count:
            local_offsets.append(wanted[wanted_i] - cursor)
            wanted_i += 1
        cursor += count
        for offset in local_offsets:
            frame = pl.scan_parquet(shard).slice(offset, 1).collect()
            out.append(frame.row(0, named=True))
    return out


def _parquet_n_rows(shard: Path) -> int:
    import polars as pl

    return int(pl.scan_parquet(shard).select(pl.len()).collect().item())


def _artifact_data_column(row: dict[str, Any], manifest: ArtifactManifest) -> str | None:
    schema = manifest.data_schema or {}
    candidates: list[str] = []
    for key in ("data_column", "signal_column", "image_column", "array_column", "feature_column"):
        value = schema.get(key)
        if isinstance(value, str):
            candidates.append(value)
    columns = schema.get("columns")
    if isinstance(columns, dict):
        for name, spec in columns.items():
            if isinstance(spec, dict) and spec.get("role") in {"data", "signal", "image", "features"}:
                candidates.append(str(name))
    for col in candidates:
        if col in row and _is_array_like_value(row[col]):
            return col
    return next(
        (
            k for k, value in row.items()
            if k not in _META_COLUMNS and _is_array_like_value(value)
        ),
        None,
    )


def _is_array_like_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, bytes, bytearray, memoryview, np.ndarray)):
        return True
    return hasattr(value, "to_numpy") or hasattr(value, "to_list")


def _artifact_path_label(row: dict[str, Any], *, split: str | None, row_index: int) -> str:
    parts = [f"{split or 'all'}/row_{row_index}"]
    for key, prefix in (
        ("label", "label"),
        ("subject", "sub"),
        ("session", "ses"),
        ("task", "task"),
        ("run", "run"),
    ):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{prefix}={value}")
    source = row.get("source_path") or row.get("source_file")
    if source:
        parts.append(f"src={Path(str(source)).name}")
    return "  ".join(parts)


def _artifact_row_metadata(
    row: dict[str, Any],
    *,
    data_col: str | None,
    shape: tuple[int, ...],
) -> dict[str, Any]:
    keys = (
        "split", "label", "subject", "session", "task", "run",
        "source_path", "source_file", "trial_type", "onset", "duration",
    )
    meta = {key: row[key] for key in keys if key in row and row[key] not in (None, "")}
    meta["data_column"] = data_col or ""
    meta["shape"] = "x".join(str(s) for s in shape)
    return meta


def _figure_to_png_b64(fig: Any, pio: Any, *, width: int, height: int, timeout_s: int = 5) -> str | None:
    import base64
    import os
    import signal

    if os.environ.get("QORTEX_ENABLE_STATIC_THUMBNAILS") != "1":
        return None

    def _timeout(_signum, _frame):
        raise TimeoutError("Plotly static export timed out")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(timeout_s)
        png_bytes = pio.to_image(fig, format="png", width=width, height=height)
        return base64.b64encode(png_bytes).decode()
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _pl_to_array(value) -> "np.ndarray":
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    if isinstance(value, tuple):
        return np.array(value, dtype=np.float32)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.float32)
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float32)
    if hasattr(value, "to_list"):
        return np.asarray(value.to_list(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _plot_signal_1d(arr: "np.ndarray", title: str) -> Any:
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        y=arr.tolist(), mode="lines",
        line=dict(color="#6af", width=1.2), showlegend=False,
    ))
    fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="Amplitude",
                      height=300)
    return fig


def _plot_signal_2d(arr: "np.ndarray", title: str) -> Any:
    """Butterfly plot for (channels × samples) 2D array."""
    import plotly.graph_objects as go
    import math
    n_ch, n_s = arr.shape
    mu = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-10
    normed = (arr - mu) / std
    alpha = max(0.08, min(0.5, 1.0 / math.sqrt(n_ch)))
    fig = go.Figure()
    for ch in normed:
        fig.add_trace(go.Scatter(y=ch.tolist(), mode="lines",
                                  line=dict(color=f"rgba(100,180,255,{alpha:.2f})", width=0.8),
                                  showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="Amplitude (z-scored)",
                      height=300)
    return fig


def _plot_volume_slice(arr: "np.ndarray", title: str) -> Any:
    """Center axial slice of a 3D+ array."""
    import plotly.graph_objects as go
    import numpy as np
    vol = arr
    while vol.ndim > 3:
        vol = vol[..., vol.shape[-1] // 2]
    cz = vol.shape[-1] // 2 if vol.ndim == 3 else 0
    slc = vol[:, :, cz].T[::-1, :] if vol.ndim == 3 else vol
    flat = slc.ravel()
    flat = flat[np.isfinite(flat)]
    vmin = float(np.percentile(flat, 1.0)) if flat.size else 0.0
    vmax = float(np.percentile(flat, 99.0)) if flat.size else 1.0
    normed = np.clip((slc - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    fig = go.Figure(go.Heatmap(z=normed.tolist(), colorscale="Gray",
                                zmin=0, zmax=1, showscale=False))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_layout(title=title, height=280)
    return fig
