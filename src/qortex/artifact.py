"""User-facing handle for converted Qortex artifacts."""

from __future__ import annotations

import json
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
            import polars as pl
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

        # Read enough rows to reach the requested index (no full-shard load)
        row_cursor = 0
        target_row = None
        for shard in shards:
            df = pl.read_parquet(shard)
            n = len(df)
            if row_cursor + n > index:
                target_row = df.row(index - row_cursor, named=True)
                break
            row_cursor += n

        if target_row is None:
            raise IndexError(
                f"Sample index {index} out of range "
                f"(split '{split}' has {row_cursor} samples)"
            )

        # Detect and render the data column
        data_col = next(
            (k for k in target_row if k not in ("label", "subject", "session",
                                                  "task", "split", "source_path")
             and isinstance(target_row[k], (list, bytes))),
            None,
        )

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
            import polars as pl
            import plotly.io as pio
        except ImportError:
            raise ImportError("visual_audit() requires plotly and polars: pip install plotly polars")

        from qortex.visualize._audit import VisualAuditReport, AuditEntry
        from qortex.visualize._asset import VisualAsset

        split_dir = self.path / split if split != "all" else self.path
        shards = sorted(split_dir.glob("**/*.parquet")) if split_dir.exists() else sorted(self.path.glob("**/*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found for split={split!r} in {self.path}")

        # Collect samples across shards (no full load)
        samples = []
        for shard in shards:
            df = pl.read_parquet(shard)
            take = min(max(1, n // len(shards) + 1), len(df))
            samples.extend(df.head(take).to_dicts())
            if len(samples) >= n:
                break
        samples = samples[:n]

        entries: list[AuditEntry] = []
        n_rendered = 0
        n_failed = 0

        for i, row in enumerate(samples):
            label = str(row.get("label", "?"))
            sub = str(row.get("subject", ""))
            path_label = f"sample_{i}  label={label}" + (f"  sub={sub}" if sub else "")
            asset = VisualAsset(path=self.path / f"sample_{i}", family="array",
                                intent="artifact_sample", modality="signal")
            try:
                data_col = next(
                    (k for k in row if k not in ("label", "subject", "session", "task", "split", "source_path")
                     and isinstance(row[k], list)),
                    None,
                )
                thumb_b64 = None
                if data_col and row[data_col]:
                    arr = _pl_to_array(row[data_col])
                    if arr.ndim >= 2:
                        import plotly.graph_objects as go
                        fig = _plot_volume_slice(arr, f"Sample {i}") if arr.ndim >= 3 else _plot_signal_2d(arr, f"Sample {i}")
                        png_bytes = pio.to_image(fig, format="png", width=300, height=180)
                        import base64
                        thumb_b64 = base64.b64encode(png_bytes).decode()
                    elif arr.ndim == 1:
                        import plotly.graph_objects as go
                        fig = _plot_signal_1d(arr, f"Sample {i}")
                        png_bytes = pio.to_image(fig, format="png", width=300, height=180)
                        import base64
                        thumb_b64 = base64.b64encode(png_bytes).decode()
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
            import polars as pl
            import plotly.io as pio
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

            # Collect n samples across shards — minimal row reads
            rows_needed = n
            samples: list[dict] = []
            for shard in shards:
                if len(samples) >= rows_needed:
                    break
                df = pl.read_parquet(shard)
                take = min(max(1, rows_needed // max(1, len(shards)) + 1), len(df))
                samples.extend(df.head(take).to_dicts())
            samples = samples[:rows_needed]

            for i, row in enumerate(samples):
                label = str(row.get("label", "?"))
                sub = str(row.get("subject", ""))
                path_label = (
                    f"{split_name}/sample_{i}"
                    f"  label={label}"
                    + (f"  sub={sub}" if sub else "")
                )
                asset = VisualAsset(
                    path=self.path / split_name / f"sample_{i}",
                    family="array",
                    intent="artifact_sample",
                    modality="signal",
                )
                try:
                    data_col = next(
                        (k for k in row
                         if k not in ("label", "subject", "session", "task", "split", "source_path")
                         and isinstance(row[k], list)),
                        None,
                    )
                    thumb_b64 = None
                    if data_col and row[data_col]:
                        arr = _pl_to_array(row[data_col])
                        import plotly.graph_objects as go
                        if arr.ndim >= 3:
                            fig = _plot_volume_slice(arr, f"{split_name}/{i}")
                        elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                            fig = _plot_signal_2d(arr, f"{split_name}/{i}")
                        else:
                            fig = _plot_signal_1d(arr.ravel(), f"{split_name}/{i}")
                        png_bytes = pio.to_image(fig, format="png", width=280, height=160)
                        import base64
                        thumb_b64 = base64.b64encode(png_bytes).decode()
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

def _pl_to_array(value) -> "np.ndarray":
    import numpy as np
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
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
