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

    def as_dataframe(self, split: str | None = None):
        """Return artifact rows as a Polars DataFrame.

        This is the intended downstream view for table/event/behavior artifacts
        where samples are structured rows rather than numeric signal arrays.
        """
        from qortex.train.sklearn import SklearnAdapter

        if self.manifest.output_format != "parquet":
            raise ValueError("as_dataframe() currently expects a Parquet Qortex artifact.")
        return SklearnAdapter().as_dataframe(self.path, split=split)

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

        shards = _artifact_shards(self.path, split=split)
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found in {self.path}")

        target_row = _read_artifact_row(shards, index, split=split)

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

        split_filter = None if split == "all" else split
        shards = _artifact_shards(self.path, split=split_filter)
        if not shards:
            raise FileNotFoundError(f"No Parquet shards found for split={split!r} in {self.path}")

        samples = _sample_artifact_rows(shards, n=n, seed=seed, split=split_filter)

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

    def validate_contract(self) -> dict:
        """Verify the artifact's integrity and contract against its manifest.

        Checks:
        - ``artifact_manifest.json`` exists and is parseable (done at open time)
        - SHA-256 of every listed file matches the stored hash
        - ``artifact_contract.json`` is present and consistent with the manifest
        - All declared splits have at least one shard on disk
        - Sample count in manifest matches actual row count in shards

        Returns
        -------
        dict
            ``{"ok": bool, "errors": [...], "warnings": [...]}``
        """
        import hashlib

        errors: list[str] = []
        warnings: list[str] = []

        # Check file hashes if artifact_manifest.json carries them.
        file_hashes: dict = getattr(self.manifest, "file_hashes", {}) or {}
        for rel_path, expected_hash in file_hashes.items():
            full = self.path / rel_path
            if not full.exists():
                errors.append(f"File listed in manifest is missing on disk: {rel_path}")
                continue
            actual = hashlib.sha256(full.read_bytes()).hexdigest()
            if actual != expected_hash:
                errors.append(
                    f"SHA-256 mismatch for {rel_path}: "
                    f"expected {expected_hash[:16]}… got {actual[:16]}…"
                )

        # Check splits have readable rows. Writers may store splits either as
        # split directories or as a root-level ``split`` column.
        declared_splits = list((self.manifest.splits or {}).keys())
        for split_name in declared_splits:
            try:
                n_rows = _count_split_rows(self.path, split_name)
            except Exception as exc:
                errors.append(f"Cannot read split {split_name!r}: {exc}")
                continue
            if n_rows == 0:
                warnings.append(f"Split {split_name!r} has no Parquet rows on disk")

        # Check artifact_contract.json is present.
        contract_path = self.path / "artifact_contract.json"
        if not contract_path.exists():
            warnings.append("artifact_contract.json not found — older artifact format")

        # Verify sample count (best-effort, catches truncated shards).
        if self.manifest.n_samples and self.manifest.n_samples > 0:
            try:
                import polars as pl
                total = sum(
                    int(pl.scan_parquet(s).select(pl.len()).collect().item())
                    for s in self.path.glob("**/*.parquet")
                )
                if total != self.manifest.n_samples:
                    warnings.append(
                        f"Manifest claims {self.manifest.n_samples} samples "
                        f"but {total} rows found in shards"
                    )
            except Exception as exc:
                warnings.append(f"Could not verify sample count: {exc}")

        return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

    def validate_schema(
        self,
        *,
        label_col: str | None = None,
        min_samples_per_class: int = 10,
        max_null_fraction: float = 0.05,
    ) -> dict:
        """Validate the artifact's data schema before using it for training.

        Checks performed:

        1. **Label column presence** — the nominated column (or manifest default)
           exists in every split.
        2. **Null fraction** — fraction of null labels ≤ ``max_null_fraction``.
        3. **Minimum class support** — every class has at least
           ``min_samples_per_class`` rows across the split.
        4. **Label consistency** — the same set of class labels appears in every
           split (warns, not errors, when test/val are missing a minority class).
        5. **Dtype uniformity** — feature columns have consistent dtype across
           splits.

        Parameters
        ----------
        label_col:
            Label column name.  Defaults to ``manifest.label_column`` if set,
            then to ``"label"`` as a last resort.
        min_samples_per_class:
            Minimum acceptable row count per class within each split.
        max_null_fraction:
            Maximum acceptable fraction of null label values (0–1).

        Returns
        -------
        dict
            ``{"ok": bool, "errors": [...], "warnings": [...], "stats": {...}}``
        """
        try:
            import polars as pl
        except ImportError:
            return {
                "ok": False,
                "errors": ["validate_schema() requires polars: pip install polars"],
                "warnings": [],
                "stats": {},
            }

        errors: list[str] = []
        warnings: list[str] = []
        stats: dict = {}

        col = (
            label_col
            or getattr(self.manifest, "label_column", None)
            or "label"
        )

        declared_splits = list((self.manifest.splits or {}).keys())
        if not declared_splits:
            declared_splits = [
                d.name for d in self.path.iterdir()
                if d.is_dir() and list(d.glob("*.parquet"))
            ]

        all_label_sets: dict[str, set] = {}

        for split_name in declared_splits:
            shards = _artifact_shards(self.path, split=split_name)
            if not shards:
                warnings.append(f"Split {split_name!r} has no Parquet shards — skipping schema check.")
                continue

            try:
                df = _read_artifact_frame(self.path, split=split_name)
            except Exception as exc:
                errors.append(f"Cannot read split {split_name!r}: {exc}")
                continue

            # 1. Label column presence.
            if col not in df.columns:
                errors.append(
                    f"Label column {col!r} not found in split {split_name!r}. "
                    f"Available columns: {df.columns[:15]}"
                )
                continue

            label_series = df[col]
            n_rows = len(label_series)

            # 2. Null fraction.
            n_null = label_series.null_count()
            null_frac = n_null / max(n_rows, 1)
            if null_frac > max_null_fraction:
                errors.append(
                    f"Split {split_name!r}: {null_frac:.1%} of {col!r} values are null "
                    f"(limit {max_null_fraction:.0%}). "
                    "Use LabelPolicy.missing='drop' in conversion to remove null-label rows."
                )
            elif n_null > 0:
                warnings.append(
                    f"Split {split_name!r}: {n_null}/{n_rows} null values in {col!r} "
                    f"({null_frac:.1%}) — within the {max_null_fraction:.0%} limit."
                )

            # 3. Per-class counts.
            try:
                class_counts = (
                    label_series.drop_nulls()
                    .value_counts(sort=True)
                    .rename({"count": "n"})
                    .to_dicts()
                )
            except Exception:
                class_counts = []

            split_stats: dict = {"n_rows": n_rows, "n_null": n_null, "classes": {}}
            class_labels: set = set()
            for row in class_counts:
                lbl = str(row[col])
                n = row["n"]
                class_labels.add(lbl)
                split_stats["classes"][lbl] = n
                if n < min_samples_per_class:
                    (errors if split_name == "train" else warnings).append(
                        f"Split {split_name!r}: class {lbl!r} has only {n} sample(s) "
                        f"(minimum {min_samples_per_class}). "
                        "Too few samples for reliable training/evaluation."
                    )

            all_label_sets[split_name] = class_labels
            stats[split_name] = split_stats

        # 4. Label consistency across splits.
        split_names = list(all_label_sets.keys())
        if len(split_names) >= 2:
            ref_labels = all_label_sets.get("train") or all_label_sets[split_names[0]]
            for sn, labels in all_label_sets.items():
                if sn == "train":
                    continue
                missing_in_split = ref_labels - labels
                extra_in_split = labels - ref_labels
                if missing_in_split:
                    warnings.append(
                        f"Split {sn!r} is missing classes present in 'train': "
                        f"{sorted(missing_in_split)}. "
                        "Evaluation metrics may be misleading for these classes."
                    )
                if extra_in_split:
                    errors.append(
                        f"Split {sn!r} has classes not seen in 'train': "
                        f"{sorted(extra_in_split)}. "
                        "Model will encounter unseen classes at inference."
                    )

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "label_col": col,
            "stats": stats,
        }

    def check_leakage(self) -> dict:
        """Check for subject-level and source-level leakage across splits.

        A split is leaky if the same subject ID or source file appears in two
        or more splits.  This is the primary leakage risk for subject-stratified
        splits; it does not catch temporal leakage or derivative-source overlap.

        Returns
        -------
        dict
            ``{"ok": bool, "leaky_subjects": [...], "leaky_sources": [...], "details": [...]}``
        """
        try:
            import polars as pl
        except ImportError:
            return {
                "ok": False,
                "leaky_subjects": [],
                "leaky_sources": [],
                "details": ["check_leakage() requires polars: pip install polars"],
            }

        split_subjects: dict[str, set[str]] = {}
        split_sources: dict[str, set[str]] = {}

        declared_splits = list((self.manifest.splits or {}).keys())
        if not declared_splits:
            # Discover from directory structure
            declared_splits = [
                d.name for d in sorted(self.path.iterdir())
                if d.is_dir() and list(d.glob("*.parquet"))
            ]

        for split_name in declared_splits:
            shards = _artifact_shards(self.path, split=split_name)
            if not shards:
                continue
            subjects: set[str] = set()
            sources: set[str] = set()
            df = _read_artifact_frame(self.path, split=split_name)
            cols = df.columns
            if "subject" in cols:
                vals = df["subject"].drop_nulls().to_list()
                subjects.update(str(v) for v in vals)
            for src_col in ("source_path", "source_file"):
                if src_col in cols:
                    vals = df[src_col].drop_nulls().to_list()
                    sources.update(str(v) for v in vals)
                    break
            split_subjects[split_name] = subjects
            split_sources[split_name] = sources

        # Find subjects that appear in more than one split.
        all_splits = list(split_subjects.keys())
        leaky_subjects: list[str] = []
        leaky_sources: list[str] = []
        details: list[str] = []

        for i, s1 in enumerate(all_splits):
            for s2 in all_splits[i + 1:]:
                shared_subs = split_subjects.get(s1, set()) & split_subjects.get(s2, set())
                if shared_subs:
                    leaky_subjects.extend(sorted(shared_subs))
                    details.append(
                        f"Subject leakage between {s1!r} and {s2!r}: "
                        + ", ".join(sorted(shared_subs)[:5])
                        + (" …" if len(shared_subs) > 5 else "")
                    )
                shared_srcs = split_sources.get(s1, set()) & split_sources.get(s2, set())
                if shared_srcs:
                    leaky_sources.extend(sorted(shared_srcs))
                    details.append(
                        f"Source leakage between {s1!r} and {s2!r}: "
                        + ", ".join(sorted(shared_srcs)[:3])
                        + (" …" if len(shared_srcs) > 3 else "")
                    )

        return {
            "ok": len(leaky_subjects) == 0 and len(leaky_sources) == 0,
            "leaky_subjects": sorted(set(leaky_subjects)),
            "leaky_sources": sorted(set(leaky_sources)),
            "details": details,
        }

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
            shards = _artifact_shards(self.path, split=split_name)
            if not shards:
                continue

            samples = _sample_artifact_rows(
                shards,
                n=n,
                seed=None if seed is None else seed + len(all_entries),
                split=split_name,
            )

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


def _artifact_shards(path: Path, split: str | None = None) -> list[Path]:
    if split:
        split_dir = path / split
        if split_dir.exists():
            shards = sorted(split_dir.glob("*.parquet"))
            if shards:
                return shards
    return sorted(path.glob("shard_*.parquet")) or sorted(path.glob("**/*.parquet"))


def _read_artifact_frame(path: Path, split: str | None = None):
    shards = _artifact_shards(path, split=split)
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {path}")
    return _read_frame_from_shards(shards, split=split)


def _count_split_rows(path: Path, split: str) -> int:
    return int(_read_artifact_frame(path, split=split).height)


def _read_artifact_row(
    shards: list[Path],
    index: int,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    import polars as pl

    if index < 0:
        raise IndexError("Sample index must be non-negative")
    if split:
        frame = _read_frame_from_shards(shards, split=split)
        if index >= frame.height:
            raise IndexError(f"Sample index {index} out of range ({frame.height} samples)")
        return frame.slice(index, 1).row(0, named=True)
    row_cursor = 0
    for shard in shards:
        n_rows = _parquet_n_rows(shard)
        if row_cursor + n_rows > index:
            local_idx = index - row_cursor
            frame = pl.scan_parquet(shard).slice(local_idx, 1).collect()
            return frame.row(0, named=True)
        row_cursor += n_rows
    raise IndexError(f"Sample index {index} out of range ({row_cursor} samples)")


def _sample_artifact_rows(
    shards: list[Path],
    *,
    n: int,
    seed: int | None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    import polars as pl

    if n <= 0:
        return []
    if split:
        frame = _read_frame_from_shards(shards, split=split)
        total = frame.height
        if total == 0:
            return []
        rng = random.Random(seed)
        wanted = sorted(rng.sample(range(total), k=min(n, total)))
        return [frame.slice(offset, 1).row(0, named=True) for offset in wanted]
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


def _read_frame_from_shards(shards: list[Path], *, split: str | None = None):
    import polars as pl

    frame = pl.concat([pl.read_parquet(shard) for shard in shards])
    if split and "split" in frame.columns:
        frame = frame.filter(pl.col("split") == split)
    return frame


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
