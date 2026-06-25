"""Qortex by GinkgoQ — ML-ready neurodata from OpenNeuro.

Quick start::

    from qortex import Dataset

    ds = Dataset("ds004130")
    ds.download(subjects=["01", "02"], modalities=["eeg"])
    report = ds.eda()
    result = ds.convert(output_format="parquet", window_duration=2.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex._version import __version__
from qortex.artifact import Artifact
from qortex.core.config import configure, get_config
from qortex.core.entities import (
    ConversionResult,
    DownloadResult,
    EDAReport,
    EventLabelSummary,
    FilePreview,
    FileRecord,
    LocalIndexReport,
    Manifest,
    ReadinessReport,
    SelectionSpec,
    ValidationDiff,
    ValidationReport,
)
from qortex.core.exceptions import QortexError


class Dataset:
    """High-level facade for working with a single OpenNeuro dataset.

    Parameters
    ----------
    dataset_id:
        OpenNeuro dataset ID, e.g. ``"ds004130"``.
    snapshot:
        Snapshot tag to pin to.  ``None`` uses the latest published snapshot.
    token:
        Optional API token.  Falls back to env var ``QORTEX_API_TOKEN`` or
        ``~/.config/qortex/credentials.json``.
    data_dir:
        Override the download destination.  Defaults to
        ``~/.cache/qortex/datasets/{dataset_id}/{snapshot}/``.
    """

    def __init__(
        self,
        dataset_id: str,
        snapshot: str | None = None,
        token: str | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.snapshot = snapshot
        self._token = token
        self._data_dir = data_dir
        self._manifest: Manifest | None = None
        self._snap_ref = None

    # ── Manifest / introspection ──────────────────────────────────────────

    def manifest(self, *, force_refresh: bool = False) -> Manifest:
        """Fetch and cache the file manifest for this dataset."""
        if self._manifest is not None and not force_refresh:
            return self._manifest

        from qortex.client.graphql import OpenNeuroClient
        from qortex.manifest.builder import ManifestBuilder

        client = OpenNeuroClient(token=self._token)
        builder = ManifestBuilder()

        if self.snapshot:
            snap_ref = client.get_snapshot(self.dataset_id, self.snapshot)
        else:
            snap_ref = client.get_latest_snapshot(self.dataset_id)

        self._snap_ref = snap_ref
        self.snapshot = snap_ref.tag
        snap_ref, raw_files = client.get_files(self.dataset_id, snap_ref.tag)
        self._snap_ref = snap_ref
        self._manifest = builder.build(self.dataset_id, snap_ref, raw_files)
        return self._manifest

    def info(self) -> dict[str, Any]:
        """Return a summary dict of dataset metadata."""
        m = self.manifest()
        s = m.summary
        return {
            "dataset_id": self.dataset_id,
            "snapshot": self.snapshot,
            "doi": m.doi,
            "n_files": s.file_count,
            "n_subjects": s.n_subjects,
            "n_sessions": len(s.sessions),
            "n_tasks": len(s.tasks),
            "total_size_gb": round(s.total_size / 1e9, 3),
            "modalities": s.modalities,
            "has_events": s.has_events,
            "has_derivatives": s.has_derivatives,
        }

    def files(
        self,
        *,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        datatypes: list[str] | None = None,
        extensions: list[str] | None = None,
        metadata_only: bool = False,
    ) -> list[FileRecord]:
        """Return manifest files matching structural filters."""
        manifest = self.manifest()
        files = manifest.filter(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            modalities=modalities,
            datatypes=datatypes,
            extensions=extensions,
        )
        if metadata_only:
            files = [
                file for file in files
                if file.is_essential
                or file.extension in {".json", ".tsv", ".csv", ".bvec", ".bval"}
            ]
        return files

    def metadata_files(self) -> list[FileRecord]:
        """Return essential metadata and lightweight sidecar/table files."""
        return self.files(metadata_only=True)

    # ── Download ──────────────────────────────────────────────────────────

    def download(
        self,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        include_derivatives: bool = False,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        metadata_only: bool = False,
        event_complete: bool = False,
        label_ready: bool = False,
        loadable_only: bool = False,
        max_size_gb: float | None = None,
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> DownloadResult:
        """Download the dataset (or a filtered subset) to local disk.

        Parameters
        ----------
        subjects / sessions / tasks / modalities:
            Filter which files to download.  ``None`` means include all.
        exclude_derivatives:
            Skip the ``derivatives/`` subdirectory.
        include / exclude:
            Gitignore-style glob patterns applied on top of entity filters.
        output_dir:
            Override the download destination.
        dry_run:
            Plan the download but don't actually transfer any files.

        Returns
        -------
        DownloadResult
            Summary of what was downloaded, skipped, or failed.
        """
        from qortex.fetch.engine import DownloadEngine
        from qortex.plan.planner import DownloadPlanner

        manifest = self.manifest()

        spec = SelectionSpec(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            modalities=modalities,
            include_derivatives=include_derivatives,
            include=include or [],
            exclude=exclude or [],
            metadata_only=metadata_only,
            event_complete=event_complete,
            label_ready=label_ready,
            loadable_only=loadable_only,
            max_size_gb=max_size_gb,
        )

        target = output_dir or self._resolve_data_dir()
        self._data_dir = target

        planner = DownloadPlanner()
        plan = planner.plan(manifest, spec, target)

        if dry_run:
            from qortex.core.entities import DownloadResult
            return DownloadResult(plan=plan)

        engine = DownloadEngine()
        return engine.execute(plan)

    def download_metadata(
        self,
        output_dir: Path | None = None,
        *,
        dry_run: bool = False,
        max_size_gb: float | None = None,
    ) -> DownloadResult:
        """Download only essential metadata, sidecars, and small tables."""
        return self.download(
            metadata_only=True,
            output_dir=output_dir,
            dry_run=dry_run,
            max_size_gb=max_size_gb,
        )

    def download_paths(
        self,
        paths: list[str],
        output_dir: Path | None = None,
        *,
        with_companions: bool = True,
        dry_run: bool = False,
        max_size_gb: float | None = None,
    ) -> DownloadResult:
        """Download exact manifest paths, optionally including companions."""
        from qortex.fetch.engine import DownloadEngine
        from qortex.plan.planner import DownloadPlanner

        manifest = self.manifest()
        spec = SelectionSpec(
            include=paths,
            with_companions=with_companions,
            max_size_gb=max_size_gb,
        )
        target = output_dir or self._resolve_data_dir()
        self._data_dir = target
        plan = DownloadPlanner().plan(manifest, spec, target)
        if dry_run:
            from qortex.core.entities import DownloadResult

            return DownloadResult(plan=plan)
        return DownloadEngine().execute(plan)

    def preview(
        self,
        path: str,
        *,
        local_path: Path | None = None,
        n_rows: int = 5,
        max_bytes: int = 64_000,
    ) -> FilePreview:
        """Preview a local or remote file without downloading the full file."""
        from qortex.preview import preview_file

        return preview_file(
            self.manifest(),
            path,
            local_path=local_path or self._data_dir,
            n_rows=n_rows,
            max_bytes=max_bytes,
        )

    def first_rows(
        self,
        path: str,
        *,
        n: int = 5,
        local_path: Path | None = None,
        max_bytes: int = 64_000,
    ) -> list[dict[str, Any]]:
        """Return first rows of a TSV/CSV file from local disk or remote URL."""
        return self.preview(
            path,
            local_path=local_path,
            n_rows=n,
            max_bytes=max_bytes,
        ).rows

    def preview_metadata(
        self,
        *,
        local_path: Path | None = None,
        n_rows: int = 5,
        max_files: int | None = None,
    ) -> list[FilePreview]:
        """Preview essential metadata and sidecar/table files."""
        from qortex.preview import preview_metadata

        return preview_metadata(
            self.manifest(),
            local_path=local_path or self._data_dir,
            n_rows=n_rows,
            max_files=max_files,
        )

    def check(
        self,
        local_path: Path | None = None,
        conversion_target: str | None = None,
        inspect_loaders: bool = False,
    ) -> ReadinessReport:
        """Return a decision-oriented readiness report for this dataset."""
        from qortex.check import compute_readiness

        return compute_readiness(
            self.manifest(),
            local_path=local_path or self._data_dir,
            conversion_target=conversion_target,
            inspect_loaders=inspect_loaders,
        )

    def validate(
        self,
        local_path: Path | None = None,
        *,
        config_path: Path | None = None,
        output_json: Path | None = None,
        ignore_warnings: bool = False,
        ignore_nifti_headers: bool = False,
        timeout_s: float = 600.0,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> ValidationReport:
        """Run the official BIDS Validator on a local dataset path."""
        from qortex.validation import validate_bids

        path = local_path or self._data_dir
        if path is None:
            raise RuntimeError(
                "No local dataset path is available. Pass local_path or call download() first."
            )
        return validate_bids(
            path,
            config_path=config_path,
            output_json=output_json,
            ignore_warnings=ignore_warnings,
            ignore_nifti_headers=ignore_nifti_headers,
            timeout_s=timeout_s,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )

    def index_local(
        self,
        local_path: Path | None = None,
        *,
        include_dirs: bool = False,
        use_pybids: bool = True,
    ) -> LocalIndexReport:
        """Index a local BIDS tree and reconcile it with the remote manifest."""
        from qortex.indexing import index_local_bids

        path = local_path or self._data_dir
        if path is None:
            raise RuntimeError(
                "No local dataset path is available. Pass local_path or call download() first."
            )
        return index_local_bids(
            path,
            manifest=self.manifest(),
            include_dirs=include_dirs,
            use_pybids=use_pybids,
        )

    # ── EDA ───────────────────────────────────────────────────────────────

    def eda(
        self,
        local_path: Path | None = None,
        output_html: Path | None = None,
    ) -> EDAReport:
        """Run exploratory data analysis on the manifest.

        Parameters
        ----------
        local_path:
            If provided, also reads local JSON sidecars for extra checks
            (e.g. sampling-frequency consistency).
        output_html:
            If provided, writes the HTML report to this path.
        """
        from qortex.eda.report import EDAEngine

        manifest = self.manifest()
        engine = EDAEngine(manifest)
        report = engine.run(local_path=local_path or self._data_dir)

        if output_html and report.html:
            report.to_html(output_html)

        return report

    # ── Convert ───────────────────────────────────────────────────────────

    def convert(
        self,
        output_dir: Path | None = None,
        output_format: str = "parquet",
        window_duration: float | None = None,
        window_overlap: float = 0.0,
        split_strategy: str = "subject",
        shard_size: int = 1000,
    ) -> ConversionResult:
        """Convert a downloaded dataset into an ML-ready artifact.

        The dataset must have been downloaded first (call ``download()``).

        Parameters
        ----------
        output_dir:
            Destination for the converted artifact.
        output_format:
            One of ``parquet``, ``zarr``, ``hdf5``, ``webdataset``,
            ``huggingface``, ``tfrecord``.
        window_duration:
            Window duration in seconds.  ``None`` = no windowing (one sample
            per file).
        window_overlap:
            Fraction of window overlap for fixed sliding windows (0–1).
        split_strategy:
            One of ``subject``, ``random``, ``stratified``.
        shard_size:
            Number of samples per output shard.
        """
        from qortex.convert.pipeline import ConversionPipeline
        from qortex.convert.splits import SplitSpec
        from qortex.convert.windows import WindowSpec

        if self._data_dir is None or not self._data_dir.exists():
            raise RuntimeError(
                "Dataset has not been downloaded yet. Call download() first."
            )

        manifest = self.manifest()

        default_out = self._resolve_data_dir().parent / "converted" / output_format
        dest = output_dir or default_out

        win_spec = (
            WindowSpec(duration_s=window_duration, overlap=window_overlap)
            if window_duration else None
        )

        pipeline = ConversionPipeline(
            manifest=manifest,
            data_dir=self._data_dir,
            output_dir=dest,
            output_format=output_format,
            window_spec=win_spec,
            split_spec=SplitSpec(strategy=split_strategy),
            shard_size=shard_size,
        )
        return pipeline.run()

    # ── ML adapters ───────────────────────────────────────────────────────

    def torch_dataset(
        self,
        output_dir: Path | None = None,
        split: str | None = "train",
        iterable: bool = False,
    ):
        """Return a torch Dataset/IterableDataset from a converted artifact."""
        from qortex.train.torch import QortexIterableTorchDataset, QortexTorchDataset

        artifact_dir = output_dir or self._resolve_data_dir().parent / "converted" / "parquet"
        if iterable:
            return QortexIterableTorchDataset(artifact_dir, split=split)
        return QortexTorchDataset(artifact_dir, split=split)

    def lightning_datamodule(
        self,
        output_dir: Path | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """Return a PyTorch Lightning DataModule."""
        from qortex.train.lightning import QortexDataModule

        artifact_dir = output_dir or self._resolve_data_dir().parent / "converted" / "parquet"
        return QortexDataModule(
            artifact_dir, batch_size=batch_size, num_workers=num_workers
        )

    def sklearn_arrays(
        self,
        output_dir: Path | None = None,
        split: str | None = None,
    ):
        """Return (X, y) numpy arrays for scikit-learn."""
        from qortex.train.sklearn import SklearnAdapter

        artifact_dir = output_dir or self._resolve_data_dir().parent / "converted" / "parquet"
        return SklearnAdapter().from_dir(artifact_dir, split=split)

    # ── Plan / lock ───────────────────────────────────────────────────────

    def plan(
        self,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
        tasks: list[str] | None = None,
        modalities: list[str] | None = None,
        datatypes: list[str] | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        include_derivatives: bool = False,
        metadata_only: bool = False,
        event_complete: bool = False,
        label_ready: bool = False,
        loadable_only: bool = False,
        max_size_gb: float | None = None,
        output_dir: Path | None = None,
    ):
        """Return a DownloadPlan without downloading."""
        from qortex.plan.planner import DownloadPlanner

        manifest = self.manifest()
        spec = SelectionSpec(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            modalities=modalities,
            datatypes=datatypes,
            include=include or [],
            exclude=exclude or [],
            include_derivatives=include_derivatives,
            metadata_only=metadata_only,
            event_complete=event_complete,
            label_ready=label_ready,
            loadable_only=loadable_only,
            max_size_gb=max_size_gb,
        )
        return DownloadPlanner().plan(manifest, spec, output_dir or self._resolve_data_dir())

    def select(self, **kwargs):
        """Alias for plan(); returns an explainable DownloadPlan."""
        return self.plan(**kwargs)

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Dataset(dataset_id={self.dataset_id!r}, "
            f"snapshot={self.snapshot!r})"
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _resolve_data_dir(self) -> Path:
        if self._data_dir:
            return self._data_dir
        cfg = get_config()
        snap = self.snapshot or "latest"
        return cfg.cache_dir / "datasets" / self.dataset_id / snap / "data"


# ── Module-level convenience exports ─────────────────────────────────────────

def search(*args, **kwargs):
    """Search the local catalog, importing catalog dependencies only when used."""
    from qortex.catalog.search import search as _search

    return _search(*args, **kwargs)

__all__ = [
    "__version__",
    "Dataset",
    "Artifact",
    "configure",
    "get_config",
    "QortexError",
    "search",
    "LocalIndexReport",
    "EventLabelSummary",
    "FilePreview",
    "ValidationDiff",
    "ValidationReport",
]
