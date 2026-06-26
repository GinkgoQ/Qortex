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
from qortex.catalog.search import DatasetQuery, PagedResults, facets, live_search, search
from qortex.core.exceptions import QortexError
from qortex.decision import (
    CanTrainReport,
    ContentStatusReport,
    DecisionFinding,
    DoctorReport,
    FirstBatchReport,
    LeakageReport,
    MinimumPlanReport,
    Recipe,
    content_status,
    leakage_check,
    read_recipe,
    write_recipe,
)
from qortex.inspect.dataset import DatasetInspector, DatasetProfile
from qortex.inspect.selector import DatasetFitness, DatasetSelector, ResearchGoal
from qortex import visualize


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

    def inspect(self, *, level: str = "manifest") -> "DatasetProfile":
        """Fetch metadata and return a DatasetProfile — no download needed.

        Parameters
        ----------
        level:
            Inspection depth:

            ``"summary"``
                API-only, completes in <2 s. Returns subjects, sessions,
                tasks, modalities, size, demographics, and engagement from
                the OpenNeuro GraphQL API without fetching the file tree.
                ``profile.manifest`` will be ``None`` at this level.
            ``"manifest"`` *(default)*
                Fetches the full recursive file tree, builds a typed
                ``Manifest``, and computes modality breakdown, companion
                coverage, ML readiness score, and recommendations.
            ``"deep"``
                Manifest level plus concurrent remote events and sidecar
                analysis. Adds ``LabelLandscape`` and ``SignalBudget`` data
                to the profile. Takes 10–60 s depending on dataset size.

        Returns
        -------
        DatasetProfile
            Inspection report. Call ``.summary()`` for a compact view,
            ``.report()`` for the full modality breakdown.

        Examples
        --------
        >>> ds = Dataset("ds000117")
        >>> # Fast API-only check
        >>> quick = ds.inspect(level="summary")
        >>> print(quick.n_subjects, quick.tasks)
        >>> # Full manifest analysis
        >>> profile = ds.inspect()
        >>> print(profile.ml_readiness.grade)
        >>> # With remote events + signal budget
        >>> deep = ds.inspect(level="deep")
        """
        inspector = DatasetInspector(token=self._token)
        return inspector.inspect(self.dataset_id, tag=self.snapshot, level=level)

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
        include_derivatives:
            Include the ``derivatives/`` subdirectory (default False).
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
        window_tmin: float = 0.0,
        event_aligned: bool = False,
        split_strategy: str = "subject",
        stratify_by_label: bool = True,
        shard_size: int = 1000,
        skip_missing: bool = True,
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
            Only used when event_aligned=False.
        window_tmin:
            Seconds before event onset for event-aligned windows (default 0.0).
            A negative value captures a pre-stimulus baseline.
        event_aligned:
            When True, use event_aligned_windows() instead of fixed_windows().
            Requires an events TSV to be present for each signal file.
        split_strategy:
            One of ``subject``, ``random``, ``stratified``.
        stratify_by_label:
            When split_strategy="subject", also stratify on majority label per
            subject to preserve class balance.
        shard_size:
            Number of samples per output shard.
        skip_missing:
            If True (default), files that fail to load are skipped and counted
            in ConversionResult.warnings rather than aborting the pipeline.
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
            WindowSpec(
                duration_s=window_duration,
                overlap=window_overlap if not event_aligned else 0.0,
                tmin=window_tmin,
                event_aligned=event_aligned,
            )
            if window_duration else None
        )

        pipeline = ConversionPipeline(
            manifest=manifest,
            data_dir=self._data_dir,
            output_dir=dest,
            output_format=output_format,
            window_spec=win_spec,
            split_spec=SplitSpec(
                strategy=split_strategy,
                stratify_by_label=stratify_by_label,
            ),
            shard_size=shard_size,
            skip_missing=skip_missing,
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

    # ── Decision workflows ─────────────────────────────────────────────────

    def doctor(self, local_path: Path | None = None) -> DoctorReport:
        """Return a decision report for download, conversion, and training readiness."""
        from qortex.decision import doctor

        return doctor(self.manifest(), local_path=local_path or self._data_dir)

    def minimum(
        self,
        goal: str = "first-batch",
        *,
        modality: str | None = None,
        target: str | None = None,
        output_dir: Path | None = None,
    ) -> MinimumPlanReport:
        """Plan the smallest real download needed for a concrete workflow goal."""
        from qortex.decision import minimum_plan

        return minimum_plan(
            self.manifest(),
            goal=goal,  # type: ignore[arg-type]
            modality=modality,
            target=target,
            output_dir=output_dir,
        )

    def can_train(
        self,
        *,
        modality: str | None = None,
        target: str | None = None,
        local_path: Path | None = None,
    ) -> CanTrainReport:
        """Assess whether the dataset can support supervised training."""
        from qortex.decision import can_train

        return can_train(
            self.manifest(),
            modality=modality,
            target=target,
            local_path=local_path or self._data_dir,
        )

    def first_batch(
        self,
        *,
        artifact_path: Path | None = None,
        local_path: Path | None = None,
        modality: str | None = None,
        target: str | None = None,
        limit: int = 8,
    ) -> FirstBatchReport:
        """Preview the first rows from an artifact or create a first-batch plan."""
        from qortex.decision import first_batch

        manifest = None if artifact_path is not None else self.manifest()
        return first_batch(
            artifact_path=artifact_path,
            manifest=manifest,
            local_path=local_path or self._data_dir,
            modality=modality,
            target=target,
            limit=limit,
        )

    def content_status(self, local_path: Path | None = None) -> ContentStatusReport:
        """Check local content completeness and pointer-like files against the manifest."""
        from qortex.decision import content_status

        path = local_path or self._data_dir
        if path is None:
            raise RuntimeError(
                "No local dataset path is available. Pass local_path or call download() first."
            )
        return content_status(path, manifest=self.manifest())

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

    # ── Remote inspection (no download needed) ────────────────────────────

    def participants(self, *, prefer_api: bool = True):
        """Return participant demographics as a Polars DataFrame.

        Tries the OpenNeuro API first (``snapshot.summary.subjectMetadata``)
        which gives age, sex, and group without downloading participants.tsv.
        Falls back to fetching participants.tsv via CDN if the API returns no
        demographics.

        Parameters
        ----------
        prefer_api:
            When True (default), try the API before the remote TSV. Set False
            to always read directly from the CDN file.

        Returns
        -------
        polars.DataFrame
            Columns: participant_id, age (int | null), sex (str | null),
            group (str | null).  Additional columns from the TSV are included
            when the fallback path is taken.

        Examples
        --------
        >>> df = Dataset("ds000117").participants()
        >>> df.filter(pl.col("sex") == "M")["age"].mean()
        """
        import polars as pl
        from qortex.client.graphql import OpenNeuroClient

        if prefer_api:
            with OpenNeuroClient(token=self._token) as client:
                try:
                    snap_tag = self.snapshot or client.get_latest_snapshot(self.dataset_id).tag
                    summary = client.get_snapshot_summary(self.dataset_id, snap_tag)
                    if summary.subject_demographics:
                        return summary.demographics_dataframe()
                except Exception:
                    pass

        # Fallback: fetch participants.tsv via CDN
        manifest = self.manifest()
        from qortex.client.remote import RemoteFileGateway, best_url_for_path
        gateway = RemoteFileGateway()
        url = best_url_for_path(manifest, "participants.tsv")
        if url is None:
            return pl.DataFrame(schema={"participant_id": pl.Utf8, "age": pl.Int64, "sex": pl.Utf8, "group": pl.Utf8})
        return gateway.fetch_tsv(url)

    def events(
        self,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
        run: str | None = None,
    ):
        """Fetch a remote events TSV as a Polars DataFrame without downloading.

        All parameters are optional; when None, the first matching events file
        in the manifest is used.

        Parameters
        ----------
        subject / session / task / run:
            BIDS entity filters.  Partial matches are accepted — e.g. just
            ``task="rest"`` will find any events file for the rest task.

        Returns
        -------
        polars.DataFrame
            Columns from the events TSV: onset, duration, trial_type, …

        Examples
        --------
        >>> df = Dataset("ds000117").events(subject="01", task="facerecognition")
        >>> df.head()
        """
        manifest = self.manifest()
        matches = [
            fr for fr in manifest.files
            if fr.suffix == "events"
            and (subject is None or fr.subject == subject)
            and (session is None or fr.session == session)
            and (task is None or fr.task == task)
            and (run is None or fr.run == run)
        ]
        if not matches:
            raise FileNotFoundError(
                f"No events file found in manifest for "
                f"sub={subject!r} ses={session!r} task={task!r} run={run!r}"
            )
        fr = matches[0]
        from qortex.client.remote import RemoteFileGateway, _pick_url
        gateway = RemoteFileGateway()
        url = _pick_url(fr)
        if not url:
            raise FileNotFoundError(f"No URL for events file {fr.path!r}")
        return gateway.fetch_tsv(url)

    def sidecar(self, path: str):
        """Fetch and merge BIDS JSON sidecars for a file path, without downloading.

        Follows BIDS inheritance: most-general (dataset root) sidecar values
        are overridden by more-specific ones (subject → session → file-level).

        Parameters
        ----------
        path:
            BIDS-relative path, e.g. ``"sub-01/eeg/sub-01_task-rest_eeg.set"``.

        Returns
        -------
        dict
            Merged JSON sidecar key-value pairs.

        Examples
        --------
        >>> meta = Dataset("ds004130").sidecar("sub-01/eeg/sub-01_task-rest_eeg.set")
        >>> meta["SamplingFrequency"]
        256
        """
        manifest = self.manifest()
        from qortex.client.remote import RemoteFileGateway, _pick_url
        from qortex.manifest.sidecar import SidecarResolver

        # Find the target file in the manifest
        target = next((fr for fr in manifest.files if fr.path == path), None)
        if target is None:
            raise FileNotFoundError(f"Path {path!r} not found in manifest")

        resolver = SidecarResolver(manifest.files)
        sidecar_records = resolver.resolve(target)

        gateway = RemoteFileGateway()
        merged: dict = {}
        for fr in sidecar_records:
            url = _pick_url(fr)
            if url:
                try:
                    data = gateway.fetch_json(url)
                    merged.update(data)
                except Exception:
                    pass
        return merged

    def nifti_info(self, path: str):
        """Extract NIfTI header info remotely using an HTTP Range request.

        Fetches only 352 bytes (NIfTI-1) or up to 64 KB (NIfTI-2 / gzip)
        to determine image shape, voxel sizes, TR, and number of volumes.

        Parameters
        ----------
        path:
            BIDS-relative path to a .nii or .nii.gz file.

        Returns
        -------
        NIfTIHeader
            Shape, voxel sizes in mm, TR in seconds, number of volumes.

        Examples
        --------
        >>> info = Dataset("ds000117").nifti_info(
        ...     "sub-01/func/sub-01_task-facerecognition_bold.nii.gz"
        ... )
        >>> print(info)
        4D fMRI 64×64×33×208 vox=3.00×3.00×4.05mm TR=2.000s
        """
        manifest = self.manifest()
        from qortex.client.remote import RemoteFileGateway, _pick_url

        target = next((fr for fr in manifest.files if fr.path == path), None)
        if target is None:
            raise FileNotFoundError(f"Path {path!r} not found in manifest")

        url = _pick_url(target)
        if not url:
            raise FileNotFoundError(f"No URL available for {path!r}")

        gateway = RemoteFileGateway()
        return gateway.fetch_nifti_header(url)

    def label_landscape(
        self,
        *,
        label_column: str | None = None,
        concurrency: int = 24,
        max_events_files: int | None = None,
    ):
        """Analyse all events TSVs remotely and return a LabelLandscape.

        Concurrently fetches all events.tsv files from CDN URLs, auto-detects
        the label column, and computes trial-type statistics, class balance,
        inter-stimulus interval jitter, cross-subject consistency, and
        actionable ML recommendations — with zero bytes downloaded to disk.

        Parameters
        ----------
        label_column:
            Column to use as the class label (e.g. ``"trial_type"``).  When
            None, auto-detected from the first events file.
        concurrency:
            Number of parallel CDN requests.
        max_events_files:
            Cap for number of events files to fetch (useful for large datasets).

        Returns
        -------
        LabelLandscape
            Rich label analysis.  Call ``.summary()`` for a compact report.

        Examples
        --------
        >>> landscape = Dataset("ds000117").label_landscape()
        >>> print(landscape.summary())
        >>> landscape.imbalance_severity   # "balanced" | "moderate" | ...
        """
        from qortex.client.remote import RemoteFileGateway
        from qortex.inspect.label_landscape import LabelLandscapeAnalyzer

        manifest = self.manifest()
        gateway = RemoteFileGateway()
        analyzer = LabelLandscapeAnalyzer(gateway)
        return analyzer.analyze(
            manifest,
            label_column=label_column,
            concurrency=concurrency,
            max_events_files=max_events_files,
        )

    def signal_budget(
        self,
        *,
        concurrency: int = 24,
        include_nifti_headers: bool = True,
    ):
        """Estimate total signal hours and achievable windows without downloading.

        Fetches JSON sidecars remotely for all signal files, extracting
        SamplingFrequency, RecordingDuration, EEGChannelCount, etc.  For fMRI
        files with missing TR or volume counts, fetches the NIfTI header
        (352 bytes via HTTP Range) to fill the gaps.

        Parameters
        ----------
        concurrency:
            Parallel CDN connections.
        include_nifti_headers:
            Whether to fetch NIfTI headers for fMRI files missing TR info.

        Returns
        -------
        SignalBudget
            Per-modality recording hours and window estimates.  Call
            ``budget.estimate_windows(2.0)`` for a 2-second window breakdown.

        Examples
        --------
        >>> budget = Dataset("ds000117").signal_budget()
        >>> budget.estimate_windows(window_duration_s=2.0, overlap=0.5)
        {'meg': 183200, 'eeg': 42100}
        >>> budget.minimum_download_for_n_windows(10000, window_s=2.0)
        {'subjects_needed': 4, 'windows_achieved': 11200}
        """
        from qortex.client.remote import RemoteFileGateway
        from qortex.inspect.signal_budget import SignalBudgetEstimator

        manifest = self.manifest()
        gateway = RemoteFileGateway()
        estimator = SignalBudgetEstimator(gateway)
        return estimator.estimate(
            manifest,
            concurrency=concurrency,
            include_nifti_headers=include_nifti_headers,
        )

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


def refresh_catalog(*args, **kwargs):
    """Refresh the local OpenNeuro catalog index."""
    from qortex.catalog.refresh import refresh

    return refresh(*args, **kwargs)


def refresh_catalog_dataset(*args, **kwargs):
    """Refresh and return one dataset profile in the local catalog index."""
    from qortex.catalog.refresh import refresh_dataset

    return refresh_dataset(*args, **kwargs)

__all__ = [
    "__version__",
    "Dataset",
    "Artifact",
    "configure",
    "get_config",
    "QortexError",
    "search",
    "refresh_catalog",
    "refresh_catalog_dataset",
    # Visualize
    "visualize",
    # Inspect
    "DatasetInspector",
    "DatasetProfile",
    "DatasetSelector",
    "DatasetFitness",
    "ResearchGoal",
    # Catalog
    "DatasetQuery",
    "PagedResults",
    "live_search",
    "facets",
    # Core entities
    "LocalIndexReport",
    "EventLabelSummary",
    "FilePreview",
    "ValidationDiff",
    "ValidationReport",
    # Decision
    "DecisionFinding",
    "DoctorReport",
    "MinimumPlanReport",
    "CanTrainReport",
    "FirstBatchReport",
    "LeakageReport",
    "ContentStatusReport",
    "Recipe",
    "content_status",
    "leakage_check",
    "read_recipe",
    "write_recipe",
]
