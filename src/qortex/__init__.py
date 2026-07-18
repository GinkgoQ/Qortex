"""Qortex by GinkgoQ — ML-ready neurodata from OpenNeuro.

Quick start::

    from qortex import Dataset

    ds = Dataset("ds004130")
    ds.download(subjects=["01", "02"], modalities=["eeg"])
    report = ds.eda()
    result = ds.convert(output_format="parquet", window_duration=2.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

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
    LabelPolicy,
    Manifest,
    ReadinessReport,
    SelectionSpec,
    ValidationDiff,
    ValidationReport,
)
from qortex.catalog.search import DatasetQuery, PagedResults, facets, live_search
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
from qortex.visualize._audit import VisualAuditReport

# ── Advanced ML / neuroimaging subsystems ────────────────────────────────────
from qortex.harmonize import HarmonizationReporter, HarmonizationReport
from qortex.export import MONAIExporter, TorchIOExporter
from qortex.derivatives import DerivativeIndexer
from qortex.qc import QCFilter, QCMask
from qortex.cohort import CohortBuilder, CohortManifest, FederatedCohort, FederatedSubject
from qortex.stream import NiftiStreamer, EDFStreamer
from qortex.runtime import (
    BIDSImageDataset,
    BIDSSignalDataset,
    BIDSEpochDataset,
    MONAIDictBuilder,
    TorchEEGBridge,
)


# ── BIDS path helpers (module-level so they work inside Dataset methods) ──────

def _extract_subject(path: str) -> str | None:
    """Extract subject ID (without sub- prefix) from a BIDS relative path."""
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("sub-"):
            return part[4:]
    return None


def _bids_suffix(path: str) -> str:
    """Extract the BIDS suffix from a file path by inspecting the filename stem."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    # Strip compound extensions (.nii.gz etc.)
    for ext in (".nii.gz", ".nii", ".mgz", ".mgh", ".edf", ".fif", ".bdf", ".set",
                ".gz", ".json", ".tsv"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    else:
        name = name.rsplit(".", 1)[0]
    # BIDS suffix = last _word in the stem
    seg = name.rsplit("_", 1)
    return seg[-1] if len(seg) > 1 else name


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
        manifest: "Manifest | None" = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.snapshot = snapshot
        self._token = token
        self._data_dir = data_dir
        # Optional pre-fetched manifest: every Dataset method that needs one
        # calls self.manifest(), which only checks its own *instance*
        # cache — useless when a caller (e.g. the Atlas API, which builds a
        # fresh Dataset per HTTP request) already holds a manifest from its
        # own cross-request cache. Seeding it here means previewing a single
        # 200-byte file doesn't silently re-fetch the whole dataset's file
        # tree first.
        self._manifest: Manifest | None = manifest
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
        suffixes: list[str] | None = None,
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
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        """Download the dataset (or a filtered subset) to local disk.

        Parameters
        ----------
        subjects / sessions / tasks / modalities:
            Filter which files to download.  ``None`` means include all.
        suffixes:
            BIDS suffixes to download, e.g. ``["T1w", "bold", "dwi"]``.
            Translated to ``include`` glob patterns and combined with any
            explicit ``include`` patterns you supply.
        include_derivatives:
            Include the ``derivatives/`` subdirectory (default False).
        include / exclude:
            Gitignore-style glob patterns applied on top of entity filters.
        output_dir:
            Override the download destination.
        dry_run:
            Plan the download but don't actually transfer any files.
        on_progress:
            Optional callback invoked as ``on_progress(files_done, files_total)``
            each time one file finishes (success or failure) — lets a caller
            (e.g. a background job runner) report real, live progress instead
            of only knowing "started" vs. "done".

        Returns
        -------
        DownloadResult
            Summary of what was downloaded, skipped, or failed.

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01", "02"], suffixes=["T1w"])
        >>> ds.download(subjects=["01"], include=["**/*_T1w.*", "**/*_bold.*"])
        """
        from qortex.fetch.engine import DownloadEngine
        from qortex.plan.planner import DownloadPlanner

        manifest = self.manifest()

        # Convert BIDS suffix shorthand to glob patterns
        effective_include = list(include or [])
        if suffixes:
            for sfx in suffixes:
                effective_include.append(f"**/*_{sfx}.*")
                effective_include.append(f"**/*_{sfx}_*")

        spec = SelectionSpec(
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            modalities=modalities,
            include_derivatives=include_derivatives,
            include=effective_include,
            exclude=exclude or [],
            metadata_only=metadata_only,
            event_complete=event_complete,
            label_ready=label_ready,
            loadable_only=loadable_only,
            max_size_gb=max_size_gb,
        )

        target = Path(output_dir) if output_dir is not None else self._resolve_data_dir()
        self._data_dir = target

        planner = DownloadPlanner()
        plan = planner.plan(manifest, spec, target)

        if dry_run:
            from qortex.core.entities import DownloadResult
            return DownloadResult(plan=plan)

        engine = DownloadEngine()
        return engine.execute(plan, on_progress=on_progress)

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
        """Download exact manifest paths, optionally including companions.

        Paths are matched by set membership — they are *not* interpreted as
        glob patterns.  This avoids misbehaviour when paths contain glob
        metacharacters (e.g. ``[`` or ``]`` in dataset-level filenames).
        Pass ``include=[...]`` via ``download()`` if you need pattern matching.
        """
        from qortex.fetch.engine import DownloadEngine
        from qortex.plan.planner import DownloadPlanner

        manifest = self.manifest()
        spec = SelectionSpec(
            exact_paths=paths,       # exact set membership, not glob
            with_companions=with_companions,
            max_size_gb=max_size_gb,
        )
        target = Path(output_dir) if output_dir is not None else self._resolve_data_dir()
        self._data_dir = target
        plan = DownloadPlanner().plan(manifest, spec, target)
        if dry_run:
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

    # ── AI Runtime — PyTorch / MONAI / TorchEEG native adapters ─────────────

    def with_format(
        self,
        type: str = "torch",
        *,
        device: str = "cpu",
        dtype: str | None = None,
    ):
        """Return a ``BIDSImageDataset`` pre-configured for the given framework.

        Parameters
        ----------
        type:
            ``"torch"`` (default): returns a ``BIDSImageDataset`` compatible
            with ``torch.utils.data.DataLoader``.
            ``"numpy"``: same Dataset but items returned as numpy arrays.
        device:
            Target device string for tensor placement, e.g. ``"cuda"`` or
            ``"cpu"``.  Only meaningful when ``type="torch"``.

        Returns
        -------
        BIDSImageDataset
            A PyTorch Dataset covering the locally downloaded data.

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01"], suffixes=["T1w"])
        >>> torch_ds = ds.with_format("torch", device="cuda")
        >>> loader = torch_ds.to_dataloader(batch_size=4, num_workers=2)
        """
        from qortex.runtime import BIDSImageDataset

        data_dir = self._resolve_data_dir()

        def _device_transform(sample: dict) -> dict:
            if type == "torch":
                try:
                    import torch
                    for k, v in sample.items():
                        if hasattr(v, "numpy") or (hasattr(v, "shape") and hasattr(v, "dtype")):
                            import numpy as np
                            arr = np.asarray(v)
                            sample[k] = torch.from_numpy(arr).to(device)
                except ImportError:
                    pass
            return sample

        return BIDSImageDataset(
            bids_root=data_dir,
            transform=_device_transform if type == "torch" and device != "cpu" else None,
        )

    def to_monai_dicts(
        self,
        image_entities: list[str] | None = None,
        *,
        label_target: str | None = None,
        mask_entity: str | None = None,
        datatype: str = "anat",
        extension: str = ".nii.gz",
        include_metadata: bool = False,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> dict:
        """Build MONAI-style datalist dicts for CacheDataset / PersistentDataset.

        Parameters
        ----------
        image_entities:
            BIDS suffixes to use as image inputs, e.g. ``["T1w", "T2w"]``.
            When multiple suffixes share the same MONAI key (default: all map
            to ``"image"``), they are returned as a list for multi-channel use.
        label_target:
            Column in participants.tsv to use as the label, e.g.
            ``"diagnosis"``.
        mask_entity:
            BIDS suffix for the segmentation mask, e.g. ``"seg"``.  Mapped
            to the MONAI ``"label"`` key.
        include_metadata:
            When True, include sidecar JSON fields in each dict.

        Returns
        -------
        dict with keys ``"training"``, ``"validation"``, ``"test"``,
        ``"label_classes"``  (int → canonical label mapping).

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01", "02"], suffixes=["T1w"])
        >>> dicts = ds.to_monai_dicts(image_entities=["T1w"], label_target="diagnosis")
        >>> from monai.data import CacheDataset
        >>> train_set = CacheDataset(data=dicts["training"], transform=transform)
        """
        from qortex.runtime.loader import MONAIDictBuilder

        data_dir = self._resolve_data_dir()

        image_keys: dict[str, str] = {}
        if image_entities:
            if len(image_entities) == 1:
                image_keys = {image_entities[0]: "image"}
            else:
                for i, ent in enumerate(image_entities):
                    image_keys[ent] = "image" if i == 0 else f"image{i+1}"
        else:
            image_keys = {"T1w": "image"}

        seg_suffix: str | None = mask_entity

        builder = MONAIDictBuilder(
            bids_root=data_dir,
            image_keys=image_keys,
            datatype=datatype,
            extension=extension,
        )
        return builder.build(
            label_column=label_target,
            seg_suffix=seg_suffix,
            seg_key="label",
            include_metadata=include_metadata,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )

    def to_torch_dataloader(
        self,
        *,
        suffix: str = "T1w",
        datatype: str = "anat",
        extension: str = ".nii.gz",
        label_column: str | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle: bool = True,
        transform=None,
    ):
        """Return a PyTorch DataLoader from locally downloaded BIDS image data.

        Parameters
        ----------
        suffix:
            BIDS suffix for the image files (e.g. ``"T1w"``, ``"bold"``).
        label_column:
            Column in participants.tsv for labels. ``None`` = no labels.
        batch_size / num_workers / shuffle:
            Standard DataLoader arguments.
        transform:
            Optional callable applied to each sample dict before batching.

        Returns
        -------
        torch.utils.data.DataLoader

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01", "02", "03"], suffixes=["T1w"])
        >>> loader = ds.to_torch_dataloader(
        ...     suffix="T1w", label_column="diagnosis",
        ...     batch_size=4, num_workers=2, shuffle=True,
        ... )
        >>> for batch in loader:
        ...     images = batch["image"]  # (B, 1, X, Y, Z)
        """
        from qortex.runtime import BIDSImageDataset

        data_dir = self._resolve_data_dir()
        ds = BIDSImageDataset(
            bids_root=data_dir,
            suffix=suffix,
            datatype=datatype,
            extension=extension,
            label_column=label_column,
            transform=transform,
        )
        return ds.to_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def to_torcheeg_epochs(
        self,
        *,
        modality: str = "eeg",
        event_id: str | list[str] | None = None,
        tmin: float = -0.2,
        tmax: float = 0.5,
        resample_hz: float | None = None,
        stack_to_grid: bool = False,
        grid_size: tuple[int, int] = (9, 9),
    ):
        """Return a TorchEEG-compatible epoch Dataset from locally downloaded EEG.

        Parameters
        ----------
        event_id:
            Trial type(s) to include.  ``None`` = all.
        tmin / tmax:
            Epoch window in seconds relative to event onset.  Pre-stimulus
            baseline is captured when ``tmin < 0``.
        resample_hz:
            Resample all recordings before epoch extraction.
        stack_to_grid:
            Reshape ``(n_ch, n_t)`` → ``(rows, cols, n_t)`` 2D electrode grid
            for spatial CNN-based models.

        Returns
        -------
        TorchEEGBridge
            Implements ``__len__``/``__getitem__`` returning
            ``{"eeg": FloatTensor, "label": int}`` — TorchEEG's expected format.

        Examples
        --------
        >>> ds = Dataset("ds004130")
        >>> ds.download(modalities=["eeg"])
        >>> epoch_ds = ds.to_torcheeg_epochs(event_id="stimulus", tmin=-0.2, tmax=0.5)
        >>> loader = torch.utils.data.DataLoader(epoch_ds, batch_size=64)
        """
        from qortex.runtime import BIDSEpochDataset, TorchEEGBridge

        data_dir = self._resolve_data_dir()
        epoch_duration = tmax - tmin

        bids_epochs = BIDSEpochDataset(
            bids_root=data_dir,
            modality=modality,
            epoch_duration_s=epoch_duration,
            tmin=tmin,
            event_id=event_id,
            resample_hz=resample_hz,
        )
        return TorchEEGBridge(
            bids_epochs,
            stack_to_grid=stack_to_grid,
            grid_size=grid_size,
        )

    def map_labels(
        self,
        *,
        source_tsv: str = "participants",
        target_column: str,
        transform=None,
        label_map: dict | None = None,
    ) -> dict[str, Any]:
        """Read and transform a participants.tsv column into a subject→label mapping.

        Useful for injecting external labels (age buckets, diagnosis scores,
        composite phenotypes) into downstream Datasets without re-downloading.

        Parameters
        ----------
        source_tsv:
            Which TSV file to read: ``"participants"`` (default) maps to
            ``participants.tsv`` at the dataset root; any BIDS-relative path
            is accepted.
        target_column:
            Column to extract, e.g. ``"age"``, ``"diagnosis"``.
        transform:
            Optional callable applied to each raw value before returning.
            Example: ``transform=lambda x: float(x) / 100.0`` to normalize age.
        label_map:
            Optional raw → canonical value mapping applied before ``transform``.

        Returns
        -------
        dict
            ``{subject_id: label}`` mapping where subject_id is the BIDS
            ``sub-XX`` string.

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> age_map = ds.map_labels(
        ...     target_column="age",
        ...     transform=lambda x: (float(x) - 18) / 50,  # normalise 18-68 → 0-1
        ... )
        >>> age_map["sub-01"]
        0.42
        """
        import csv

        manifest = self.manifest()
        tsv_path = (
            "participants.tsv"
            if source_tsv == "participants"
            else source_tsv
        )
        fr = next((f for f in manifest.files if f.path == tsv_path), None)
        if fr is None or not fr.urls:
            raise FileNotFoundError(
                f"TSV file {tsv_path!r} not found in manifest."
            )

        from qortex.client.remote import RemoteFileGateway
        with RemoteFileGateway() as gw:
            text = gw.fetch_text(fr.urls[0])

        lines = text.strip().splitlines()
        reader = csv.DictReader(lines, delimiter="\t")
        result: dict[str, Any] = {}
        for row in reader:
            pid = row.get("participant_id", "").strip()
            if not pid:
                continue
            if not pid.startswith("sub-"):
                pid = f"sub-{pid}"
            raw = row.get(target_column)
            if raw is None:
                continue
            if label_map:
                raw = label_map.get(str(raw), raw)
            try:
                val = transform(raw) if transform is not None else raw
            except Exception:
                val = raw
            result[pid] = val
        return result

    def train_test_split(
        self,
        *,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify_by: str | None = None,
        group_by: str = "subject",
        seed: int = 42,
    ) -> dict[str, list[str]]:
        """Split subject IDs into train / val / test ensuring no subject leakage.

        Subject-level splitting: all files for a subject go to one split.
        When ``stratify_by`` is given, the split is class-balanced.

        Parameters
        ----------
        test_size:
            Fraction of subjects for the test split (0–1).
        val_size:
            Fraction of subjects for the validation split (0–1).
        stratify_by:
            Column in participants.tsv to stratify on (e.g. ``"diagnosis"``).
            When None, subjects are split without stratification.
        group_by:
            Grouping level.  ``"subject"`` (default) splits at the subject
            level; ``"session"`` splits at the session level (allows same
            subject in multiple splits — less conservative).
        seed:
            Reproducibility seed.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"`` — each a list of
        BIDS subject IDs (e.g. ``["sub-01", "sub-02", ...]``).

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> splits = ds.train_test_split(
        ...     test_size=0.2, stratify_by="diagnosis", group_by="subject"
        ... )
        >>> splits["train"]  # doctest: +SKIP
        # ['sub-01', 'sub-03', ...]
        """
        import hashlib

        manifest = self.manifest()
        all_subjects = sorted({
            f"sub-{s}" if not s.startswith("sub-") else s
            for s in manifest.summary.subjects
        })

        # Optional stratification
        label_map: dict[str, str] = {}
        if stratify_by:
            try:
                label_map = self.map_labels(target_column=stratify_by)
            except Exception as exc:
                log.warning("train_test_split: cannot load stratify_by=%r: %s", stratify_by, exc)

        # Group subjects by label for stratified split
        if label_map and stratify_by:
            strata: dict[str, list[str]] = {}
            for sub in all_subjects:
                label = str(label_map.get(sub, "unknown"))
                strata.setdefault(label, []).append(sub)
        else:
            strata = {"_all": all_subjects}

        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []

        for label_group in strata.values():
            # Deterministic shuffle keyed by seed + subject hash
            ordered = sorted(
                label_group,
                key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(),
            )
            n = len(ordered)
            n_test = max(1, round(n * test_size))
            n_val = max(0, round(n * val_size))
            n_train = n - n_test - n_val

            train_ids.extend(ordered[:n_train])
            val_ids.extend(ordered[n_train: n_train + n_val])
            test_ids.extend(ordered[n_train + n_val:])

        return {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }

    # ── Remote lazy streaming ──────────────────────────────────────────────

    def stream_header(
        self,
        *,
        subject: str,
        session: str | None = None,
        modality: str = "T1w",
        run: str | None = None,
    ):
        """Fetch the NIfTI or EDF header for a remote file via byte-range requests.

        No full file download is required — only the header bytes are fetched.

        Parameters
        ----------
        subject:
            Subject ID (with or without the ``sub-`` prefix).
        session:
            Session ID (with or without the ``ses-`` prefix).  ``None`` = any.
        modality:
            BIDS suffix, e.g. ``"T1w"``, ``"bold"``, ``"eeg"``.
        run:
            Run label, e.g. ``"1"``.  ``None`` = any.

        Returns
        -------
        NiftiStreamHeader | EDFStreamHeader
            Parsed header object.

        Examples
        --------
        >>> hdr = Dataset("ds000001").stream_header(subject="01", modality="T1w")
        >>> hdr.shape, hdr.voxel_sizes_mm
        ((256, 256, 176), (1.0, 1.0, 1.0))
        """
        url = self._resolve_modality_url(
            subject=subject, session=session, modality=modality, run=run
        )
        if modality.lower() in ("eeg", "edf", "bdf"):
            from qortex.stream import EDFStreamer
            return EDFStreamer(url).header()
        else:
            from qortex.stream import NiftiStreamer
            return NiftiStreamer(url).header()

    def stream_slice(
        self,
        *,
        subject: str,
        modality: str = "bold",
        run: str | None = None,
        session: str | None = None,
        time_index: int = 0,
        axis: int = 2,
        slice_index: int | None = None,
    ) -> Any:
        """Stream a single 2D slice from a remote NIfTI file.

        Parameters
        ----------
        subject:
            Subject ID.
        modality:
            BIDS suffix, e.g. ``"bold"``, ``"T1w"``, ``"dwi"``.
        time_index:
            Volume index for 4D fMRI (ignored for 3D anatomicals).
        axis:
            Slicing axis for 2D extraction (0=sagittal, 1=coronal, 2=axial).
        slice_index:
            Which slice along ``axis`` to extract.  ``None`` = center slice
            along the chosen ``axis``.

        Returns
        -------
        np.ndarray
            2D array; shape depends on ``axis`` (see ``NiftiStreamer.get_slice``).

        Examples
        --------
        >>> sl = Dataset("ds000001").stream_slice(
        ...     subject="01", modality="bold", run="1", time_index=150
        ... )
        >>> sl.shape  # (64, 64)
        """
        from qortex.stream import NiftiStreamer

        url = self._resolve_modality_url(
            subject=subject, session=session, modality=modality, run=run
        )
        streamer = NiftiStreamer(url)
        hdr = streamer.header()

        if slice_index is None:
            slice_index = hdr.spatial_shape[axis] // 2

        return streamer.get_slice(axis=axis, index=slice_index, t=time_index)

    def get_lazy_array(
        self,
        *,
        subject: str,
        modality: str = "T1w",
        session: str | None = None,
        run: str | None = None,
    ) -> Any:
        """Return a lazy nibabel ArrayProxy without downloading the full file.

        The proxy fetches data on-demand when sliced, using nibabel's built-in
        lazy evaluation for local paths.  For remote URLs, returns the
        NiftiStreamer's ``get_lazy_array()`` object (header + deferred load).

        Parameters
        ----------
        subject:
            Subject ID.
        modality:
            BIDS suffix.

        Returns
        -------
        nibabel.ArrayProxy or NiftiStreamer-proxy object

        Examples
        --------
        >>> arr = Dataset("ds000001").get_lazy_array(subject="01", modality="T1w")
        >>> center = arr[128, 128, 88]   # triggers partial load on remote
        """
        from qortex.stream import NiftiStreamer

        url = self._resolve_modality_url(
            subject=subject, session=session, modality=modality, run=run
        )
        streamer = NiftiStreamer(url)
        return streamer.get_lazy_array()

    def prefetch_metadata(
        self,
        modalities: list[str] | None = None,
        *,
        concurrency: int = 16,
    ) -> dict[str, Any]:
        """Prefetch remote headers for all subjects concurrently.

        Populates the NiftiStreamer/EDFStreamer header cache for all files of
        the specified modalities, enabling subsequent ``stream_slice`` calls to
        skip the header fetch round-trip.

        Parameters
        ----------
        modalities:
            BIDS suffixes to prefetch headers for.  ``None`` = all image files.
        concurrency:
            Number of parallel Range requests to fire simultaneously.

        Returns
        -------
        dict
            ``{manifest_path: header_object}`` mapping for all successfully
            prefetched files.

        Examples
        --------
        >>> Dataset("ds000001").prefetch_metadata(modalities=["eeg", "meg"])
        """
        import concurrent.futures
        from qortex.client.remote import _pick_url
        from qortex.stream import EDFStreamer, NiftiStreamer

        manifest = self.manifest()
        signal_exts = {".edf", ".bdf", ".fif"}
        nifti_exts = {".nii", ".nii.gz"}

        def _is_target(fr) -> bool:
            if modalities:
                if fr.suffix not in modalities:
                    return False
            ext = "." + ".".join(fr.path.rsplit(".", 2)[1:]) if ".gz" in fr.path else fr.extension
            return ext.lower() in (signal_exts | nifti_exts)

        targets = [fr for fr in manifest.files if _is_target(fr)]
        if not targets:
            log.warning("prefetch_metadata: no matching files for modalities=%r", modalities)
            return {}

        def _fetch_one(fr):
            url = _pick_url(fr)
            if not url:
                return fr.path, None
            try:
                if any(fr.path.lower().endswith(ext) for ext in (".edf", ".bdf")):
                    hdr = EDFStreamer(url).header()
                else:
                    hdr = NiftiStreamer(url).header()
                return fr.path, hdr
            except Exception as exc:
                log.debug("prefetch_metadata: failed for %s: %s", fr.path, exc)
                return fr.path, None

        results: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_fetch_one, fr): fr for fr in targets}
            for fut in concurrent.futures.as_completed(futures):
                path, hdr = fut.result()
                if hdr is not None:
                    results[path] = hdr

        log.info(
            "prefetch_metadata: fetched %d/%d headers for modalities=%r",
            len(results), len(targets), modalities,
        )
        return results

    # ── Internal resolver ─────────────────────────────────────────────────

    def _resolve_modality_url(
        self,
        *,
        subject: str,
        session: str | None,
        modality: str,
        run: str | None,
    ) -> str:
        from qortex.client.remote import _pick_url

        sub = subject.removeprefix("sub-")
        ses = session.removeprefix("ses-") if session else None

        # Sidecar/companion extensions never carry the primary data payload —
        # a .json/.tsv file can share a data file's BIDS suffix (e.g. both
        # "sub-01_T1w.nii.gz" and its "sub-01_T1w.json" sidecar have
        # suffix == "T1w"), so they must be excluded here or this resolver
        # can hand a NIfTI/EDF streamer a JSON sidecar's URL by mistake.
        _NON_DATA_EXTENSIONS = {".json", ".tsv", ".bval", ".bvec"}

        manifest = self.manifest()
        candidates = [
            fr for fr in manifest.files
            if fr.subject == sub
            and (ses is None or fr.session == ses)
            and fr.suffix == modality
            and (run is None or fr.run == run)
            and fr.extension not in _NON_DATA_EXTENSIONS
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No data file found for sub={sub!r} ses={ses!r} "
                f"suffix={modality!r} run={run!r} (sidecar-only matches were excluded)"
            )
        url = _pick_url(candidates[0])
        if not url:
            raise FileNotFoundError(
                f"No URL for {candidates[0].path!r}"
            )
        return url

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
        label_policy: LabelPolicy | None = None,
        split_strategy: Literal["subject", "subject_session", "recording"] = "subject",
    ) -> CanTrainReport:
        """Assess whether the dataset can support supervised training."""
        from qortex.decision import can_train

        return can_train(
            self.manifest(),
            modality=modality,
            target=target,
            local_path=local_path or self._data_dir,
            label_policy=label_policy,
            split_strategy=split_strategy,
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
        demographics or the API path fails with a recognised transient error.

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
            when the fallback path is taken.  The dataframe carries a
            ``_source`` attribute (``"api"`` or ``"participants.tsv"``)
            indicating which path was used.

        Examples
        --------
        >>> df = Dataset("ds000117").participants()
        >>> df.filter(pl.col("sex") == "M")["age"].mean()
        >>> df.attrs.get("_source")   # "api" or "participants.tsv"
        """
        import polars as pl
        from qortex.client.graphql import OpenNeuroClient
        from qortex.core.exceptions import APIError, QortexError

        _source: str = "participants.tsv"

        if prefer_api:
            with OpenNeuroClient(token=self._token) as client:
                try:
                    snap_tag = self.snapshot or client.get_latest_snapshot(self.dataset_id).tag
                    summary = client.get_snapshot_summary(self.dataset_id, snap_tag)
                    if summary.subject_demographics:
                        df = summary.demographics_dataframe()
                        _source = "api"
                        log.debug(
                            "participants(%s): loaded %d rows from API summary",
                            self.dataset_id, len(df),
                        )
                        return df
                    # API returned successfully but no demographics — use TSV
                    log.debug(
                        "participants(%s): API summary has no subject_demographics; "
                        "falling back to participants.tsv", self.dataset_id,
                    )
                except (APIError, QortexError) as exc:
                    log.warning(
                        "participants(%s): API path failed (%s: %s); "
                        "falling back to participants.tsv",
                        self.dataset_id, type(exc).__name__, exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Unexpected failures (network, schema change) are logged at
                    # WARNING rather than silently dropped so callers can detect them.
                    log.warning(
                        "participants(%s): unexpected error in API path (%s: %s); "
                        "falling back to participants.tsv",
                        self.dataset_id, type(exc).__name__, exc,
                    )

        # Fallback: fetch participants.tsv via CDN
        manifest = self.manifest()
        from qortex.client.remote import RemoteFileGateway, best_url_for_path
        gateway = RemoteFileGateway()
        url = best_url_for_path(manifest, "participants.tsv")
        if url is None:
            log.warning(
                "participants(%s): participants.tsv not found in manifest; "
                "returning empty DataFrame", self.dataset_id,
            )
            return pl.DataFrame(
                schema={"participant_id": pl.Utf8, "age": pl.Int64, "sex": pl.Utf8, "group": pl.Utf8}
            )
        df = gateway.fetch_tsv(url)
        log.debug(
            "participants(%s): loaded %d rows from participants.tsv (CDN)",
            self.dataset_id, len(df),
        )
        return df

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
        # Strip BIDS prefixes so both "01" and "sub-01" work identically.
        # Manifests store entity values without prefixes (e.g. subject="01").
        subject = subject.removeprefix("sub-") if subject else subject
        session = session.removeprefix("ses-") if session else session

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

    def sidecar(self, path: str, *, strict: bool = False):
        """Fetch and merge BIDS JSON sidecars for a file path, without downloading.

        Follows BIDS inheritance: most-general (dataset root) sidecar values
        are overridden by more-specific ones (subject → session → file-level).
        Sidecar fetch failures are logged at WARNING level and recorded in the
        returned dict under the ``"_sidecar_warnings"`` key so callers can
        detect partial merges without crashing.

        Parameters
        ----------
        path:
            BIDS-relative path, e.g. ``"sub-01/eeg/sub-01_task-rest_eeg.set"``.
        strict:
            When True, raise the first sidecar fetch/parse error rather than
            continuing.  Useful for debugging incomplete CDN responses or
            unexpected JSON schema changes.

        Returns
        -------
        dict
            Merged JSON sidecar key-value pairs.  On partial failure the dict
            includes ``"_sidecar_warnings": [...]`` listing failed sidecar
            paths and their error messages.

        Examples
        --------
        >>> meta = Dataset("ds004130").sidecar("sub-01/eeg/sub-01_task-rest_eeg.set")
        >>> meta["SamplingFrequency"]
        256
        >>> meta.get("_sidecar_warnings")   # None or list of {"path": ..., "error": ...}
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
        sidecar_warnings: list[dict] = []

        for fr in sidecar_records:
            url = _pick_url(fr)
            if not url:
                continue
            try:
                data = gateway.fetch_json(url)
                merged.update(data)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "sidecar(%s): failed to fetch/parse %s — %s",
                    path, fr.path, msg,
                )
                sidecar_warnings.append({"path": fr.path, "error": msg})
                if strict:
                    raise

        if sidecar_warnings:
            merged["_sidecar_warnings"] = sidecar_warnings
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
        max_sidecars: int | None = None,
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
        max_sidecars:
            Cap the number of signal files whose sidecars are fetched (passed
            through to :meth:`SignalBudgetEstimator.estimate`). Acquisition
            parameters are near-homogeneous within a BIDS dataset, so a
            bounded sample yields the same per-modality modes as an
            exhaustive scan at a fraction of the round trips. ``None`` keeps
            the exhaustive behavior.

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
            max_sidecars=max_sidecars,
        )

    # ── Visual QC ────────────────────────────────────────────────────────

    def visualize(
        self,
        *,
        subjects: list[str] | None = None,
        suffixes: list[str] | None = None,
        datatypes: list[str] | None = None,
        local_path: Path | None = None,
        output_dir: Path | None = None,
        mode: str = "thumbnail",
        max_files: int = 12,
        n_per_suffix: int = 2,
        open_browser: bool = False,
    ) -> "VisualAuditReport":
        """Render a visual sample of locally-downloaded dataset files.

        Returns a :class:`VisualAuditReport` whose ``.show()`` opens an HTML
        gallery in the browser and ``.to_html()`` writes the file.

        Each NIfTI reads **exactly one center slice** from the nibabel
        ArrayProxy — the full volume is never loaded.

        Parameters
        ----------
        subjects:
            Restrict to these subject IDs (without ``sub-`` prefix).
            ``None`` includes all subjects.
        suffixes:
            BIDS suffixes to include, e.g. ``["T1w", "bold"]``.
            ``None`` includes all visualizable suffixes.
        datatypes:
            BIDS datatype folders, e.g. ``["anat", "func", "eeg"]``.
        local_path:
            Root of the locally downloaded dataset.  Falls back to the
            default cache directory (set during ``download()``).
        output_dir:
            If given, write the HTML report to
            ``{output_dir}/visual_audit.html``.
        mode:
            Rendering mode.  ``"thumbnail"`` (default) reads one slice per
            file.  ``"static"`` renders a multi-panel PNG.
        max_files:
            Hard cap on the number of files rendered.
        n_per_suffix:
            Maximum files per BIDS suffix; ensures variety when many
            suffixes are present.
        open_browser:
            Open the generated HTML report automatically.

        Returns
        -------
        VisualAuditReport

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01", "02"], suffixes=["T1w"])
        >>> report = ds.visualize(suffixes=["T1w"])
        >>> report.show()
        """
        from qortex.visualize._audit import run_visual_audit_with_manifest

        root = Path(local_path) if local_path else self._resolve_data_dir()
        if not root.exists():
            raise RuntimeError(
                f"Local data directory not found: {root}\n"
                "Download the data first with ds.download() or pass local_path=."
            )

        manifest = self.manifest()
        report = run_visual_audit_with_manifest(
            dataset_id=self.dataset_id,
            manifest_files=manifest.files,
            local_root=root,
            subjects=subjects,
            suffixes=suffixes,
            datatypes=datatypes,
            max_files=max_files,
            n_per_suffix=n_per_suffix,
        )

        if output_dir is not None:
            out = Path(output_dir) / "visual_audit.html"
            report.to_html(out)

        if open_browser:
            report.show()

        return report

    def visual_audit(
        self,
        output_dir: Path | str,
        *,
        subjects: list[str] | None = None,
        suffixes: list[str] | None = None,
        datatypes: list[str] | None = None,
        local_path: Path | None = None,
        n_per_suffix: int = 3,
        max_files: int = 24,
        open_browser: bool = False,
    ) -> "VisualAuditReport":
        """Run a comprehensive visual QC report across the dataset.

        Generates a dark-theme HTML gallery with one thumbnail per file,
        grouped by BIDS suffix.  Useful for catching reconstruction
        artifacts, intensity outliers, or incomplete downloads at a glance.

        Each NIfTI thumbnail reads **one center slice** — the full volume is
        never loaded.

        Parameters
        ----------
        output_dir:
            Directory where the report is written.
            File: ``{output_dir}/visual_audit.html``.
        subjects:
            Subjects to include.  ``None`` = all downloaded subjects.
        suffixes:
            BIDS suffixes to include.  ``None`` = all visualizable suffixes.
        datatypes:
            BIDS datatype folders to filter on.
        local_path:
            Override the default download path.
        n_per_suffix:
            Files per BIDS suffix.  3 gives variety without overloading.
        max_files:
            Hard cap on total renders.
        open_browser:
            Open the report in the browser when done.

        Returns
        -------
        VisualAuditReport
            Call ``.to_html()``, ``.show()``, or ``.summary()``.

        Examples
        --------
        >>> ds = Dataset("ds000001")
        >>> ds.download(subjects=["01", "02", "03"], suffixes=["T1w", "bold"])
        >>> report = ds.visual_audit("qc/", suffixes=["T1w"])  # reads 1 slice per file
        >>> print(report.summary())
        """
        return self.visualize(
            subjects=subjects,
            suffixes=suffixes,
            datatypes=datatypes,
            local_path=local_path,
            output_dir=output_dir,
            mode="thumbnail",
            max_files=max_files,
            n_per_suffix=n_per_suffix,
            open_browser=open_browser,
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
    "VisualAuditReport",
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
    # Stream
    "NiftiStreamer",
    "EDFStreamer",
    # Runtime
    "BIDSImageDataset",
    "BIDSSignalDataset",
    "BIDSEpochDataset",
    "MONAIDictBuilder",
    "TorchEEGBridge",
    # Cohort / federated
    "FederatedCohort",
    "FederatedSubject",
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
