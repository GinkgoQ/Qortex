"""Qortex CLI — typer application with all subcommands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(
    name="qortex",
    help="Qortex by GinkgoQ — ML-ready neurodata from OpenNeuro.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ── search ───────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Free-text search term"),
    modality: Optional[str] = typer.Option(None, "--modality", "-m", help="Filter by modality (eeg, mri, …)"),
    task: Optional[str] = typer.Option(None, "--task", help="Filter by BIDS/OpenNeuro task"),
    author: Optional[str] = typer.Option(None, "--author", help="Filter by author substring"),
    license: Optional[str] = typer.Option(None, "--license", help="Filter by exact license value"),
    min_subjects: Optional[int] = typer.Option(None, "--min-subjects", "-n"),
    max_size_gb: Optional[float] = typer.Option(None, "--max-size-gb", help="Maximum latest snapshot size in GB"),
    has_events: Optional[bool] = typer.Option(None, "--has-events/--no-events", help="Filter by event-file availability when indexed"),
    has_derivatives: Optional[bool] = typer.Option(None, "--has-derivatives/--no-derivatives", help="Filter by derivative-file availability when indexed"),
    limit: int = typer.Option(20, "--limit", "-l"),
    offset: int = typer.Option(0, "--offset"),
    refresh_catalog: bool = typer.Option(False, "--refresh", help="Refresh catalog before searching"),
    deep: bool = typer.Option(False, "--deep", help="During --refresh, also ingest file summaries"),
) -> None:
    """Search the local OpenNeuro catalog."""
    if refresh_catalog:
        from qortex.catalog.refresh import refresh as do_refresh
        typer.echo("Refreshing catalog from OpenNeuro …")
        n = do_refresh(progress=False, include_file_summary=deep)
        typer.echo(f"  {n} datasets indexed.")

    from qortex.catalog.search import search as do_search
    results = do_search(
        query=query,
        modality=modality,
        task=task,
        author=author,
        license=license,
        min_subjects=min_subjects,
        max_size_gb=max_size_gb,
        has_events=has_events,
        has_derivatives=has_derivatives,
        limit=limit,
        offset=offset,
    )
    if not results:
        typer.echo("No results found.")
        raise typer.Exit(0)

    for r in results:
        mods = ", ".join(r.get("modalities") or [])
        tasks = ", ".join((r.get("tasks") or [])[:4])
        score = r.get("score")
        score_text = f"  score={score}" if score is not None else ""
        typer.echo(
            f"[{r['dataset_id']}]  {r.get('name') or '(no name)'}  "
            f"  subjects={r.get('n_subjects')}  modalities={mods}  tasks={tasks}{score_text}"
        )


# ── inspect ───────────────────────────────────────────────────────────────────

@app.command()
def inspect(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write manifest JSON here"),
) -> None:
    """Fetch and display the manifest for a dataset."""
    from qortex.client.graphql import OpenNeuroClient
    from qortex.manifest.builder import ManifestBuilder

    typer.echo(f"Fetching manifest for {dataset_id} …")
    client = OpenNeuroClient()
    builder = ManifestBuilder()
    snap_ref, raw_files = _fetch_files(client, dataset_id, snapshot)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    s = manifest.summary
    typer.echo(f"\nDataset:   {manifest.dataset_id}")
    typer.echo(f"Snapshot:  {manifest.snapshot}")
    typer.echo(f"Files:     {s.file_count}")
    typer.echo(f"Subjects:  {s.n_subjects}")
    typer.echo(f"Size:      {s.total_size / 1e9:.2f} GB")
    typer.echo(f"Modalities:{', '.join(s.modalities) or 'N/A'}")

    if output:
        from qortex.manifest.builder import save_manifest
        save_manifest(manifest, output)
        typer.echo(f"\nManifest saved to {output}")


# ── metadata ─────────────────────────────────────────────────────────────────

@app.command()
def metadata(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    download: bool = typer.Option(False, "--download", help="Download metadata files instead of listing them"),
    limit: int = typer.Option(50, "--limit", "-l"),
) -> None:
    """List or download essential metadata and sidecar/table files."""
    from qortex import Dataset

    ds = Dataset(dataset_id, snapshot=snapshot)
    if download:
        result = ds.download_metadata(output_dir=output_dir)
        typer.echo(result.report())
        return

    for file in ds.metadata_files()[:limit]:
        size = f"{(file.size or 0) / 1e3:.1f} KB" if file.size is not None else "unknown"
        typer.echo(f"{file.path}  {size}")


# ── preview ──────────────────────────────────────────────────────────────────

@app.command()
def preview(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    path: str = typer.Argument(..., help="BIDS-relative file path"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    local_path: Optional[Path] = typer.Option(None, "--local-path"),
    rows: int = typer.Option(5, "--rows", "-n"),
    max_bytes: int = typer.Option(64_000, "--max-bytes"),
) -> None:
    """Preview first rows/text of a local or remote file without full download."""
    from qortex import Dataset

    ds = Dataset(dataset_id, snapshot=snapshot)
    result = ds.preview(
        path,
        local_path=local_path,
        n_rows=rows,
        max_bytes=max_bytes,
    )
    typer.echo(f"{result.path} ({result.source}, {result.bytes_read} bytes)")
    if result.columns:
        typer.echo("Columns: " + ", ".join(result.columns))
    if result.rows:
        for row in result.rows:
            typer.echo(row)
    elif result.text:
        typer.echo(result.text)


# ── decision workflows ───────────────────────────────────────────────────────

@app.command("doctor")
def doctor_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    local_path: Optional[Path] = typer.Option(None, "--local-path"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Explain whether a dataset is usable and what the next real action is."""
    ds = _dataset(dataset_id, snapshot)
    report = ds.doctor(local_path=local_path)
    typer.echo(report.to_text())
    _write_model_json(report, output_json)


@app.command("minimum")
def minimum_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    goal: str = typer.Option("first-batch", "--goal", help="label-check|first-batch|validation|metadata"),
    modality: Optional[str] = typer.Option(None, "--modality"),
    target: Optional[str] = typer.Option(None, "--target"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    download: bool = typer.Option(False, "--download", help="Execute the plan after printing it"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Plan the smallest real download needed for a concrete workflow goal."""
    from qortex.fetch.engine import DownloadEngine

    ds = _dataset(dataset_id, snapshot)
    try:
        report = ds.minimum(
            goal=goal,
            modality=modality,
            target=target,
            output_dir=output_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2)
    typer.echo(report.to_text())
    _write_model_json(report, output_json)
    if download:
        result = DownloadEngine().execute(report.plan)
        typer.echo(result.report())


@app.command("can-train")
def can_train_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    modality: Optional[str] = typer.Option(None, "--modality"),
    target: Optional[str] = typer.Option(None, "--target"),
    local_path: Optional[Path] = typer.Option(None, "--local-path"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Assess if supervised training is actually supported by this dataset."""
    ds = _dataset(dataset_id, snapshot)
    report = ds.can_train(modality=modality, target=target, local_path=local_path)
    typer.echo(report.to_text())
    _write_model_json(report, output_json)


@app.command("first-batch")
def first_batch_cmd(
    dataset_id: Optional[str] = typer.Option(None, "--dataset", help="Dataset ID when reading from local BIDS data"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    artifact: Optional[Path] = typer.Option(None, "--artifact", help="Converted Qortex artifact directory"),
    local_path: Optional[Path] = typer.Option(None, "--local-path", help="Downloaded BIDS dataset root"),
    modality: Optional[str] = typer.Option(None, "--modality"),
    target: Optional[str] = typer.Option(None, "--target"),
    limit: int = typer.Option(8, "--limit", "-n"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Print first artifact rows or the smallest plan needed to produce them."""
    from qortex.decision import first_batch

    if artifact is not None:
        report = first_batch(artifact_path=artifact, limit=limit)
    else:
        if dataset_id is None:
            typer.echo("Provide --artifact or --dataset.", err=True)
            raise typer.Exit(2)
        ds = _dataset(dataset_id, snapshot)
        report = ds.first_batch(
            local_path=local_path,
            modality=modality,
            target=target,
            limit=limit,
        )
    typer.echo(report.to_text())
    _write_model_json(report, output_json)


@app.command("leakage-check")
def leakage_check_cmd(
    artifact: Path = typer.Argument(..., help="Converted Qortex artifact directory"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Check a converted artifact for subject/source split leakage."""
    from qortex.decision import leakage_check

    report = leakage_check(artifact)
    typer.echo(report.to_text())
    _write_model_json(report, output_json)


@app.command("content-status")
def content_status_cmd(
    path: Path = typer.Argument(..., help="Local BIDS dataset root"),
    dataset_id: Optional[str] = typer.Option(None, "--dataset", help="Dataset ID for manifest reconciliation"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Check local files, pointer-like content, and optional manifest mismatches."""
    from qortex.decision import content_status

    manifest = _dataset(dataset_id, snapshot).manifest() if dataset_id else None
    report = content_status(path, manifest=manifest)
    typer.echo(report.to_text())
    _write_model_json(report, output_json)


@app.command("make-recipe")
def make_recipe_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds004130"),
    output: Path = typer.Argument(..., help="Recipe JSON path"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    modality: Optional[str] = typer.Option(None, "--modality"),
    target: Optional[str] = typer.Option(None, "--target"),
    split: str = typer.Option("subject", "--split"),
    goal: str = typer.Option("first-batch", "--goal"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    metadata_only: bool = typer.Option(False, "--metadata-only"),
) -> None:
    """Write a reproducible Qortex workflow recipe."""
    from qortex.decision import Recipe, write_recipe

    try:
        recipe = Recipe(
            dataset_id=dataset_id,
            snapshot=snapshot,
            modality=modality,
            target=target,
            split=split,
            goal=goal,
            output_dir=str(output_dir) if output_dir else None,
            metadata_only=metadata_only,
        )
    except Exception as exc:
        typer.echo(f"Invalid recipe: {exc}", err=True)
        raise typer.Exit(2)
    write_recipe(recipe, output)
    typer.echo(f"Recipe saved to {output}")


@app.command("run-recipe")
def run_recipe_cmd(
    recipe_path: Path = typer.Argument(..., help="Recipe JSON path"),
    download: bool = typer.Option(False, "--download", help="Execute the planned download"),
) -> None:
    """Load a Qortex recipe and run its minimum download decision."""
    from qortex.decision import read_recipe
    from qortex.fetch.engine import DownloadEngine

    recipe = read_recipe(recipe_path)
    ds = _dataset(recipe.dataset_id, recipe.snapshot)
    report = ds.minimum(
        goal=recipe.goal,
        modality=recipe.modality,
        target=recipe.target,
        output_dir=Path(recipe.output_dir) if recipe.output_dir else None,
    )
    typer.echo(report.to_text())
    if download:
        result = DownloadEngine().execute(report.plan)
        typer.echo(result.report())


# ── plan ──────────────────────────────────────────────────────────────────────

@app.command()
def plan(
    dataset_id: str = typer.Argument(...),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    subjects: Optional[str] = typer.Option(None, "--subjects", help="Comma-separated subject IDs"),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma-separated task names"),
    modalities: Optional[str] = typer.Option(None, "--modalities"),
    include_derivatives: bool = typer.Option(False, "--include-derivatives"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Compute a download plan without downloading anything."""
    from qortex.client.graphql import OpenNeuroClient
    from qortex.core.entities import SelectionSpec
    from qortex.manifest.builder import ManifestBuilder
    from qortex.plan.planner import DownloadPlanner

    client = OpenNeuroClient()
    builder = ManifestBuilder()
    snap_ref, raw_files = _fetch_files(client, dataset_id, snapshot)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    spec = SelectionSpec(
        subjects=subjects.split(",") if subjects else None,
        tasks=tasks.split(",") if tasks else None,
        modalities=modalities.split(",") if modalities else None,
        include_derivatives=include_derivatives,
    )

    target = output_dir or (Path.home() / ".cache" / "qortex" / "datasets" / dataset_id)
    planner = DownloadPlanner(check_disk_space=False)
    download_plan = planner.plan(manifest, spec, target)

    typer.echo(f"\nDownload plan for {dataset_id}")
    typer.echo(f"  Files to download: {len(download_plan.files)}")
    size_gb = sum(f.size or 0 for f in download_plan.files) / 1e9
    typer.echo(f"  Estimated size:    {size_gb:.2f} GB")
    typer.echo(f"  Target directory:  {download_plan.target_dir}")


# ── download ─────────────────────────────────────────────────────────────────

@app.command()
def download(
    dataset_id: str = typer.Argument(...),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    subjects: Optional[str] = typer.Option(None, "--subjects"),
    tasks: Optional[str] = typer.Option(None, "--tasks"),
    modalities: Optional[str] = typer.Option(None, "--modalities"),
    include_derivatives: bool = typer.Option(False, "--include-derivatives"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Download a dataset (or a subset of it) from OpenNeuro."""
    from qortex.client.graphql import OpenNeuroClient
    from qortex.core.entities import SelectionSpec
    from qortex.fetch.engine import DownloadEngine
    from qortex.manifest.builder import ManifestBuilder
    from qortex.plan.planner import DownloadPlanner

    client = OpenNeuroClient()
    builder = ManifestBuilder()
    snap_ref, raw_files = _fetch_files(client, dataset_id, snapshot)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    spec = SelectionSpec(
        subjects=subjects.split(",") if subjects else None,
        tasks=tasks.split(",") if tasks else None,
        modalities=modalities.split(",") if modalities else None,
        include_derivatives=include_derivatives,
    )

    target = output_dir or (Path.home() / ".cache" / "qortex" / "datasets" / dataset_id)
    planner = DownloadPlanner()
    download_plan = planner.plan(manifest, spec, target)

    n = len(download_plan.files)
    size_gb = sum(f.size or 0 for f in download_plan.files) / 1e9
    typer.echo(f"Plan: {n} files, {size_gb:.2f} GB → {download_plan.target_dir}")

    if dry_run:
        typer.echo("Dry run — nothing downloaded.")
        return

    engine = DownloadEngine()
    result = engine.execute(download_plan)
    typer.echo(
        f"\nDone: {result.n_downloaded} downloaded, "
        f"{result.n_skipped} skipped, "
        f"{result.n_failed} failed."
    )


# ── validate ─────────────────────────────────────────────────────────────────

@app.command()
def validate(
    data_dir: Path = typer.Argument(..., help="Local BIDS dataset root"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="BIDS Validator config JSON"),
    output_json: Optional[Path] = typer.Option(None, "--json-output", help="Write normalized report JSON"),
    output_md: Optional[Path] = typer.Option(None, "--markdown-output", help="Write Markdown report"),
    output_html: Optional[Path] = typer.Option(None, "--html-output", help="Write HTML report"),
    ignore_warnings: bool = typer.Option(False, "--ignore-warnings"),
    ignore_nifti_headers: bool = typer.Option(False, "--ignore-nifti-headers"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write validation cache"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Force validator execution and replace cache entry"),
    timeout: float = typer.Option(600.0, "--timeout"),
) -> None:
    """Run the official BIDS Validator and normalize its JSON report."""
    from qortex.validation import validate_bids

    try:
        report = validate_bids(
            data_dir,
            config_path=config,
            ignore_warnings=ignore_warnings,
            ignore_nifti_headers=ignore_nifti_headers,
            timeout_s=timeout,
            use_cache=not no_cache,
            refresh_cache=refresh_cache,
        )
    except Exception as exc:
        typer.echo(f"Validation failed to run: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(report.summary())
    if output_json:
        report.to_json(output_json)
        typer.echo(f"\nJSON report saved to {output_json}")
    if output_md:
        report.to_markdown(output_md)
        typer.echo(f"Markdown report saved to {output_md}")
    if output_html:
        report.to_html(output_html)
        typer.echo(f"HTML report saved to {output_html}")


# ── local-index ───────────────────────────────────────────────────────────────

@app.command("local-index")
def local_index(
    data_dir: Path = typer.Argument(..., help="Local BIDS dataset root"),
    manifest_dir: Optional[Path] = typer.Option(None, "--manifest-dir", help="Directory containing qortex manifest files"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
    no_pybids: bool = typer.Option(False, "--no-pybids", help="Use Qortex filesystem indexer only"),
) -> None:
    """Index a local BIDS tree and optionally reconcile it with a manifest."""
    from qortex.indexing import index_local_bids

    manifest = None
    if manifest_dir is not None:
        from qortex.manifest.builder import load_manifest

        manifest = load_manifest(manifest_dir)

    report = index_local_bids(
        data_dir,
        manifest=manifest,
        use_pybids=not no_pybids,
    )
    typer.echo(report.summary())
    if output_json:
        report.to_json(output_json)
        typer.echo(f"\nLocal index report saved to {output_json}")


# ── eda ───────────────────────────────────────────────────────────────────────

@app.command()
def eda(
    dataset_id: str = typer.Argument(...),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", "-s"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path for HTML report"),
) -> None:
    """Run EDA on a dataset and generate an HTML report."""
    from qortex.client.graphql import OpenNeuroClient
    from qortex.eda.report import EDAEngine
    from qortex.manifest.builder import ManifestBuilder

    client = OpenNeuroClient()
    builder = ManifestBuilder()
    snap_ref, raw_files = _fetch_files(client, dataset_id, snapshot)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    engine = EDAEngine(manifest)
    report = engine.run()

    typer.echo(f"\nBIDS score:        {report.quality.bids_score:.0f}/100")
    typer.echo(f"ML-readiness:      {report.quality.ml_readiness_score:.0f}/100")
    typer.echo(f"Loadability:       {report.quality.loadability_score:.0f}/100")

    if report.quality.issues:
        typer.echo("\nIssues:")
        for issue in report.quality.issues:
            typer.echo(f"  • {issue}")

    if output and report.html:
        report.to_html(output)
        typer.echo(f"\nHTML report saved to {output}")


# ── convert ───────────────────────────────────────────────────────────────────

@app.command()
def convert(
    data_dir: Path = typer.Argument(..., help="Downloaded dataset root"),
    output_dir: Path = typer.Argument(..., help="Output directory for the ML artifact"),
    output_format: str = typer.Option("parquet", "--format", "-f", help="parquet|zarr|hdf5|webdataset|huggingface|tfrecord"),
    window_duration: Optional[float] = typer.Option(None, "--window", "-w", help="Window duration in seconds"),
    window_overlap: float = typer.Option(0.0, "--overlap"),
    split_strategy: str = typer.Option("subject", "--split"),
    shard_size: int = typer.Option(1000, "--shard-size"),
) -> None:
    """Convert a downloaded BIDS dataset to an ML-ready artifact."""
    from qortex.convert.pipeline import ConversionPipeline
    from qortex.convert.splits import SplitSpec
    from qortex.convert.windows import WindowSpec
    from qortex.manifest.builder import load_manifest

    manifest_dir = data_dir / ".qortex"
    try:
        manifest = load_manifest(manifest_dir)
    except Exception as e:
        typer.echo(f"Cannot load manifest from {manifest_dir}: {e}", err=True)
        raise typer.Exit(1)

    win_spec = (
        WindowSpec(duration_s=window_duration, overlap=window_overlap)
        if window_duration else None
    )

    pipeline = ConversionPipeline(
        manifest=manifest,
        data_dir=data_dir,
        output_dir=output_dir,
        output_format=output_format,
        window_spec=win_spec,
        split_spec=SplitSpec(strategy=split_strategy),
        shard_size=shard_size,
    )
    result = pipeline.run()
    typer.echo(
        f"Converted {result.n_samples} samples ({result.n_subjects} subjects) "
        f"in {result.elapsed:.1f}s → {result.output_path}"
    )


# ── cache ─────────────────────────────────────────────────────────────────────

@app.command()
def cache(
    action: str = typer.Argument("info", help="info|list|remove|clear"),
    dataset_id: Optional[str] = typer.Option(None, "--dataset-id"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot"),
    yes: bool = typer.Option(False, "--yes", help="Required for destructive actions"),
) -> None:
    """Show or manage the local Qortex cache."""
    from qortex.core.config import get_config
    from qortex.lake.registry import LocalRegistry

    cfg = get_config()

    if action == "info":
        cache_dir = cfg.cache_dir
        if cache_dir.exists():
            total = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            typer.echo(f"Cache:   {cache_dir}")
            typer.echo(f"Size:    {total / 1e9:.2f} GB")
        else:
            typer.echo(f"Cache directory does not exist: {cache_dir}")
    elif action == "list":
        registry = LocalRegistry(cfg)
        try:
            entries = registry.list_all()
        finally:
            registry.close()
        if not entries:
            typer.echo("No downloaded snapshots are registered.")
            return
        for entry in entries:
            typer.echo(
                f"{entry.dataset_id}  snapshot={entry.snapshot}  "
                f"files={entry.n_files}  failed={entry.n_failed}  "
                f"size={entry.total_bytes / 1e9:.2f} GB  dir={entry.data_dir}"
            )
    elif action == "remove":
        if not dataset_id or not snapshot:
            typer.echo("--dataset-id and --snapshot are required for cache remove.", err=True)
            raise typer.Exit(1)
        registry = LocalRegistry(cfg)
        try:
            entry = registry.get(dataset_id, snapshot)
            if entry is None:
                typer.echo(f"No registry entry for {dataset_id} snapshot {snapshot}.")
                return
            registry.remove(dataset_id, snapshot)
        finally:
            registry.close()
        typer.echo(f"Removed registry entry for {dataset_id} snapshot {snapshot}.")
    elif action == "clear":
        import shutil
        cache_dir = cfg.cache_dir
        if not yes:
            typer.echo("Refusing to clear cache without --yes.", err=True)
            raise typer.Exit(1)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            typer.echo(f"Cache cleared: {cache_dir}")
        else:
            typer.echo("Cache is already empty.")
    else:
        typer.echo(f"Unknown action: {action}. Use info, list, remove, or clear.", err=True)
        raise typer.Exit(1)


# ── login ─────────────────────────────────────────────────────────────────────

@app.command()
def login(
    token: Optional[str] = typer.Option(None, "--token", "-t", help="API token (or prompted interactively)"),
    delete: bool = typer.Option(False, "--delete", help="Remove stored token"),
) -> None:
    """Save or remove an OpenNeuro API token."""
    from qortex.client.auth import delete_token, prompt_and_save, save_token

    if delete:
        delete_token()
        typer.echo("Token deleted.")
        return

    if token:
        save_token(token)
        typer.echo("Token saved.")
    else:
        prompt_and_save()


# ── catalog ───────────────────────────────────────────────────────────────────

@app.command()
def catalog_refresh(
    max_pages: int = typer.Option(40, "--max-pages"),
    page_size: int = typer.Option(50, "--page-size"),
    deep: bool = typer.Option(False, "--deep", help="Also fetch recursive file manifests and digest file summaries"),
    deep_limit: Optional[int] = typer.Option(None, "--deep-limit", help="Limit how many datasets receive deep file-summary ingestion"),
) -> None:
    """Refresh the local dataset catalog from OpenNeuro."""
    from qortex.catalog.refresh import refresh as do_refresh
    typer.echo("Refreshing catalog …")
    n = do_refresh(
        max_pages=max_pages,
        page_size=page_size,
        include_file_summary=deep,
        file_summary_limit=deep_limit,
        progress=True,
    )
    typer.echo(f"Done. {n} datasets in catalog.")


@app.command("catalog-profile")
def catalog_profile(
    dataset_id: str = typer.Argument(..., help="Dataset ID, e.g. ds000001"),
    refresh: bool = typer.Option(False, "--refresh", help="Fetch and index this dataset before printing"),
    deep: bool = typer.Option(True, "--deep/--metadata-only", help="When refreshing, ingest recursive file summaries"),
    output_json: Optional[Path] = typer.Option(None, "--json-output"),
) -> None:
    """Print a digested local catalog profile for one dataset."""
    if refresh:
        from qortex.catalog.refresh import refresh_dataset

        profile = refresh_dataset(dataset_id, include_file_summary=deep)
    else:
        from qortex.catalog.index import CatalogIndex
        from qortex.core.config import get_config

        cfg = get_config()
        index = CatalogIndex(cfg.cache_dir / "catalog" / "catalog.duckdb")
        try:
            profile = index.profile(dataset_id)
        finally:
            index.close()
        if profile is None:
            typer.echo(f"{dataset_id} is not in the local catalog. Run with --refresh.", err=True)
            raise typer.Exit(1)

    typer.echo(_profile_text(profile))
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        output_json.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        typer.echo(f"\nJSON profile saved to {output_json}")


# ── visualize ────────────────────────────────────────────────────────────────

@app.command("visualize")
def visualize_cmd(
    path: Path = typer.Argument(..., help="File or directory to visualize"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Rendering mode: auto|qc|static|interactive|thumbnail|summary"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML/PNG output to this path"),
    colormap: Optional[str] = typer.Option(None, "--colormap", help="Matplotlib colormap name (e.g. gray, hot, plasma)"),
    modality: Optional[str] = typer.Option(None, "--modality", help="Override modality: mri|ct|pet|fmri|eeg"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Inspect and render a local neuroimaging file (NIfTI, DICOM, EEG, …)."""
    try:
        from qortex import visualize as _viz
    except ImportError as exc:
        typer.echo(
            f"Visualization dependencies not installed: {exc}\n"
            "Install with: pip install qortex[visual]",
            err=True,
        )
        raise typer.Exit(1)

    if not path.exists():
        typer.echo(f"Path does not exist: {path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Inspecting {path} …")
    try:
        asset = _viz.inspect(path)
    except Exception as exc:
        typer.echo(f"Inspection failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(asset.summary())

    if mode == "summary":
        return  # Summary-only mode — don't render

    # Override modality if provided
    if modality:
        asset.modality = modality

    typer.echo(f"\nRendering (mode={mode}) …")
    kwargs: dict = {}
    if colormap:
        kwargs["colormap"] = colormap

    try:
        result = asset.render(mode=mode, **kwargs)
    except Exception as exc:
        typer.echo(f"Rendering failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        suffix = output.suffix.lower()
        if suffix == ".png":
            result.to_png(output)
            typer.echo(f"PNG saved to {output}")
        else:
            result.to_html(output, write_sidecar=True)
            typer.echo(f"HTML saved to {output}")
    elif open_browser:
        result.show()
        typer.echo("Opened in browser.")
    else:
        if result.html:
            typer.echo(f"HTML rendered ({len(result.html)} chars). Use --output or --open to save/view.")
        elif result.png_bytes:
            typer.echo(f"PNG rendered ({len(result.png_bytes)} bytes). Use --output to save.")
        else:
            typer.echo("Rendering produced no output.")


@app.command("dicom-browser")
def dicom_browser_cmd(
    directory: Path = typer.Argument(..., help="DICOM series directory to browse"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML to this path"),
    show_phi: bool = typer.Option(False, "--show-phi", help="Include patient identifiable information in output"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Browse a DICOM series directory and render an interactive series table."""
    try:
        from qortex.visualize.dicom import DicomSeriesBrowser
    except ImportError as exc:
        typer.echo(
            f"DICOM dependencies not installed: {exc}\n"
            "Install with: pip install qortex[dicom]",
            err=True,
        )
        raise typer.Exit(1)

    if not directory.exists() or not directory.is_dir():
        typer.echo(f"Directory does not exist: {directory}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning DICOM directory: {directory} …")
    try:
        browser = DicomSeriesBrowser(directory, show_phi=show_phi)
        series_list = browser.scan()
    except Exception as exc:
        typer.echo(f"DICOM scan failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(series_list)} series:")
    for s in series_list:
        typer.echo(f"  [{s.modality}] {s.description or '(no description)'}  ({s.n_images} images)")

    html = browser.to_html(show_phi=show_phi)

    if output:
        output.write_text(html, encoding="utf-8")
        typer.echo(f"\nHTML browser saved to {output}")
    elif open_browser:
        import tempfile, webbrowser
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(html)
            webbrowser.open(f"file://{f.name}")
        typer.echo("Opened in browser.")
    else:
        typer.echo(f"HTML generated ({len(html)} chars). Use --output or --open to save/view.")


# ── modality-specific QC ──────────────────────────────────────────────────────

@app.command("fmri-qc")
def fmri_qc_cmd(
    bold: Path = typer.Argument(..., help="4D BOLD/fMRI NIfTI file"),
    events: Optional[Path] = typer.Option(None, "--events", help="Optional BIDS events.tsv companion"),
    confounds: Optional[Path] = typer.Option(None, "--confounds", help="Optional confounds TSV companion"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML QC report to this path"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Generate a real fMRI QC summary: mean, variability, tSNR, drift, events/confounds."""
    if not bold.exists():
        typer.echo(f"BOLD file not found: {bold}", err=True)
        raise typer.Exit(1)
    for label, companion in (("events", events), ("confounds", confounds)):
        if companion is not None and not companion.exists():
            typer.echo(f"{label} file not found: {companion}", err=True)
            raise typer.Exit(1)

    try:
        from qortex.visualize.fmri import fmri_summary
    except ImportError as exc:
        typer.echo(f"fMRI QC dependencies not installed: {exc}\nInstall: pip install qortex[visual]", err=True)
        raise typer.Exit(1)

    try:
        fig = fmri_summary(
            bold,
            events_path=events,
            confounds_path=confounds,
            title=f"fMRI QC — {bold.name}",
        )
    except Exception as exc:
        typer.echo(f"fMRI QC failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        _write_visual_output(fig, output)
        typer.echo(f"fMRI QC HTML saved to {output}")
    elif open_browser:
        _show_visual_output(fig)
        typer.echo("Opened in browser.")
    else:
        typer.echo("fMRI QC rendered. Use --output FILE.html or --open to view.")


@app.command("dwi-qc")
def dwi_qc_cmd(
    dwi: Path = typer.Argument(..., help="4D DWI NIfTI file"),
    bval: Optional[Path] = typer.Option(None, "--bval", help="BIDS .bval gradient table"),
    bvec: Optional[Path] = typer.Option(None, "--bvec", help="BIDS .bvec gradient table"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML QC report to this path"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Generate a DWI QC summary: b0/high-b anatomy, shells, and gradient sphere."""
    if not dwi.exists():
        typer.echo(f"DWI file not found: {dwi}", err=True)
        raise typer.Exit(1)
    for label, companion in (("bval", bval), ("bvec", bvec)):
        if companion is not None and not companion.exists():
            typer.echo(f"{label} file not found: {companion}", err=True)
            raise typer.Exit(1)

    try:
        from qortex.visualize.dwi import dwi_summary
    except ImportError as exc:
        typer.echo(f"DWI QC dependencies not installed: {exc}\nInstall: pip install qortex[visual]", err=True)
        raise typer.Exit(1)

    try:
        fig = dwi_summary(dwi, bval_path=bval, bvec_path=bvec, title=f"DWI QC — {dwi.name}")
    except Exception as exc:
        typer.echo(f"DWI QC failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        _write_visual_output(fig, output)
        typer.echo(f"DWI QC HTML saved to {output}")
    elif open_browser:
        _show_visual_output(fig)
        typer.echo("Opened in browser.")
    else:
        typer.echo("DWI QC rendered. Use --output FILE.html or --open to view.")


@app.command("artifact-visualize")
def artifact_visualize_cmd(
    artifact: Path = typer.Argument(..., help="Qortex artifact directory containing artifact_manifest.json"),
    split: str = typer.Option("train", "--split", help="Artifact split to inspect, or 'all'"),
    n: int = typer.Option(16, "--n", help="Number of samples for audit views"),
    sample_index: Optional[int] = typer.Option(None, "--sample-index", help="Render exactly one sample index"),
    compare_splits: bool = typer.Option(False, "--compare-splits", help="Render a train/val/test split comparison"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML output to this path"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Visualize converted artifact samples and split-level QC reports."""
    if not artifact.exists():
        typer.echo(f"Artifact path not found: {artifact}", err=True)
        raise typer.Exit(1)

    try:
        from qortex.artifact import Artifact
    except ImportError as exc:
        typer.echo(f"Artifact visualization dependencies not installed: {exc}", err=True)
        raise typer.Exit(1)

    try:
        art = Artifact.open(artifact)
        if sample_index is not None:
            rendered = art.visualize_sample(sample_index, split=None if split == "all" else split, mode="static")
        elif compare_splits:
            rendered = art.compare_splits(n=n)
        else:
            rendered = art.visual_audit(split=split, n=n)
    except Exception as exc:
        typer.echo(f"Artifact visualization failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        _write_visual_output(rendered, output)
        typer.echo(f"Artifact visualization saved to {output}")
    elif open_browser:
        _show_visual_output(rendered)
        typer.echo("Opened in browser.")
    else:
        typer.echo("Artifact visualization rendered. Use --output FILE.html or --open to view.")


@app.command("compare-masks")
def compare_masks_cmd(
    base: Path = typer.Argument(..., help="Base anatomical image"),
    prediction: Path = typer.Argument(..., help="Predicted segmentation mask"),
    ground_truth: Path = typer.Argument(..., help="Ground-truth segmentation mask"),
    exact: bool = typer.Option(False, "--exact", help="Compute exact full-volume metrics"),
    per_slice: bool = typer.Option(False, "--per-slice", help="Include per-slice Dice metrics in provenance"),
    resample: bool = typer.Option(False, "--resample", help="Resample masks to base geometry if needed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML comparison to this path"),
    open_browser: bool = typer.Option(False, "--open", help="Open the result in the default browser"),
) -> None:
    """Compare predicted and ground-truth masks with TP/FP/FN overlays and Dice metrics."""
    for p in (base, prediction, ground_truth):
        if not p.exists():
            typer.echo(f"File not found: {p}", err=True)
            raise typer.Exit(1)

    try:
        from qortex.visualize import compare_masks
    except ImportError as exc:
        typer.echo(f"Segmentation comparison dependencies not installed: {exc}\nInstall: pip install qortex[visual]", err=True)
        raise typer.Exit(1)

    try:
        rendered = compare_masks(
            base,
            prediction,
            ground_truth,
            exact=exact,
            per_slice=per_slice,
            resample=resample,
        )
    except Exception as exc:
        typer.echo(f"Mask comparison failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        _write_visual_output(rendered, output)
        typer.echo(f"Mask comparison saved to {output}")
    elif open_browser:
        _show_visual_output(rendered)
        typer.echo("Opened in browser.")
    else:
        typer.echo("Mask comparison rendered. Use --output FILE.html or --open to view.")


# ── visualize-overlay ─────────────────────────────────────────────────────────

@app.command("visualize-overlay")
def visualize_overlay_cmd(
    base: Path = typer.Argument(..., help="Base anatomical image (NIfTI path)"),
    overlay_path: Path = typer.Argument(..., help="Overlay volume (mask, stat map, segmentation, PET)"),
    overlay_type: str = typer.Option(
        "auto", "--type", "-t",
        help="Overlay type: auto|mask|stat|labelmap|pet",
    ),
    threshold: float = typer.Option(2.3, "--threshold", help="Z/T threshold for stat map (|z| < threshold → transparent)"),
    alpha: float = typer.Option(0.65, "--alpha", help="Overlay opacity 0–1"),
    colormap: Optional[str] = typer.Option(None, "--colormap", help="Colormap override (e.g. RdBu_r, hot, plasma)"),
    resample: bool = typer.Option(False, "--resample", help="Resample overlay to base geometry if shapes differ"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write HTML to this path"),
    open_browser: bool = typer.Option(False, "--open", help="Open result in default browser"),
) -> None:
    """Overlay a mask, stat map, segmentation, or PET volume on an anatomical image.

    Examples:

        qortex visualize-overlay T1w.nii.gz brain_mask.nii.gz --type mask

        qortex visualize-overlay T1w.nii.gz zmap.nii.gz --type stat --threshold 2.3

        qortex visualize-overlay T1w.nii.gz aparc+aseg.nii.gz --type labelmap -o seg.html

        qortex visualize-overlay T1w.nii.gz pet_suv.nii.gz --type pet --colormap hot
    """
    try:
        from qortex.visualize import overlay_mask, overlay_labelmap, overlay_stat, overlay_pet
        from qortex.visualize._dispatch import inspect_file
    except ImportError as exc:
        typer.echo(f"Visualization dependencies not installed: {exc}\nInstall: pip install qortex[visual]", err=True)
        raise typer.Exit(1)

    for p in (base, overlay_path):
        if not p.exists():
            typer.echo(f"File not found: {p}", err=True)
            raise typer.Exit(1)

    # Auto-detect overlay type from filename when type="auto"
    if overlay_type == "auto":
        name = overlay_path.name.lower()
        if any(k in name for k in ("mask",)):
            overlay_type = "mask"
        elif any(k in name for k in ("zmap", "tmap", "stat", "contrast", "t-map", "z-map")):
            overlay_type = "stat"
        elif any(k in name for k in ("seg", "dseg", "label", "aparc", "aseg", "atlas", "parc")):
            overlay_type = "labelmap"
        elif any(k in name for k in ("pet", "suv", "fdg")):
            overlay_type = "pet"
        else:
            # Fall back to inspecting via qortex.visualize.inspect
            try:
                ov_asset = inspect_file(overlay_path)
                intent = ov_asset.intent
                if "mask" in intent:
                    overlay_type = "mask"
                elif "label" in intent or "seg" in intent:
                    overlay_type = "labelmap"
                elif "stat" in intent:
                    overlay_type = "stat"
                elif "pet" in intent:
                    overlay_type = "pet"
                else:
                    overlay_type = "mask"  # safe default
            except Exception:
                overlay_type = "mask"

    typer.echo(f"Overlay type: {overlay_type}")
    typer.echo(f"Base:    {base}")
    typer.echo(f"Overlay: {overlay_path}")

    kwargs: dict = {"resample": resample}
    try:
        if overlay_type == "mask":
            result = overlay_mask(base, overlay_path, alpha=alpha, **kwargs)
        elif overlay_type == "labelmap":
            result = overlay_labelmap(base, overlay_path, alpha=alpha, **kwargs)
        elif overlay_type == "stat":
            kw = {**kwargs, "threshold": threshold, "alpha": alpha}
            if colormap:
                kw["colormap"] = colormap
            result = overlay_stat(base, overlay_path, **kw)
        elif overlay_type == "pet":
            kw = {**kwargs, "alpha": alpha}
            if colormap:
                kw["colormap"] = colormap
            result = overlay_pet(base, overlay_path, **kw)
        else:
            typer.echo(f"Unknown overlay type: {overlay_type!r}. Choose: mask|stat|labelmap|pet", err=True)
            raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Overlay rendering failed: {exc}", err=True)
        raise typer.Exit(1)

    if output:
        result.to_html(output)
        typer.echo(f"Overlay HTML saved to {output}")
    elif open_browser:
        result.show()
        typer.echo("Opened in browser.")
    else:
        n = len(result.html) if result.html else 0
        typer.echo(f"Overlay rendered ({n:,} chars). Use --output FILE.html or --open to view.")


# ── visual-audit ──────────────────────────────────────────────────────────────

@app.command("visual-audit")
def visual_audit_cmd(
    dataset_id: str = typer.Argument(..., help="OpenNeuro dataset ID (e.g. ds000001)"),
    local_path: Optional[Path] = typer.Option(None, "--local", "-l",
                                               help="Root of the locally downloaded dataset"),
    output_dir: Path = typer.Option(Path("visual_audit"), "--output-dir", "-o",
                                    help="Directory for the HTML report"),
    json_output: bool = typer.Option(False, "--json", help="Also write visual_audit.json"),
    markdown_output: bool = typer.Option(False, "--markdown", help="Also write visual_audit.md"),
    manifest_json: bool = typer.Option(False, "--manifest-json", help="Also write visual_manifest.json"),
    subjects: Optional[str] = typer.Option(None, "--subjects", "-s",
                                           help="Comma-separated subject IDs (e.g. 01,02,04)"),
    suffixes: Optional[str] = typer.Option(None, "--suffixes",
                                           help="Comma-separated BIDS suffixes (e.g. T1w,bold)"),
    datatypes: Optional[str] = typer.Option(None, "--datatypes",
                                            help="Comma-separated BIDS datatypes (e.g. anat,func)"),
    max_files: int = typer.Option(24, "--max-files"),
    n_per_suffix: int = typer.Option(3, "--n-per-suffix"),
    open_browser: bool = typer.Option(False, "--open", is_flag=True),
) -> None:
    """Run a visual QC audit on a locally-downloaded OpenNeuro dataset.

    Reads exactly one center slice per NIfTI — the full volume is never loaded.

    \\b
    Examples:
      qortex visual-audit ds000001 --local data/ds000001 --output-dir qc/
      qortex visual-audit ds000001 -l data/ds000001 --suffixes T1w,bold --open
    """
    from qortex import Dataset

    sub_list = [s.strip() for s in subjects.split(",")] if subjects else None
    suf_list = [s.strip() for s in suffixes.split(",")] if suffixes else None
    dt_list = [s.strip() for s in datatypes.split(",")] if datatypes else None

    ds = Dataset(dataset_id)
    try:
        report = ds.visual_audit(
            output_dir=output_dir,
            subjects=sub_list,
            suffixes=suf_list,
            datatypes=dt_list,
            local_path=local_path,
            max_files=max_files,
            n_per_suffix=n_per_suffix,
            open_browser=open_browser,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    out_html = Path(output_dir) / "visual_audit.html"
    typer.echo(report.summary())
    typer.echo(f"\nReport written to {out_html}")
    if json_output:
        out_json = Path(output_dir) / "visual_audit.json"
        out_json.write_text(report.to_json(), encoding="utf-8")
        typer.echo(f"JSON written to {out_json}")
    if markdown_output:
        out_md = Path(output_dir) / "visual_audit.md"
        report.to_markdown(out_md)
        typer.echo(f"Markdown written to {out_md}")
    if manifest_json:
        out_manifest = Path(output_dir) / "visual_manifest.json"
        report.visual_manifest_json(out_manifest)
        typer.echo(f"Visual manifest written to {out_manifest}")
    if not open_browser:
        typer.echo("Run with --open to view in your browser.")


# ── visualize-openneuro ────────────────────────────────────────────────────────

@app.command("visualize-openneuro")
def visualize_openneuro_cmd(
    dataset_id: str = typer.Argument(..., help="OpenNeuro dataset ID (e.g. ds000001)"),
    subject: Optional[str] = typer.Option(None, "--subject", "-s",
                                          help="Subject ID without sub- prefix"),
    suffix: Optional[str] = typer.Option(None, "--suffix",
                                         help="BIDS suffix (T1w, bold, dwi, ...)"),
    datatype: Optional[str] = typer.Option(None, "--datatype", "-d",
                                           help="BIDS datatype folder (anat, func, ...)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                          help="Output HTML path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir",
                                              help="Download destination (defaults to cache)"),
    mode: str = typer.Option("auto", "--mode", "-m",
                             help="Rendering mode: auto|qc|thumbnail|interactive|static"),
    open_browser: bool = typer.Option(False, "--open", is_flag=True),
    max_size_mb: float = typer.Option(500.0, "--max-size-mb",
                                      help="Skip files larger than this (MB)"),
    n_per_suffix: int = typer.Option(1, "--n-per-suffix"),
) -> None:
    """Download a single file from OpenNeuro and visualize it.

    Fetches only the matching file (not the full dataset), then renders
    a viewer in the browser or writes HTML output.

    \\b
    Examples:
      qortex visualize-openneuro ds000001 --subject 01 --suffix T1w --open
      qortex visualize-openneuro ds004130 --datatype func --suffix bold -o bold.html
    """
    from qortex import Dataset
    from qortex.visualize._audit import run_visual_audit, select_visual_file_records

    sub_list = [subject] if subject else None
    suf_list = [suffix] if suffix else None
    dt_list = [datatype] if datatype else None

    ds = Dataset(dataset_id, data_dir=output_dir)

    typer.echo(f"Fetching manifest for {dataset_id}…")
    try:
        manifest = ds.manifest()
    except Exception as exc:
        typer.echo(f"Could not fetch manifest: {exc}", err=True)
        raise typer.Exit(1)

    selected = select_visual_file_records(
        manifest,
        subjects=sub_list,
        suffixes=suf_list,
        datatypes=dt_list,
        max_size_mb=max_size_mb,
        n_per_suffix=n_per_suffix,
    )

    if not selected:
        typer.echo("No matching files found in the dataset manifest.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Downloading {len(selected)} file(s)…")
    try:
        ds.download_paths([fr.path for fr in selected])
        local_root = ds._resolve_data_dir()
    except Exception as exc:
        typer.echo(f"Download failed: {exc}", err=True)
        raise typer.Exit(1)

    report = run_visual_audit(
        dataset_id=dataset_id,
        file_records=selected,
        local_root=local_root,
        max_files=n_per_suffix,
    )

    if len(report.entries) == 1:
        asset = report.entries[0].asset
        try:
            result = _render_modality_specific(asset, mode=mode)
            if output:
                _write_visual_output(result, output)
                typer.echo(f"Visualization saved to {output}")
            elif open_browser:
                _show_visual_output(result)
            else:
                typer.echo(
                    f"Rendered {asset.intent} [{asset.modality}]. "
                    f"Use --output FILE.html or --open to view."
                )
            return
        except Exception as exc:
            typer.echo(f"Render failed: {exc}", err=True)

    # Multiple files or fallback — show audit report
    out_path = output or Path(f"{dataset_id}_preview.html")
    report.to_html(out_path)
    typer.echo(report.summary())
    typer.echo(f"\nPreview written to {out_path}")
    if open_browser:
        report.show()
    else:
        typer.echo("Run with --open to view in your browser.")


# ── dashboard ─────────────────────────────────────────────────────────────────

@app.command()
def dashboard(
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8501, "--port"),
) -> None:
    """Launch the Qortex Streamlit dashboard."""
    import subprocess
    import importlib.util

    spec = importlib.util.find_spec("qortex.console.app")
    if spec is None or spec.origin is None:
        typer.echo("Dashboard not available. Install with: pip install qortex[dashboard]", err=True)
        raise typer.Exit(1)

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", spec.origin,
            "--server.address", host,
            "--server.port", str(port),
        ]
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dataset(dataset_id: str, snapshot: str | None):
    from qortex import Dataset

    return Dataset(dataset_id, snapshot=snapshot)


def _write_model_json(model, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"\nJSON report saved to {path}")


def _render_modality_specific(asset, *, mode: str):
    """Render one visual asset with modality-specific QC when available."""
    from qortex.visualize._asset import INTENT_BOLD, INTENT_DWI, INTENT_RAW_SIGNAL
    from qortex.visualize._dispatch import render_asset

    if asset.intent == INTENT_BOLD and asset.family == "nifti":
        from qortex.visualize.fmri import fmri_summary
        return fmri_summary(asset.path, title=f"fMRI QC — {asset.path.name}")
    if asset.intent == INTENT_DWI and asset.family == "nifti":
        from qortex.visualize.dwi import dwi_summary
        return dwi_summary(asset.path, title=f"DWI QC — {asset.path.name}")
    if asset.intent == INTENT_RAW_SIGNAL or asset.family == "eeg":
        from qortex.visualize.timeseries import TimeSeriesViewer
        return TimeSeriesViewer(asset.path).dashboard(title=f"Signal QC — {asset.path.name}")
    return render_asset(asset, mode=mode)


def _write_visual_output(rendered, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(rendered, "to_html") and not hasattr(rendered, "data"):
        rendered.to_html(output)
        return
    if isinstance(rendered, str):
        output.write_text(rendered, encoding="utf-8")
        return
    import plotly.io as pio
    output.write_text(pio.to_html(rendered, include_plotlyjs="cdn", full_html=True), encoding="utf-8")


def _show_visual_output(rendered) -> None:
    if hasattr(rendered, "show"):
        rendered.show()
        return
    import tempfile
    import webbrowser
    if isinstance(rendered, str):
        html = rendered
    else:
        import plotly.io as pio
        html = pio.to_html(rendered, include_plotlyjs="cdn", full_html=True)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as fh:
        fh.write(html)
        webbrowser.open(f"file://{fh.name}")


def _profile_text(profile: dict[str, Any]) -> str:
    lines = [
        f"Dataset    : {profile.get('dataset_id')}",
        f"Name       : {profile.get('name')}",
        f"Snapshot   : {profile.get('snapshot')}",
        f"Subjects   : {profile.get('n_subjects')}",
        f"Sessions   : {profile.get('n_sessions')}",
        f"Files      : {profile.get('n_files')}",
        f"Size       : {(profile.get('total_bytes') or 0) / 1e9:.2f} GB",
        f"License    : {profile.get('license')}",
        f"DOI        : {profile.get('doi')}",
        f"Modalities : {', '.join(profile.get('modalities') or [])}",
        f"Tasks      : {', '.join(profile.get('tasks') or [])}",
        f"Events     : {profile.get('has_events')} ({profile.get('n_event_files')} files)",
        f"Derivatives: {profile.get('has_derivatives')} ({profile.get('n_derivative_files')} files)",
    ]
    summaries = profile.get("file_summaries") or []
    if summaries:
        lines.append("File summary:")
        for row in summaries[:20]:
            lines.append(
                f"  {row.get('category')}={row.get('value')}  files={row.get('n_files')}  bytes={row.get('bytes') or 0}"
            )
    return "\n".join(lines)


def _fetch_files(client, dataset_id: str, snapshot: str | None):
    """Resolve the latest (or named) snapshot and return (snap_ref, raw_files)."""
    if snapshot:
        snap_ref = client.get_snapshot(dataset_id, snapshot)
    else:
        snap_ref = client.get_latest_snapshot(dataset_id)
    snap_ref, raw_files = client.get_files(dataset_id, snap_ref.tag)
    return snap_ref, raw_files


# ── neuroai subcommands ────────────────────────────────────────────────────────

neuroai_app = typer.Typer(
    name="neuroai",
    help="NeuroAI runtime — declarative source→model→output pipelines.",
    no_args_is_help=True,
)
app.add_typer(neuroai_app, name="neuroai")


@neuroai_app.command("check")
def neuroai_check(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full transform list"),
) -> None:
    """Check source-model compatibility without loading weights.

    Probes the source and model, verifies modality/channel/rate/shape/dtype
    compatibility, and lists required preprocessing transforms and any blockers.

    Exit code 0 = compatible (possibly with transforms).
    Exit code 1 = incompatible or uncertain.
    """
    try:
        from qortex.neuroai import Pipeline
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    try:
        pipe = Pipeline.from_yaml(pipeline)
        report = pipe.check()
    except Exception as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(report.summary())

    if verbose and pipe.preprocess_plan:
        typer.echo("\n" + pipe.preprocess_plan.summary())

    status = report.status.value if hasattr(report.status, "value") else str(report.status)
    if status == "incompatible":
        raise typer.Exit(1)
    elif status == "uncertain":
        raise typer.Exit(2)


@neuroai_app.command("run")
def neuroai_run(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check only; do not execute"),
) -> None:
    """Run a NeuroAI pipeline: source → preprocess → model → outputs.

    Automatically checks compatibility before loading model weights.
    Writes provenance sidecar alongside each output file.
    """
    try:
        from qortex.neuroai import Pipeline
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    try:
        pipe = Pipeline.from_yaml(pipeline)
        report = pipe.check()
    except Exception as exc:
        typer.echo(f"[ERROR] check failed: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(report.summary())

    status = report.status.value if hasattr(report.status, "value") else str(report.status)
    if status == "incompatible":
        typer.echo("[BLOCKED] Pipeline is incompatible. Fix blockers before running.", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo("[DRY RUN] Check passed. Use without --dry-run to execute.")
        return

    typer.echo("Running pipeline…")
    try:
        run_report = pipe.run()
    except Exception as exc:
        typer.echo(f"[ERROR] run failed: {exc}", err=True)
        raise typer.Exit(1)

    if run_report.latency_report:
        typer.echo("\n" + run_report.latency_report.summary())

    if run_report.errors:
        typer.echo(f"\nErrors ({len(run_report.errors)}):")
        for err in run_report.errors[:10]:
            typer.echo(f"  • {err}")

    exit_code = 0 if run_report.success else 1
    raise typer.Exit(exit_code)


@neuroai_app.command("benchmark")
def neuroai_benchmark(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML"),
    n_windows: int = typer.Option(20, "--windows", "-n", help="Number of windows to time"),
) -> None:
    """Benchmark pipeline latency without writing real outputs.

    Loads the model and runs N windows through source → preprocess → inference,
    reporting p50/p95/p99 latency and whether the latency budget is met.
    """
    try:
        from qortex.neuroai import Pipeline
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    try:
        pipe = Pipeline.from_yaml(pipeline)
        typer.echo(f"Benchmarking {n_windows} windows…")
        report = pipe.benchmark(n_windows=n_windows)
    except Exception as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(report.summary())

    if report.status == "FAIL":
        raise typer.Exit(1)


@neuroai_app.command("replay")
def neuroai_replay(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML"),
    source: Path = typer.Argument(..., help="Recorded session file (XDF, EDF, …)"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override output directory"),
) -> None:
    """Replay a recorded session through the pipeline.

    Useful for debugging closed-loop workflows without live hardware.
    Same preprocessing and model as the live pipeline.
    """
    try:
        from qortex.neuroai import Pipeline
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    try:
        pipe = Pipeline.from_yaml(pipeline)
        pipe.check()
        typer.echo(f"Replaying {source.name} at {speed}× speed…")
        report = pipe.replay(source, speed=speed, output_dir=output_dir)
    except Exception as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    if report.latency_report:
        typer.echo(report.latency_report.summary())

    exit_code = 0 if report.success else 1
    raise typer.Exit(exit_code)


@neuroai_app.command("inspect-source")
def neuroai_inspect_source(
    source: Path = typer.Argument(..., help="Local file or BIDS directory to inspect"),
    modality: Optional[str] = typer.Option(None, "--modality", "-m"),
    suffix: Optional[str] = typer.Option(None, "--suffix"),
) -> None:
    """Probe a local data source and print its SourceProfile.

    No model required.  Useful for verifying what channels, sampling rate,
    shape, and coordinate frame a source exposes before building a pipeline.
    """
    try:
        from qortex.neuroai.sources._registry import make_source_adapter
        from qortex.neuroai.spec import SourceSpec
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    src_type = "bids" if source.is_dir() else "local_file"
    spec = SourceSpec(type=src_type, path=str(source), modality=modality, suffix=suffix)
    try:
        adapter = make_source_adapter(spec)
        profile = adapter.probe()
    except Exception as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    lines = [
        f"Source      : {profile.source_id}",
        f"Type        : {profile.source_type}",
        f"Modality    : {profile.modality}",
        f"Abstraction : {profile.abstraction}",
    ]
    if profile.n_channels is not None:
        lines.append(f"Channels    : {profile.n_channels}")
    if profile.sampling_rate_hz is not None:
        lines.append(f"Sampling Hz : {profile.sampling_rate_hz:.1f}")
    if profile.duration_s is not None:
        lines.append(f"Duration    : {profile.duration_s:.1f}s")
    if profile.spatial_shape:
        lines.append(f"Shape       : {profile.spatial_shape}")
    if profile.voxel_sizes_mm:
        sizes = "×".join(f"{v:.2f}" for v in profile.voxel_sizes_mm)
        lines.append(f"Voxel (mm)  : {sizes}")
    if profile.n_subjects:
        lines.append(f"Subjects    : {profile.n_subjects}")
    lines.append(f"Evidence    : {profile.evidence_status}")
    if profile.warnings:
        lines.append(f"Warnings    : {len(profile.warnings)}")
        for w in profile.warnings:
            lines.append(f"  ⚠ {w.message}")
    typer.echo("\n".join(lines))


@neuroai_app.command("inspect-model")
def neuroai_inspect_model(
    model_id: str = typer.Argument(..., help="Model ID or path (e.g. hf://org/model or model.onnx)"),
    provider: str = typer.Option("huggingface", "--provider", "-p"),
    task: Optional[str] = typer.Option(None, "--task"),
) -> None:
    """Inspect a model and print its ModelProfile + InputContract.

    Does not load model weights.  Shows expected channels, sampling rate,
    input shape, dtype, and output schema.
    """
    try:
        from qortex.neuroai.models._registry import make_model_adapter
        from qortex.neuroai.spec import ModelSpec
    except ImportError as e:
        typer.echo(f"NeuroAI runtime requires additional dependencies: {e}", err=True)
        raise typer.Exit(1)

    # Handle hf:// prefix
    model_path = model_id.removeprefix("hf://")
    spec = ModelSpec(provider=provider, id=model_path, task=task)
    try:
        adapter = make_model_adapter(spec)
        profile = adapter.inspect()
    except Exception as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)

    lines = [
        f"Model       : {profile.model_id}",
        f"Provider    : {profile.provider}",
        f"Task        : {profile.task}",
        f"License     : {profile.license}",
        f"Revision    : {profile.revision}",
    ]
    if profile.input_contract:
        ic = profile.input_contract
        lines.append(f"Input modality  : {ic.modality}")
        lines.append(f"Input axes      : {ic.axis_convention}")
        if ic.n_channels:
            lines.append(f"Required channels: {ic.n_channels}")
        if ic.sampling_rate_hz:
            lines.append(f"Required Fs Hz  : {ic.sampling_rate_hz}")
        if ic.spatial_shape:
            lines.append(f"Required shape  : {ic.spatial_shape}")
        lines.append(f"Input dtype     : {ic.dtype}")
        lines.append(f"Evidence        : {ic.evidence_status}")
    if profile.output_contract:
        oc = profile.output_contract
        lines.append(f"Output type     : {oc.output_type}")
        if oc.classes:
            lines.append(f"Classes ({oc.n_classes}): {', '.join(oc.classes[:10])}")
    if profile.warnings:
        for w in profile.warnings:
            lines.append(f"  ⚠ [{w.severity}] {w.message}")
    typer.echo("\n".join(lines))
