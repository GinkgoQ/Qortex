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
    planner = DownloadPlanner()
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
