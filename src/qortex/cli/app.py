"""Qortex CLI — typer application with all subcommands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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
    min_subjects: Optional[int] = typer.Option(None, "--min-subjects", "-n"),
    limit: int = typer.Option(20, "--limit", "-l"),
    refresh_catalog: bool = typer.Option(False, "--refresh", help="Refresh catalog before searching"),
) -> None:
    """Search the local OpenNeuro catalog."""
    if refresh_catalog:
        from qortex.catalog.refresh import refresh as do_refresh
        typer.echo("Refreshing catalog from OpenNeuro …")
        n = do_refresh(progress=False)
        typer.echo(f"  {n} datasets indexed.")

    from qortex.catalog.search import search as do_search
    results = do_search(
        query=query,
        modality=modality,
        min_subjects=min_subjects,
        limit=limit,
    )
    if not results:
        typer.echo("No results found.")
        raise typer.Exit(0)

    for r in results:
        mods = ", ".join(r.get("modalities") or [])
        typer.echo(
            f"[{r['dataset_id']}]  {r.get('name') or '(no name)'}  "
            f"  subjects={r.get('n_subjects')}  modalities={mods}"
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
) -> None:
    """Refresh the local dataset catalog from OpenNeuro."""
    from qortex.catalog.refresh import refresh as do_refresh
    typer.echo("Refreshing catalog …")
    n = do_refresh(max_pages=max_pages, progress=True)
    typer.echo(f"Done. {n} datasets in catalog.")


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

def _fetch_files(client, dataset_id: str, snapshot: str | None):
    """Resolve the latest (or named) snapshot and return (snap_ref, raw_files)."""
    if snapshot:
        snap_ref = client.get_snapshot(dataset_id, snapshot)
    else:
        snap_ref = client.get_latest_snapshot(dataset_id)
    snap_ref, raw_files = client.get_files(dataset_id, snap_ref.tag)
    return snap_ref, raw_files
