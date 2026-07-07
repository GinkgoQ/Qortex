"""Catalog ingestion from OpenNeuro into the local Qortex database."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.catalog.index import CatalogIndex, summarize_manifest_files
from qortex.client.graphql import OpenNeuroClient
from qortex.core.config import get_config


_Q_ALL_DATASETS = """
query AllDatasets($after: String, $first: Int!) {
  datasets(first: $first, after: $after, orderBy: { created: descending }) {
    edges {
      node {
        id
        metadata {
          datasetName
          modalities
          tasksCompleted
        }
        latestSnapshot {
          id
          tag
          created
          size
          description {
            Name
            Authors
            License
            DatasetDOI
            BIDSVersion
            HowToAcknowledge
            Funding
            ReferencesAndLinks
            EthicsApprovals
          }
          summary {
            subjects
            sessions
            totalFiles
            size
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


def _offset_cursor(offset: int) -> str | None:
    """Build OpenNeuro's base64 ``{"offset": N}`` cursor for page N. Returns
    None for offset 0 (the first page is requested with no ``after``). This is
    what makes concurrent paging possible: every page's cursor is known from
    its index alone, with no need to chain through prior ``endCursor`` values."""
    if offset <= 0:
        return None
    import base64
    import json as _json

    return base64.b64encode(_json.dumps({"offset": int(offset)}).encode()).decode()


def _advance_cursor(cursor: str, page_size: int) -> str | None:
    """OpenNeuro's dataset cursor is base64 ``{"offset": N}``. Bump the offset
    by one page so a hard-failed page can be skipped rather than retried
    forever. Returns None if the cursor isn't the expected offset shape."""
    import base64
    import json as _json

    try:
        payload = _json.loads(base64.b64decode(cursor).decode())
        payload["offset"] = int(payload.get("offset", 0)) + page_size
        return base64.b64encode(_json.dumps(payload).encode()).decode()
    except Exception:  # noqa: BLE001 - unknown cursor shape, give up cleanly
        return None


def _fetch_page(client, cursor, page_size):
    """Fetch one page of dataset nodes and return (rows, page_info).

    ``partial_ok``: OpenNeuro has a handful of datasets with a broken
    latestSnapshot that return a field-level "Not Found". Without tolerating
    that the whole sweep aborts on the first such page (~offset 250) — the
    exact reason the catalog was stuck near 300 instead of the full ~1.8k.
    Null edges/nodes (the broken datasets) are simply skipped."""
    variables: dict[str, Any] = {"first": page_size}
    if cursor:
        variables["after"] = cursor
    data = client._query(_Q_ALL_DATASETS, variables=variables, partial_ok=True)
    page_data = data.get("datasets") or {}
    rows = [
        normalize_dataset_node(node)
        for edge in (page_data.get("edges") or [])
        if (node := (edge or {}).get("node"))
    ]
    return rows, (page_data.get("pageInfo") or {})


def refresh(
    catalog_path: Path | None = None,
    max_pages: int = 40,
    progress: bool = True,
    *,
    page_size: int = 50,
    include_file_summary: bool = False,
    file_summary_limit: int | None = None,
    workers: int = 8,
    on_progress=None,
) -> int:
    """Fetch dataset metadata from OpenNeuro and populate the local catalog.

    Count-first, then fast: the total is fetched up front (one cheap query),
    which both gives an honest progress denominator and lets the metadata
    sweep pre-compute every page's offset cursor and fetch them CONCURRENTLY
    (``OpenNeuroClient`` is thread-safe). Writes stay on this thread — DuckDB's
    single connection serializes them, and the set-based bulk upsert makes that
    cheap. Deep file-summary ingestion (``include_file_summary``) stays on the
    sequential cursor-chained path, one dataset at a time.

    Parameters
    ----------
    workers:
        Concurrent page fetchers for the metadata sweep. 1 forces the
        sequential path. Ignored when ``include_file_summary`` is set.
    on_progress:
        Optional ``callable(done, total)`` invoked as datasets land — wires
        straight into the job system's progress hook.
    include_file_summary / file_summary_limit:
        As before — deep, per-dataset file-content facets (slower).
    """
    cfg = get_config()
    if catalog_path is None:
        catalog_path = cfg.cache_dir / "catalog" / "catalog.duckdb"

    index = CatalogIndex(catalog_path)
    client = OpenNeuroClient()
    try:
        total = 0
        try:
            total = client.count_datasets()
        except Exception:  # noqa: BLE001 - count is a nicety; sweep still works
            total = 0
        if progress and total:
            print(f"OpenNeuro reports {total} datasets — indexing…")

        if total and workers and workers > 1 and not include_file_summary:
            return _refresh_concurrent(
                client, index, total, page_size, max_pages, workers, progress, on_progress
            )
        return _refresh_sequential(
            client, index, total, page_size, max_pages,
            include_file_summary, file_summary_limit, progress, on_progress,
        )
    finally:
        client.close()
        index.close()


def _refresh_concurrent(client, index, total, page_size, max_pages, workers, progress, on_progress) -> int:
    """Known count → independent offset cursors → parallel fetch, serial write."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_pages = min(max_pages, (total + page_size - 1) // page_size)
    offsets = [k * page_size for k in range(n_pages)]
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_page, client, _offset_cursor(off), page_size): off for off in offsets}
        for future in as_completed(futures):
            off = futures[future]
            try:
                rows, _ = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad page never sinks the run
                if progress:
                    print(f"  offset {off}: skipped ({type(exc).__name__}: {exc})")
                continue
            index.upsert_many(rows)  # main thread: DuckDB writes stay serialized
            count += len(rows)
            if on_progress:
                on_progress(count, total)
            if progress:
                print(f"  +{len(rows)} (offset {off}) → {count}/{total}")
    return count


def _refresh_sequential(client, index, total, page_size, max_pages,
                        include_file_summary, file_summary_limit, progress, on_progress) -> int:
    """Cursor-chained fallback (and the deep-profile path). Resilient: one bad
    page is skipped by advancing the offset rather than aborting the sweep."""
    count = 0
    deep_count = 0
    cursor: str | None = None
    for page in range(max_pages):
        try:
            rows, page_info = _fetch_page(client, cursor, page_size)
        except Exception as exc:  # noqa: BLE001 - keep sweeping past a bad page
            if progress:
                print(f"  Page {page + 1}: skipped ({type(exc).__name__}: {exc})")
            if cursor is None:
                break
            next_cursor = _advance_cursor(cursor, page_size)
            if next_cursor is None:
                break
            cursor = next_cursor
            continue

        if include_file_summary:
            for row in rows:
                if file_summary_limit is not None and deep_count >= file_summary_limit:
                    break
                row.update(_fetch_file_summary(client, row["dataset_id"], row.get("snapshot")))
                deep_count += 1

        index.upsert_many(rows)
        count += len(rows)
        if on_progress and total:
            on_progress(count, total)
        if progress:
            suffix = f", deep summaries: {deep_count}" if include_file_summary else ""
            print(f"  Page {page + 1}: {len(rows)} datasets (total: {count}{suffix})")

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
    return count

    return count


def refresh_dataset(
    dataset_id: str,
    *,
    catalog_path: Path | None = None,
    snapshot: str | None = None,
    include_file_summary: bool = True,
) -> dict[str, Any]:
    """Fetch and index one dataset profile, optionally with file summaries."""
    cfg = get_config()
    if catalog_path is None:
        catalog_path = cfg.cache_dir / "catalog" / "catalog.duckdb"

    client = OpenNeuroClient()
    index = CatalogIndex(catalog_path)
    try:
        ref = client.get_dataset(dataset_id)
        snap = client.get_snapshot(dataset_id, snapshot) if snapshot else client.get_latest_snapshot(dataset_id)
        row = {
            "dataset_id": ref.id,
            "name": ref.name,
            "description": ref.description,
            "authors": ref.authors,
            "doi": ref.doi,
            "license": ref.license,
            "modalities": ref.modalities,
            "tasks": ref.tasks,
            "snapshot": snap.tag,
            "snapshot_created": snap.created,
            "total_bytes": snap.size,
            "raw_metadata": ref.raw_metadata,
            "raw_description": {},
            "updated_at": _now(),
        }
        if include_file_summary:
            row.update(_fetch_file_summary(client, dataset_id, snap.tag))
        index.upsert(row)
        profile = index.profile(dataset_id)
        if profile is None:
            raise RuntimeError(f"Failed to index dataset {dataset_id}")
        return profile
    finally:
        client.close()
        index.close()


def normalize_dataset_node(node: dict[str, Any]) -> dict[str, Any]:
    """Normalize one OpenNeuro dataset GraphQL node into a catalog row."""
    meta = node.get("metadata") or {}
    snapshot = node.get("latestSnapshot") or {}
    desc = snapshot.get("description") or {}
    summary = snapshot.get("summary") or {}
    sessions = summary.get("sessions")
    tasks = meta.get("tasksCompleted") or []
    modalities = meta.get("modalities") or []
    authors = desc.get("Authors") or []
    name = meta.get("datasetName") or desc.get("Name")
    description = _description_text(desc)
    return {
        "dataset_id": node.get("id", ""),
        "name": name,
        "description": description or name,
        "authors": authors,
        "doi": _normalise_doi(desc.get("DatasetDOI")),
        "license": desc.get("License"),
        "n_subjects": summary.get("subjects"),
        "n_sessions": len(sessions or []),
        "n_tasks": len(tasks),
        "modalities": modalities,
        "tasks": tasks,
        "keywords": _metadata_keywords(desc),
        "snapshot": snapshot.get("tag"),
        "snapshot_created": snapshot.get("created"),
        "n_files": summary.get("totalFiles"),
        "total_bytes": summary.get("size") or snapshot.get("size"),
        "raw_metadata": meta,
        "raw_description": desc,
        "updated_at": _now(),
    }


def _fetch_file_summary(
    client: OpenNeuroClient,
    dataset_id: str,
    snapshot: str | None,
) -> dict[str, Any]:
    try:
        snap_ref, files = client.get_files(dataset_id, snapshot)
    except Exception as exc:
        return {
            "file_summaries": [
                {
                    "category": "ingest_error",
                    "value": exc.__class__.__name__,
                    "n_files": 1,
                    "bytes": 0,
                }
            ]
        }
    summary = summarize_manifest_files(files)
    try:
        from qortex.manifest.builder import ManifestBuilder

        manifest = ManifestBuilder().build(dataset_id, snap_ref, files)
        summary.update(
            {
                "snapshot": manifest.snapshot,
                "doi": manifest.doi,
                "n_subjects": manifest.summary.n_subjects,
                "n_sessions": len(manifest.summary.sessions),
                "n_tasks": len(manifest.summary.tasks),
                "modalities": manifest.summary.modalities,
                "tasks": manifest.summary.tasks,
                "has_events": manifest.summary.has_events,
                "has_derivatives": manifest.summary.has_derivatives,
            }
        )
    except Exception:
        pass
    summary["n_files"] = len([file for file in files if not file.get("directory")])
    summary["total_bytes"] = sum(int(file.get("size") or 0) for file in files if not file.get("directory"))
    return summary


def _description_text(desc: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for key in [
        "Name",
        "BIDSVersion",
        "HowToAcknowledge",
        "Funding",
        "ReferencesAndLinks",
        "EthicsApprovals",
    ]:
        value = desc.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        else:
            parts.append(str(value))
    return "\n".join(parts) if parts else None


def _metadata_keywords(desc: dict[str, Any]) -> list[str]:
    keywords: set[str] = set()
    for key in ["BIDSVersion", "License", "DatasetDOI"]:
        value = desc.get(key)
        if value:
            keywords.add(str(value))
    for key in ["Funding", "ReferencesAndLinks", "EthicsApprovals"]:
        value = desc.get(key)
        if isinstance(value, list):
            keywords.update(str(item) for item in value if item)
        elif value:
            keywords.add(str(value))
    return sorted(keywords)


def _normalise_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    return doi.removeprefix("doi:")


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
