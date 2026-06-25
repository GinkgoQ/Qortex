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


def refresh(
    catalog_path: Path | None = None,
    max_pages: int = 40,
    progress: bool = True,
    *,
    page_size: int = 50,
    include_file_summary: bool = False,
    file_summary_limit: int | None = None,
) -> int:
    """Fetch dataset metadata from OpenNeuro and populate the local catalog.

    Parameters
    ----------
    include_file_summary:
        When true, fetches the recursive file manifest for each indexed dataset
        and digests extensions, datatypes, suffixes, event files, derivative
        files, and metadata/primary counts into the catalog. This is more
        expensive than metadata-only ingestion, but it gives the search layer
        real file-content facets.
    file_summary_limit:
        Optional cap on how many datasets receive deep file-summary ingestion
        during this refresh call.
    """
    cfg = get_config()
    if catalog_path is None:
        catalog_path = cfg.cache_dir / "catalog" / "catalog.duckdb"

    index = CatalogIndex(catalog_path)
    client = OpenNeuroClient()
    count = 0
    deep_count = 0

    try:
        cursor: str | None = None
        for page in range(max_pages):
            variables: dict[str, Any] = {"first": page_size}
            if cursor:
                variables["after"] = cursor
            data = client._query(_Q_ALL_DATASETS, variables=variables)

            page_data = data.get("datasets", {})
            edges = page_data.get("edges", [])
            page_info = page_data.get("pageInfo", {})

            rows = []
            for edge in edges:
                node = edge.get("node", {})
                row = normalize_dataset_node(node)
                if include_file_summary and (
                    file_summary_limit is None or deep_count < file_summary_limit
                ):
                    row.update(_fetch_file_summary(client, row["dataset_id"], row.get("snapshot")))
                    deep_count += 1
                rows.append(row)

            index.upsert_many(rows)
            count += len(rows)

            if progress:
                suffix = f", deep summaries: {deep_count}" if include_file_summary else ""
                print(f"  Page {page + 1}: {len(rows)} datasets (total: {count}{suffix})")

            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
    finally:
        client.close()
        index.close()

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
