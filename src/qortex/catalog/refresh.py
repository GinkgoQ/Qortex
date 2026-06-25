"""Catalog refresh — pull dataset metadata from OpenNeuro and update the index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from qortex.catalog.index import CatalogIndex
from qortex.client.graphql import OpenNeuroClient
from qortex.core.config import get_config


_Q_ALL_DATASETS = """
query AllDatasets($after: String) {
  datasets(first: 25, after: $after, orderBy: { created: descending }) {
    edges {
      node {
        id
        metadata {
          datasetName
          modalities
          tasksCompleted
        }
        latestSnapshot {
          tag
          description {
            Name
            Authors
            License
            DatasetDOI
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
) -> int:
    """Fetch dataset metadata from OpenNeuro and populate the local catalog.

    Returns the number of datasets inserted/updated.
    """
    cfg = get_config()
    if catalog_path is None:
        catalog_path = cfg.cache_dir / "catalog" / "catalog.duckdb"

    index = CatalogIndex(catalog_path)
    client = OpenNeuroClient()
    count = 0

    cursor: str | None = None
    for page in range(max_pages):
        variables: dict = {"after": cursor} if cursor else {}
        data = client._query(_Q_ALL_DATASETS, variables=variables)

        page_data = data.get("datasets", {})
        edges = page_data.get("edges", [])
        page_info = page_data.get("pageInfo", {})

        rows = []
        for edge in edges:
            node = edge.get("node", {})
            meta = node.get("metadata") or {}
            snapshot = node.get("latestSnapshot") or {}
            desc = snapshot.get("description") or {}
            summary = snapshot.get("summary") or {}

            rows.append({
                "dataset_id": node.get("id", ""),
                "name": meta.get("datasetName") or desc.get("Name"),
                "authors": ", ".join(desc.get("Authors") or []),
                "description": desc.get("Name"),
                "doi": _normalise_doi(desc.get("DatasetDOI")),
                "license": desc.get("License"),
                "n_subjects": summary.get("subjects"),
                "n_sessions": len(summary.get("sessions") or []),
                "n_tasks": len(meta.get("tasksCompleted") or []),
                "modalities": meta.get("modalities") or [],
                "snapshot": snapshot.get("tag"),
                "n_files": summary.get("totalFiles"),
                "total_bytes": summary.get("size"),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            })

        index.upsert_many(rows)
        count += len(rows)

        if progress:
            print(f"  Page {page + 1}: {len(rows)} datasets (total so far: {count})")

        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")

    index.close()
    return count


def _normalise_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    return doi.removeprefix("doi:")
