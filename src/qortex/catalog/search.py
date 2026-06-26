"""High-level search interface over the catalog index.

Provides two search paths:
  1. Local catalog (SQLite/DuckDB) — fast, structured, works offline.
     Accessible via DatasetQuery fluent builder or the search() convenience fn.

  2. Live API search — hits OpenNeuro GraphQL in real time, then optionally
     upserts results into the local catalog for future offline use.
     Accessible via live_search() or DatasetQuery.live().

Results are always returned as list[dict] for composability. Use
as_dataframe() to get a Polars DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.catalog.index import CatalogIndex
from qortex.core.config import get_config


def _default_catalog_path() -> Path:
    cfg = get_config()
    return cfg.cache_dir / "catalog" / "catalog.duckdb"


class PagedResults:
    """Container for paginated search results with metadata."""

    def __init__(
        self,
        results: list[dict[str, Any]],
        total: int,
        offset: int,
        limit: int,
    ) -> None:
        self.results = results
        self.total = total
        self.offset = offset
        self.limit = limit

    @property
    def has_more(self) -> bool:
        return (self.offset + len(self.results)) < self.total

    @property
    def next_offset(self) -> int | None:
        return self.offset + self.limit if self.has_more else None

    def as_dataframe(self):
        import polars as pl
        return pl.DataFrame(self.results) if self.results else pl.DataFrame()

    def __repr__(self) -> str:
        return (
            f"PagedResults(n={len(self.results)}, "
            f"total={self.total}, "
            f"offset={self.offset}/{self.limit}, "
            f"has_more={self.has_more})"
        )


class DatasetQuery:
    """Fluent builder for catalog searches.

    Usage::

        # Local catalog (fast, offline)
        results = (
            DatasetQuery()
            .modality("eeg")
            .min_subjects(20)
            .containing("auditory")
            .limit(10)
            .fetch()
        )

        # Paginated with total count
        page = DatasetQuery().modality("bold").fetch_page()
        print(f"Showing {len(page.results)} of {page.total}")

        # Live search from OpenNeuro API
        results = DatasetQuery().modality("eeg").live(sync_local=True)

        # Facet aggregations
        facets = DatasetQuery().facets()
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._catalog_path = catalog_path or _default_catalog_path()
        self._query: str | None = None
        self._modality: str | None = None
        self._task: str | None = None
        self._author: str | None = None
        self._license: str | None = None
        self._min_subjects: int | None = None
        self._max_size_gb: float | None = None
        self._has_events: bool | None = None
        self._has_derivatives: bool | None = None
        self._limit: int = 50
        self._offset: int = 0

    # ── Filter methods ────────────────────────────────────────────────────

    def containing(self, text: str) -> "DatasetQuery":
        self._query = text or None
        return self

    def modality(self, mod: str) -> "DatasetQuery":
        self._modality = mod or None
        return self

    def task(self, task: str) -> "DatasetQuery":
        self._task = task or None
        return self

    def author(self, author: str) -> "DatasetQuery":
        self._author = author or None
        return self

    def license(self, license_name: str) -> "DatasetQuery":
        self._license = license_name or None
        return self

    def min_subjects(self, n: int) -> "DatasetQuery":
        self._min_subjects = n if n is not None else None
        return self

    def max_size_gb(self, size_gb: float) -> "DatasetQuery":
        self._max_size_gb = size_gb if size_gb is not None else None
        return self

    def has_events(self, value: bool = True) -> "DatasetQuery":
        self._has_events = value
        return self

    def has_derivatives(self, value: bool = True) -> "DatasetQuery":
        self._has_derivatives = value
        return self

    def limit(self, n: int) -> "DatasetQuery":
        self._limit = n
        return self

    def offset(self, n: int) -> "DatasetQuery":
        self._offset = max(0, n)
        return self

    # ── Execution methods ─────────────────────────────────────────────────

    def fetch(self) -> list[dict[str, Any]]:
        """Execute against the local catalog and return matching rows."""
        index = CatalogIndex(self._catalog_path)
        try:
            return index.search(
                query=self._query,
                modality=self._modality,
                task=self._task,
                author=self._author,
                license=self._license,
                min_subjects=self._min_subjects,
                max_size_gb=self._max_size_gb,
                has_events=self._has_events,
                has_derivatives=self._has_derivatives,
                limit=self._limit,
                offset=self._offset,
            )
        finally:
            index.close()

    def fetch_page(self) -> PagedResults:
        """Execute against the local catalog and return a PagedResults.

        Includes total count so callers can paginate without re-running the
        full query.
        """
        index = CatalogIndex(self._catalog_path)
        try:
            # Get matching results (all of them up to a safety cap for counting)
            all_results = index.search(
                query=self._query,
                modality=self._modality,
                task=self._task,
                author=self._author,
                license=self._license,
                min_subjects=self._min_subjects,
                max_size_gb=self._max_size_gb,
                has_events=self._has_events,
                has_derivatives=self._has_derivatives,
                limit=100_000,  # fetch all matching to count accurately
                offset=0,
            )
            total = len(all_results)
            page = all_results[self._offset : self._offset + self._limit]
            return PagedResults(
                results=page,
                total=total,
                offset=self._offset,
                limit=self._limit,
            )
        finally:
            index.close()

    def count(self) -> int:
        """Return the total number of matching rows (not just the current page)."""
        return self.fetch_page().total

    def as_dataframe(self):
        """Return results as a Polars DataFrame."""
        import polars as pl
        rows = self.fetch()
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def facets(self, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Return discovery facets: modalities, tasks, licenses, keywords."""
        index = CatalogIndex(self._catalog_path)
        try:
            return index.facets(limit=limit)
        finally:
            index.close()

    def live(
        self,
        token: str | None = None,
        sync_local: bool = True,
    ) -> list[dict[str, Any]]:
        """Search OpenNeuro API in real-time and optionally sync results locally.

        Parameters
        ----------
        token:
            Optional API token for private dataset access.
        sync_local:
            When True (default), upsert results into the local catalog so
            subsequent offline searches include these datasets.

        Returns
        -------
        list[dict]
            Dataset records from the live API, filtered by this query's
            modality and task filters. Shape matches local catalog records.
        """
        from qortex.client.graphql import OpenNeuroClient

        client = OpenNeuroClient(token=token)
        try:
            refs = client.search_datasets(
                modality=self._modality,
                task=self._task,
                limit=self._limit,
            )
        finally:
            client.close()

        results = [
            {
                "dataset_id": ref.id,
                "name": ref.name,
                "doi": ref.doi,
                "license": ref.license,
                "modalities": ref.modalities,
                "tasks": ref.tasks,
                "authors": ref.authors,
                "score": None,
            }
            for ref in refs
        ]

        # Text filter on top of API results (API has limited server-side filtering)
        if self._query:
            q = self._query.lower()
            results = [
                r for r in results
                if q in (r.get("name") or "").lower()
                or q in (r.get("dataset_id") or "").lower()
                or any(q in t.lower() for t in (r.get("tasks") or []))
            ]

        if sync_local and results:
            index = CatalogIndex(self._catalog_path)
            try:
                index.upsert_many(results)
            finally:
                index.close()

        return results[self._offset : self._offset + self._limit]

    def live_page(
        self,
        token: str | None = None,
        sync_local: bool = True,
    ) -> PagedResults:
        """Live search with pagination metadata."""
        all_results = self.live(token=token, sync_local=sync_local)
        # live() already applies limit, fetch full to count
        # Re-fetch without offset/limit for accurate total
        orig_limit, orig_offset = self._limit, self._offset
        self._limit = 10_000
        self._offset = 0
        all_full = self.live(token=token, sync_local=False)
        self._limit = orig_limit
        self._offset = orig_offset
        total = len(all_full)
        page = all_full[orig_offset : orig_offset + orig_limit]
        return PagedResults(results=page, total=total, offset=orig_offset, limit=orig_limit)


def search(
    query: str | None = None,
    *,
    modality: str | None = None,
    task: str | None = None,
    author: str | None = None,
    license: str | None = None,
    min_subjects: int | None = None,
    max_size_gb: float | None = None,
    has_events: bool | None = None,
    has_derivatives: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper: local catalog search.

    Returns matched datasets ordered by relevance score (text match +
    size heuristics). Use fetch_page() on DatasetQuery for pagination
    metadata, or live() to search the OpenNeuro API in real-time.
    """
    builder = DatasetQuery(catalog_path).limit(limit).offset(offset)
    if query:
        builder.containing(query)
    if modality:
        builder.modality(modality)
    if task:
        builder.task(task)
    if author:
        builder.author(author)
    if license:
        builder.license(license)
    if min_subjects is not None:
        builder.min_subjects(min_subjects)
    if max_size_gb is not None:
        builder.max_size_gb(max_size_gb)
    if has_events is not None:
        builder.has_events(has_events)
    if has_derivatives is not None:
        builder.has_derivatives(has_derivatives)
    return builder.fetch()


def live_search(
    modality: str | None = None,
    task: str | None = None,
    query: str | None = None,
    limit: int = 50,
    token: str | None = None,
    sync_local: bool = True,
    catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Search OpenNeuro API in real-time.

    Parameters
    ----------
    modality:
        Filter by modality (e.g. ``"eeg"``, ``"bold"``).
    task:
        Filter by task name.
    query:
        Text filter applied client-side over name, id, and tasks.
    limit:
        Maximum results to return.
    token:
        Optional API token.
    sync_local:
        When True (default), persist results to local catalog for offline use.
    catalog_path:
        Override the default catalog DB path.
    """
    builder = (
        DatasetQuery(catalog_path)
        .limit(limit)
    )
    if modality:
        builder.modality(modality)
    if task:
        builder.task(task)
    if query:
        builder.containing(query)
    return builder.live(token=token, sync_local=sync_local)


def facets(
    limit: int = 50,
    catalog_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return discovery facets from the local catalog.

    Returns
    -------
    dict with keys: ``modalities``, ``tasks``, ``licenses``, ``keywords``.
    Each value is a list of ``{"value": str, "n": int}`` dicts sorted by count.
    """
    return DatasetQuery(catalog_path).facets(limit=limit)
