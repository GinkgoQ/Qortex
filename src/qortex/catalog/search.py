"""High-level search interface over the catalog index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.catalog.index import CatalogIndex
from qortex.core.config import get_config


def _default_catalog_path() -> Path:
    cfg = get_config()
    return cfg.cache_dir / "catalog" / "catalog.duckdb"


class DatasetQuery:
    """Fluent builder for catalog searches.

    Usage::

        results = (
            DatasetQuery()
            .modality("eeg")
            .min_subjects(20)
            .containing("auditory")
            .limit(10)
            .fetch()
        )
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

    def fetch(self) -> list[dict[str, Any]]:
        """Execute the query and return matching dataset records."""
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

    def as_dataframe(self):
        """Return results as a Polars DataFrame."""
        import polars as pl

        rows = self.fetch()
        return pl.DataFrame(rows) if rows else pl.DataFrame()


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
    """Convenience wrapper around DatasetQuery.fetch()."""
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
