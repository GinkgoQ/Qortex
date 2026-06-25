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
        self._min_subjects: int | None = None
        self._limit: int = 50

    def containing(self, text: str) -> "DatasetQuery":
        self._query = text
        return self

    def modality(self, mod: str) -> "DatasetQuery":
        self._modality = mod
        return self

    def min_subjects(self, n: int) -> "DatasetQuery":
        self._min_subjects = n
        return self

    def limit(self, n: int) -> "DatasetQuery":
        self._limit = n
        return self

    def fetch(self) -> list[dict[str, Any]]:
        """Execute the query and return matching dataset records."""
        index = CatalogIndex(self._catalog_path)
        try:
            return index.search(
                query=self._query,
                modality=self._modality,
                min_subjects=self._min_subjects,
                limit=self._limit,
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
    min_subjects: int | None = None,
    limit: int = 50,
    catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper around DatasetQuery.fetch()."""
    return (
        DatasetQuery(catalog_path)
        .containing(query or "")
        .modality(modality or "")
        .min_subjects(min_subjects or 0)
        .limit(limit)
        .fetch()
    )
