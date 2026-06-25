from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import qortex
from qortex.catalog.index import CatalogIndex
from qortex.catalog.refresh import refresh
from qortex.catalog.search import search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import print_kv, print_rows, require  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        qortex.configure(cache_dir=Path(tmp) / "cache")
        n_indexed = refresh(max_pages=1, progress=False)
        catalog_path = qortex.get_config().cache_dir / "catalog" / "catalog.duckdb"
        index = CatalogIndex(catalog_path)
        count = index.count()
        index.close()
        results = search(limit=10, catalog_path=catalog_path)

        print_kv(
            "PROJECT 11: real OpenNeuro catalog refresh and search",
            {
                "indexed this run": n_indexed,
                "catalog count": count,
                "result count": len(results),
                "catalog path": catalog_path,
            },
        )
        print_rows(
            "Real catalog results",
            [
                {
                    "dataset_id": row.get("dataset_id"),
                    "name": row.get("name"),
                    "subjects": row.get("n_subjects"),
                    "modalities": row.get("modalities"),
                    "snapshot": row.get("snapshot"),
                }
                for row in results
            ],
            limit=10,
        )

        require(n_indexed > 0, "real catalog refresh indexed no datasets")
        require(count >= n_indexed, "catalog count is smaller than refresh count")
        require(results, "real catalog search returned no results")

    print("RESULT: real catalog project passed")


if __name__ == "__main__":
    main()
