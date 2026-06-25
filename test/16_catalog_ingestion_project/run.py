from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import qortex
from qortex.catalog.index import CatalogIndex
from qortex.catalog.refresh import refresh_dataset
from qortex.catalog.search import search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, SNAPSHOT, print_kv, print_rows, require  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        qortex.configure(cache_dir=Path(tmp) / "cache")
        catalog_path = qortex.get_config().cache_dir / "catalog" / "catalog.duckdb"

        profile = refresh_dataset(
            DATASET_ID,
            snapshot=SNAPSHOT,
            catalog_path=catalog_path,
            include_file_summary=True,
        )
        results = search(
            query="balloonanalogrisktask",
            has_events=True,
            limit=5,
            catalog_path=catalog_path,
        )
        index = CatalogIndex(catalog_path)
        try:
            facets = index.facets(limit=8)
            stored = index.profile(DATASET_ID)
        finally:
            index.close()

        print_kv(
            "PROJECT 16: deep catalog ingestion and metadata digestion",
            {
                "dataset": profile.get("dataset_id"),
                "snapshot": profile.get("snapshot"),
                "subjects": profile.get("n_subjects"),
                "sessions": profile.get("n_sessions"),
                "tasks": ", ".join(profile.get("tasks") or []),
                "modalities": ", ".join(profile.get("modalities") or []),
                "has events": profile.get("has_events"),
                "event files": profile.get("n_event_files"),
                "derivative files": profile.get("n_derivative_files"),
                "file summaries": len(profile.get("file_summaries") or []),
                "search hits": len(results),
            },
        )
        print_rows(
            "Task/event search results",
            [
                {
                    "dataset_id": row.get("dataset_id"),
                    "name": row.get("name"),
                    "score": row.get("score"),
                    "subjects": row.get("n_subjects"),
                    "tasks": row.get("tasks"),
                    "events": row.get("n_event_files"),
                }
                for row in results
            ],
        )
        print_rows("Top modality facets", facets["modalities"], limit=8)
        print_rows(
            "File summaries",
            [
                {
                    "category": row.get("category"),
                    "value": row.get("value"),
                    "files": row.get("n_files"),
                    "bytes": row.get("bytes"),
                }
                for row in (profile.get("file_summaries") or [])[:12]
            ],
            limit=12,
        )

        require(stored is not None, "deep-ingested dataset was not stored")
        require(profile.get("n_subjects"), "profile did not digest subjects from manifest")
        require(profile.get("tasks"), "profile did not digest tasks from manifest")
        require(profile.get("has_events") is True, "profile did not detect events")
        require((profile.get("n_event_files") or 0) > 0, "profile counted no event files")
        require(profile.get("file_summaries"), "profile has no file summaries")
        require(results and results[0]["dataset_id"] == DATASET_ID, "task search did not recover the deep-ingested dataset")

    print("RESULT: deep catalog ingestion project passed")


if __name__ == "__main__":
    main()
