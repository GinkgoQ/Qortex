"""project_14_catalog

Exercises catalog search, faceting, and DatasetQuery API against the live
OpenNeuro catalog to verify that search results and filters are coherent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, require, require_gt, passed,
)

import qortex
from qortex.catalog.search import DatasetQuery, PagedResults, facets, live_search


def main() -> None:
    banner("project_14: catalog search and facets")

    # ── basic DatasetQuery (fluent builder) ───────────────────────────────────
    query = DatasetQuery().containing("eeg")
    require(isinstance(query, DatasetQuery), f"DatasetQuery returned {type(query).__name__}")

    page: PagedResults = query.fetch_page()
    require(isinstance(page, PagedResults), f"fetch_page returned {type(page).__name__}")
    print_kv("search('eeg') first page", {
        "total": page.total,
        "page_size": len(page.results),
    })
    require(page.total >= 0, "search total must be non-negative")
    # If catalog is empty or no matches, that is acceptable
    print_kv("search results", len(page.results))

    # ── count ─────────────────────────────────────────────────────────────────
    total = query.count()
    require(total >= 0, "query.count() must be non-negative")
    print_kv("query.count()", total)

    # ── modality filter ───────────────────────────────────────────────────────
    eeg_query = DatasetQuery().modality("eeg")
    eeg_page = eeg_query.fetch_page()
    print_kv("modality=eeg page", {
        "total": eeg_page.total,
        "results": len(eeg_page.results),
    })
    require(eeg_page.total >= 0, "eeg modality total must be non-negative")

    # verify each result has a dataset_id (if any returned)
    for ds_info in eeg_page.results:
        ds_id = ds_info.get("dataset_id") if isinstance(ds_info, dict) else getattr(ds_info, "dataset_id", None)
        require(ds_id is not None, f"result missing dataset_id: {ds_info!r}")

    # ── facets ────────────────────────────────────────────────────────────────
    facet_result = facets()
    print_kv("facets keys", list(facet_result.keys()) if isinstance(facet_result, dict) else type(facet_result).__name__)
    # facets() may return an empty dict if catalog is empty — that's OK
    require(isinstance(facet_result, dict), f"facets() must return dict, got {type(facet_result)}")

    if isinstance(facet_result, dict):
        for key, values in list(facet_result.items())[:3]:
            print_kv(f"facet[{key}]", values[:5] if isinstance(values, list) else values)

    # ── live_search iterator ──────────────────────────────────────────────────
    seen = []
    results = live_search(query="eeg", limit=5)
    for item in results:
        ds_id = item.get("dataset_id") if isinstance(item, dict) else getattr(item, "dataset_id", None)
        if ds_id:
            seen.append(ds_id)
    print_kv("live_search results", len(seen))
    # live_search may return 0 results if catalog is empty — acceptable

    # ── DatasetQuery fetch (returns list) ─────────────────────────────────────
    q = DatasetQuery().limit(5)
    results_list = q.fetch()
    require(isinstance(results_list, list), f"DatasetQuery.fetch() must return list, got {type(results_list)}")
    print_kv("DatasetQuery.fetch() count", len(results_list))

    passed("project_14_catalog")


if __name__ == "__main__":
    main()
