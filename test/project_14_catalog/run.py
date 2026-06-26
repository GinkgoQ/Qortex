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
from qortex.catalog.search import DatasetQuery, PagedResults


def main() -> None:
    banner("project_14: catalog search and facets")

    # ── basic search ──────────────────────────────────────────────────────────
    query = qortex.search("eeg")
    require(isinstance(query, DatasetQuery), f"qortex.search returned {type(query).__name__}")

    page: PagedResults = query.fetch_page()
    print_kv("search('eeg') first page", {
        "total": page.total,
        "page_size": len(page.results),
    })
    require_gt(page.total, 0, "search total")
    require(page.results, "first page results are empty")

    # ── count ─────────────────────────────────────────────────────────────────
    total = query.count()
    require_gt(total, 0, "query.count()")
    print_kv("query.count()", total)

    # ── modality filter ───────────────────────────────────────────────────────
    eeg_query = DatasetQuery(modality="eeg")
    eeg_page = eeg_query.fetch_page()
    print_kv("modality=eeg page", {
        "total": eeg_page.total,
        "results": len(eeg_page.results),
    })
    require_gt(eeg_page.total, 0, "eeg modality total")

    # verify each result has a dataset_id
    for ds_info in eeg_page.results:
        require(ds_info.dataset_id, f"result missing dataset_id: {ds_info!r}")

    # ── dataset fields ────────────────────────────────────────────────────────
    sample = eeg_page.results[0]
    print_kv("first result", {
        "dataset_id": sample.dataset_id,
        "name": getattr(sample, "name", None),
        "n_subjects": getattr(sample, "n_subjects", None),
        "modalities": getattr(sample, "modalities", None),
        "license": getattr(sample, "license", None),
    })

    # ── facets ────────────────────────────────────────────────────────────────
    facet_result = qortex.facets()
    print_kv("facets keys", list(facet_result.keys()) if isinstance(facet_result, dict) else type(facet_result).__name__)
    require(facet_result, "qortex.facets() returned empty result")

    if isinstance(facet_result, dict):
        for key, values in list(facet_result.items())[:3]:
            print_kv(f"facet[{key}]", values[:5] if isinstance(values, list) else values)

    # ── live_search iterator ──────────────────────────────────────────────────
    seen = []
    for item in qortex.live_search("eeg", limit=5):
        seen.append(item.dataset_id)
    print_kv("live_search results", len(seen))
    require(seen, "live_search returned no results")

    passed("project_14_catalog")


if __name__ == "__main__":
    main()
