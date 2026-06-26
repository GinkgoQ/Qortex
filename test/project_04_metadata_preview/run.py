"""project_04_metadata_preview

Exercises remote-preview capabilities: fetching TSV/JSON sidecar content
directly from CDN URLs without downloading the full dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, real_manifest,
    require, require_gt, passed,
    first_events_file, first_metadata_table,
)

from qortex.client.remote import RemoteFileGateway


def main() -> None:
    banner("project_04: remote metadata preview without full download")

    ds, manifest = real_manifest()
    gateway = RemoteFileGateway()

    # ── events.tsv preview ────────────────────────────────────────────────────
    ev_file = first_events_file(manifest)
    print_kv("events file", ev_file.path)

    if ev_file.urls:
        url = ev_file.urls[0]
        data = gateway.fetch_bytes(url, max_bytes=16_384)
        require(len(data) > 0, "events.tsv fetch returned 0 bytes")
        text = data.decode("utf-8", errors="replace")
        require("onset" in text or "\t" in text, "events.tsv content does not look like TSV")
        print_kv("events preview (first 200 chars)", text[:200].replace("\n", " "))
    else:
        print_kv("events file", "no URL — skipping byte fetch")

    # ── participants.tsv or other metadata table ──────────────────────────────
    meta_file = first_metadata_table(manifest)
    print_kv("metadata table", meta_file.path)

    if meta_file.urls:
        url_map = {meta_file.path: meta_file.urls[0]}
        results = gateway.batch_fetch_tsv(url_map, max_bytes_per_file=512_000)
        table = results.get(meta_file.path)
        require(table is not None, f"batch_fetch_tsv returned None for {meta_file.path!r}")
        require(len(table) > 0, "metadata table is empty")
        print_kv("metadata table shape", f"{len(table)} rows × {len(table.columns)} cols")
        print_kv("columns", list(table.columns))

    # ── JSON sidecar batch fetch ──────────────────────────────────────────────
    json_files = [f for f in manifest.files if f.extension == ".json" and f.urls and not f.is_dir][:3]
    if json_files:
        json_url_map = {f.path: f.urls[0] for f in json_files}
        json_results = gateway.batch_fetch_json(json_url_map, max_bytes_per_file=262_144)
        fetched = {k: v for k, v in json_results.items() if v is not None}
        print_kv("JSON files fetched", len(fetched))
        for path, content in list(fetched.items())[:2]:
            keys = list(content.keys())[:5] if isinstance(content, dict) else []
            print_kv(f"  {path}", f"{len(keys)} top-level keys: {keys}")
        require(fetched, "batch_fetch_json returned no results")

    passed("project_04_metadata_preview")


if __name__ == "__main__":
    main()
