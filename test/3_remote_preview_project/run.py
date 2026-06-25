from __future__ import annotations

import sys
from pathlib import Path

import qortex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import first_metadata_table, print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    ds, manifest = real_manifest()
    table = first_metadata_table(manifest)
    table_preview = ds.preview(table.path, n_rows=5, max_bytes=16000)
    description_preview = ds.preview("dataset_description.json", max_bytes=16000)

    print_kv(
        "PROJECT 3: real remote metadata preview without full download",
        {
            "dataset": ds.dataset_id,
            "snapshot": manifest.snapshot,
            "table": table.path,
            "table source": table_preview.source,
            "table bytes read": table_preview.bytes_read,
            "table columns": ", ".join(table_preview.columns),
            "description bytes": description_preview.bytes_read,
        },
    )
    print_rows(f"{table.path} first rows", table_preview.rows, limit=5)
    print("dataset_description.json preview:")
    print(description_preview.text[:1200] if description_preview.text else "(no text)")

    require(table_preview.source == "remote", "preview did not use remote OpenNeuro data")
    require(table_preview.rows, "real metadata table preview returned no rows")
    require(table_preview.columns, "real metadata table preview returned no columns")
    require(description_preview.text and "Name" in description_preview.text, "real dataset description preview is missing Name")
    require(qortex.FilePreview is not None, "public FilePreview export is missing")

    print("RESULT: real preview project passed")


if __name__ == "__main__":
    main()
