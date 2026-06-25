from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import first_metadata_table, print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    ds, manifest = real_manifest()
    table = first_metadata_table(manifest)
    rows = ds.first_rows(table.path, n=3, max_bytes=4096)
    description = ds.preview("dataset_description.json", max_bytes=4096)

    print_kv(
        "PROJECT 14: real live OpenNeuro metadata smoke",
        {
            "dataset": ds.dataset_id,
            "snapshot": manifest.snapshot,
            "files": manifest.summary.file_count,
            "subjects": manifest.summary.n_subjects,
            "table": table.path,
            "table rows": len(rows),
            "description source": description.source,
            "description bytes": description.bytes_read,
        },
    )
    print_rows(f"{table.path} first rows", rows, limit=3)

    require(manifest.summary.file_count > 0, "live manifest returned no files")
    require(manifest.summary.n_subjects > 0, "live manifest returned no subjects")
    require(rows, "live table preview returned no rows")
    require(description.source == "remote", "live description preview did not use remote source")
    require(description.bytes_read > 0, "live description preview returned no bytes")

    print("RESULT: real live OpenNeuro project passed")


if __name__ == "__main__":
    main()
