from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import first_metadata_table, print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    ds, manifest = real_manifest()
    info = ds.info()
    metadata_files = ds.metadata_files()
    table = first_metadata_table(manifest)
    rows = ds.first_rows(table.path, n=5, max_bytes=16000)
    previews = ds.preview_metadata(n_rows=2, max_files=5)

    modality = manifest.summary.modalities[0] if manifest.summary.modalities else None
    filtered_files = ds.files(modalities=[modality]) if modality else ds.files()

    print_kv(
        "PROJECT 13: high-level Dataset facade on a real OpenNeuro dataset",
        {
            "dataset": ds.dataset_id,
            "snapshot": manifest.snapshot,
            "info files": info["n_files"],
            "info subjects": info["n_subjects"],
            "metadata files": len(metadata_files),
            "filtered modality": modality,
            "filtered files": len(filtered_files),
            "first rows": len(rows),
            "metadata previews": len(previews),
        },
    )
    print_rows("Dataset.info output", [info])
    print_rows("Dataset.metadata_files output", [{"path": file.path, "size": file.size} for file in metadata_files[:12]], limit=12)
    print_rows(f"Dataset.first_rows({table.path!r})", rows, limit=5)

    require(info["n_files"] > 0, "Dataset.info returned no files")
    require(info["n_subjects"] > 0, "Dataset.info returned no subjects")
    require(metadata_files, "Dataset.metadata_files returned no files")
    require(filtered_files, "Dataset.files returned no files")
    require(rows, "Dataset.first_rows returned no rows")
    require(previews, "Dataset.preview_metadata returned no previews")

    print("RESULT: real Dataset facade project passed")


if __name__ == "__main__":
    main()
