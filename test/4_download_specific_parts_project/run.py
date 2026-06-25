from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import primary_recording_with_events, print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    ds, manifest = real_manifest()
    recording = primary_recording_with_events(manifest)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metadata_result = ds.download_metadata(output_dir=root / "metadata", dry_run=True, max_size_gb=0.2)
        exact_result = ds.download_paths(
            [recording.primary.path],
            output_dir=root / "exact",
            dry_run=True,
            max_size_gb=5.0,
        )

    print_kv(
        "PROJECT 4: real metadata-only and exact-path download plans",
        {
            "dataset": ds.dataset_id,
            "metadata-only files": metadata_result.plan.n_files,
            "metadata-only bytes": metadata_result.plan.estimated_bytes,
            "exact-path closure files": exact_result.plan.n_files,
            "exact-path bytes": exact_result.plan.estimated_bytes,
        },
    )
    print_rows(
        "Real metadata-only plan",
        [{"path": file.path, "size": file.size, "extension": file.extension} for file in metadata_result.plan.files],
        limit=16,
    )
    print_rows(
        "Real exact path plus companions",
        [{"path": file.path, "size": file.size} for file in exact_result.plan.files],
        limit=16,
    )

    metadata_paths = {file.path for file in metadata_result.plan.files}
    exact_paths = {file.path for file in exact_result.plan.files}
    require("dataset_description.json" in metadata_paths, "real metadata plan omitted dataset_description.json")
    require(any(path.endswith(".tsv") or path.endswith(".json") for path in metadata_paths), "real metadata plan has no table/sidecar files")
    require(recording.primary.path in exact_paths, "exact-path primary missing")
    require(recording.companions.events.path in exact_paths, "exact-path real events companion missing")

    print("RESULT: real specific-download project passed")


if __name__ == "__main__":
    main()
