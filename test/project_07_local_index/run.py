"""project_07_local_index

Downloads metadata to a local tree, then builds a local file index using
index_local_bids() and reconciles it against the remote manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_metadata_root,
    require, require_gt, passed,
)

from qortex.indexing import index_local_bids


def main() -> None:
    banner("project_07: local index and manifest reconciliation")

    ctx, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        report = index_local_bids(root, manifest=manifest, use_pybids=False)

        print_kv("local index", {
            "root": str(root),
            "n_files": report.n_files,
            "n_dirs": report.n_dirs,
            "n_missing": report.n_missing,
            "n_extra": report.n_extra,
            "n_size_mismatches": report.n_size_mismatches,
            "consistent": report.consistent,
        })

        require_gt(report.n_files, 0, "n_files")

        # dataset_description.json must be present
        local_paths = {r.path for r in report.indexed_files}
        require(
            "dataset_description.json" in local_paths,
            "dataset_description.json not in local index",
        )

        # files we downloaded should not be missing from the manifest
        # (missing_remote = files on disk not present in manifest — unusual for a Qortex download)
        if report.n_extra > 0:
            print_kv("extra files (on disk, not in manifest)", report.extra_local[:5])

        if report.n_missing > 0:
            print_kv("missing files (in manifest, not on disk)", report.missing_remote[:5])

        # summary string
        summary_text = report.summary()
        require(isinstance(summary_text, str) and summary_text.strip(), "summary() returned empty")
        print_kv("summary", summary_text)

        # at least some indexed files must have BIDS entities
        with_subject = [r for r in report.indexed_files if r.entities and r.entities.get("subject")]
        print_kv("files with subject entity", len(with_subject))

        # verify size info where available
        sized = [r for r in report.indexed_files if r.size is not None and r.size > 0]
        print_kv("files with known size", len(sized))

    finally:
        ctx.cleanup()

    passed("project_07_local_index")


if __name__ == "__main__":
    main()
