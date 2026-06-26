"""project_02_manifest_fetch

Fetches a real OpenNeuro manifest and checks that BIDS entity parsing,
file metadata, and summary statistics are correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, require_gt, passed,
)


def main() -> None:
    banner("project_02: manifest fetch and BIDS entity parsing")

    ds, manifest = real_manifest()

    # ── summary stats ─────────────────────────────────────────────────────────
    print_kv("manifest", {
        "dataset": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "doi": manifest.doi,
        "files": manifest.summary.file_count,
        "subjects": manifest.summary.n_subjects,
        "sessions": len(manifest.summary.sessions or []),
        "tasks": ", ".join((manifest.summary.tasks or [])[:8]),
        "modalities": ", ".join(manifest.summary.modalities or []),
    })

    require_gt(manifest.summary.file_count, 0, "file_count")
    require_gt(manifest.summary.n_subjects, 0, "n_subjects")
    require(manifest.files, "manifest.files is empty")

    # ── BIDS entity parsing on a sample of files ──────────────────────────────
    non_dir = [f for f in manifest.files if not f.is_dir]
    require(non_dir, "no non-directory FileRecords in manifest")

    sample = non_dir[:20]
    print_rows("sample files", [
        {
            "path": f.path,
            "subject": f.subject,
            "extension": f.extension,
            "suffix": f.suffix,
            "modality": f.modality,
            "size": f.size,
        }
        for f in sample
    ])

    # at least some files must have subject labels
    with_subject = [f for f in non_dir if f.subject]
    require(with_subject, "no files have a parsed subject entity")

    # ── O(1) path lookup ──────────────────────────────────────────────────────
    first = non_dir[0]
    found = manifest.get_file(first.path)
    require(found is not None, f"get_file({first.path!r}) returned None")
    require(manifest.has_file(first.path), f"has_file({first.path!r}) returned False")
    require(not manifest.has_file("this/path/does/not/exist.eeg"), "has_file returned True for missing path")

    # ── files_by_suffix helper ────────────────────────────────────────────────
    tsv_files = manifest.files_by_suffix("tsv")
    require(tsv_files is not None, "files_by_suffix('tsv') returned None")
    # dataset_description.json should exist
    json_files = manifest.files_by_suffix("json")
    require(json_files is not None, "files_by_suffix('json') returned None")

    passed("project_02_manifest_fetch")


if __name__ == "__main__":
    main()
