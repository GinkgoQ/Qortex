"""project_15_remote_inspection

Exercises Dataset.participants(), .events(), .sidecar(), and .nifti_info()
to confirm that remote metadata can be read without a full download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, real_manifest,
    require, passed,
)


def main() -> None:
    banner("project_15: remote inspection without full download")

    ds, manifest = real_manifest()
    subjects = manifest.summary.subjects or []
    require(subjects, "manifest has no subjects")
    sub = subjects[0]

    # ── participants() ────────────────────────────────────────────────────────
    participants = ds.participants()
    print_kv("participants()", {
        "type": type(participants).__name__,
        "rows": len(participants) if participants is not None else None,
        "columns": list(participants.columns) if participants is not None else None,
    })
    require(participants is not None, "participants() returned None")
    require(len(participants) > 0, "participants() returned empty table")
    require(
        any("participant" in c.lower() or "subject" in c.lower() for c in participants.columns),
        "participants table has no subject/participant column",
    )

    # ── events() ─────────────────────────────────────────────────────────────
    # find a task and session from the manifest
    events_files = [
        f for f in manifest.files
        if f.suffix == "events" and f.extension == ".tsv" and f.subject == sub
    ]

    if events_files:
        ef = events_files[0]
        events_df = ds.events(
            subject=ef.subject,
            session=ef.session,
            task=ef.task,
            run=ef.run,
        )
        print_kv(f"events(sub={ef.subject!r}, task={ef.task!r})", {
            "rows": len(events_df) if events_df is not None else None,
            "columns": list(events_df.columns) if events_df is not None else None,
        })
        if events_df is not None:
            require(len(events_df) > 0, "events() returned empty DataFrame")
            require("onset" in events_df.columns, "events DataFrame missing 'onset' column")
    else:
        print_kv("events()", "no events.tsv for subject — skipped")

    # ── sidecar() ─────────────────────────────────────────────────────────────
    json_files = [f for f in manifest.files if f.extension == ".json" and not f.is_dir and f.urls]
    if json_files:
        jf = json_files[0]
        sidecar = ds.sidecar(jf.path)
        print_kv(f"sidecar({jf.path!r})", {
            "type": type(sidecar).__name__,
            "keys": list(sidecar.keys())[:8] if isinstance(sidecar, dict) else None,
        })
        require(isinstance(sidecar, dict), "sidecar() did not return a dict")
    else:
        print_kv("sidecar()", "no JSON files with URLs — skipped")

    # ── nifti_info() ─────────────────────────────────────────────────────────
    nii_files = [
        f for f in manifest.files
        if f.extension in {".nii", ".nii.gz"} and f.urls and not f.is_dir
    ]
    if nii_files:
        nf = nii_files[0]
        info = ds.nifti_info(nf.path)
        print_kv(f"nifti_info({nf.path!r})", {
            "type": type(info).__name__,
            "dim": getattr(info, "dim", None),
            "pixdim": getattr(info, "pixdim", None),
            "datatype": getattr(info, "datatype", None),
        })
        require(info is not None, "nifti_info() returned None")
    else:
        print_kv("nifti_info()", "no NIfTI files with URLs — skipped")

    passed("project_15_remote_inspection")


if __name__ == "__main__":
    main()
