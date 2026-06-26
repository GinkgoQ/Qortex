"""project_10_events_to_parquet

Loads events TSV files from downloaded metadata using BehaviorLoader, then
saves the concatenated events to Parquet using polars and verifies the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_metadata_root, artifact_dir,
    require, require_gt, passed,
)

from qortex.parse.behavior import BehaviorLoader


def main() -> None:
    banner("project_10: behavior events to Parquet")

    ctx, ds, root = real_metadata_root()
    try:
        out_dir = artifact_dir(root, "parquet_output")
        manifest = ds.manifest()
        loader = BehaviorLoader()

        events_files = [
            (f, root / f.path)
            for f in manifest.files
            if f.suffix == "events" and f.extension == ".tsv" and not f.is_dir
        ]
        events_files = [(fr, p) for fr, p in events_files if p.exists()]

        print_kv("events files on disk", len(events_files))
        require(events_files, "no events.tsv files found locally — metadata download missing events?")

        # ── load via BehaviorLoader ───────────────────────────────────────────
        all_rows: list[dict] = []
        skipped = 0
        for fr, local_path in events_files[:20]:
            try:
                record = loader.load(fr, local_path)
            except Exception as exc:
                skipped += 1
                continue
            for row in record.data.to_dicts():
                all_rows.append({
                    "subject": fr.subject,
                    "session": fr.session,
                    "task": fr.task,
                    "run": fr.run,
                    "onset": row.get("onset"),
                    "duration": row.get("duration"),
                    "trial_type": row.get("trial_type"),
                    "source_file": fr.path,
                })

        print_kv("loaded events", {"rows": len(all_rows), "skipped_files": skipped})
        require_gt(len(all_rows), 0, "events row count")

        # ── save to Parquet via polars ────────────────────────────────────────
        import polars as pl

        df = pl.DataFrame(all_rows, infer_schema_length=len(all_rows))
        parquet_path = out_dir / "events.parquet"
        df.write_parquet(parquet_path)

        require(parquet_path.exists(), f"Parquet file not written: {parquet_path}")
        require(parquet_path.stat().st_size > 0, "Parquet file is empty")

        # ── read back and verify ──────────────────────────────────────────────
        df2 = pl.read_parquet(parquet_path)
        print_kv("Parquet artifact", {
            "path": str(parquet_path),
            "size_bytes": parquet_path.stat().st_size,
            "rows": len(df2),
            "columns": list(df2.columns),
        })

        require_gt(len(df2), 0, "Parquet row count after reload")
        require("onset" in df2.columns, "Parquet schema missing 'onset'")
        require("source_file" in df2.columns, "Parquet schema missing 'source_file'")

        # ── inspect one events file ───────────────────────────────────────────
        sample_fr, sample_path = events_files[0]
        info = loader.inspect(sample_fr, sample_path)
        print_kv("events file inspection", {
            "path": sample_fr.path,
            "n_rows": info["n_rows"],
            "columns": info["columns"],
            "label_column": info["label_column"],
            "onset_column": info["onset_column"],
        })
        require("n_rows" in info, "inspect() missing n_rows")
        require("columns" in info, "inspect() missing columns")

    finally:
        ctx.cleanup()

    passed("project_10_events_to_parquet")


if __name__ == "__main__":
    main()
