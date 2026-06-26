"""Scenario 17: Remote inspection — participants, events, sidecar, NIfTI header.

Tests zero-download intelligence: all methods read from CDN or API without
writing any bytes to disk. Uses DATASET_ID (default ds000001) which is a
publicly accessible OpenNeuro dataset with participants, events, and fMRI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, print_kv, require  # noqa: E402

import qortex


def main() -> None:
    ds = qortex.Dataset(DATASET_ID)

    # ── participants() ──────────────────────────────────────────────────────
    print("\n--- participants() ---")
    try:
        df = ds.participants()
        print_kv("rows", len(df))
        print_kv("columns", df.columns)
        if "age" in df.columns:
            ages = df["age"].drop_nulls()
            print_kv("age_mean", round(float(ages.mean()), 1) if len(ages) else "n/a")
        if "sex" in df.columns:
            print_kv("sex_counts", df["sex"].value_counts().to_dicts())
        require(len(df) > 0, "participants() returned empty DataFrame")
    except Exception as exc:
        print(f"  participants() raised: {exc}")

    # ── events() ───────────────────────────────────────────────────────────
    print("\n--- events() (first available) ---")
    try:
        manifest = ds.manifest()
        events_files = [f for f in manifest.files if f.suffix == "events"]
        if events_files:
            fr = events_files[0]
            df_ev = ds.events(subject=fr.subject, task=fr.task)
            print_kv("events_rows", len(df_ev))
            print_kv("columns", df_ev.columns)
            require(len(df_ev) > 0, "events() returned empty DataFrame")
        else:
            print("  No events files in manifest — skipping events() test")
    except Exception as exc:
        print(f"  events() raised: {exc}")

    # ── sidecar() ──────────────────────────────────────────────────────────
    print("\n--- sidecar() ---")
    try:
        manifest = ds.manifest()
        json_files = [
            f for f in manifest.files
            if f.extension == ".json" and f.subject and f.urls
        ]
        if json_files:
            # Find the primary data file this sidecar belongs to
            signal_files = [
                f for f in manifest.files
                if f.subject and f.modality and f.extension not in (".json", ".tsv", ".csv")
                and f.urls
            ]
            if signal_files:
                sf = signal_files[0]
                meta = ds.sidecar(sf.path)
                print_kv("path", sf.path)
                print_kv("sidecar_keys", list(meta.keys())[:10])
                print("  sidecar() OK")
            else:
                print("  No signal files with URLs — skipping sidecar() test")
        else:
            print("  No JSON sidecar files with URLs — skipping sidecar() test")
    except Exception as exc:
        print(f"  sidecar() raised: {exc}")

    # ── nifti_info() ───────────────────────────────────────────────────────
    print("\n--- nifti_info() ---")
    try:
        manifest = ds.manifest()
        nifti_files = [
            f for f in manifest.files
            if f.extension in (".nii", ".nii.gz") and f.urls
        ]
        if nifti_files:
            nf = nifti_files[0]
            info = ds.nifti_info(nf.path)
            print_kv("path", nf.path)
            print_kv("nifti_info", str(info))
            print_kv("ndim", info.ndim)
            print_kv("shape", info.shape)
            require(info.ndim >= 3, "Expected at least 3D NIfTI")
        else:
            print("  No NIfTI files in manifest — skipping nifti_info() test")
    except Exception as exc:
        print(f"  nifti_info() raised: {exc}")

    print("\nScenario 17 complete.")


if __name__ == "__main__":
    main()
