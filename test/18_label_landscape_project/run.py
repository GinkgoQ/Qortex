"""Scenario 18: LabelLandscape — concurrent remote events analysis.

Fetches all events TSVs for DATASET_ID from CDN and produces a LabelLandscape
report with class balance, ISI jitter, cross-subject consistency, and
recommendations. No bytes are written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, print_kv, require  # noqa: E402

import qortex


def main() -> None:
    ds = qortex.Dataset(DATASET_ID)

    print("\n--- label_landscape() ---")
    manifest = ds.manifest()
    events_count = sum(
        1 for f in manifest.files if f.suffix == "events" and f.extension == ".tsv"
    )
    print_kv("events_files_in_manifest", events_count)

    if events_count == 0:
        print("  No events files — dataset has no trial_type labels. Skipping.")
        return

    # Cap at 20 files so the scenario runs quickly
    landscape = ds.label_landscape(max_events_files=20)

    print_kv("n_events_files_total", landscape.n_events_files)
    print_kv("n_files_fetched", landscape.n_files_fetched)
    print_kv("label_column", landscape.label_column)
    print_kv("n_classes", landscape.n_classes)
    print_kv("total_events", landscape.total_events)
    print_kv("imbalance_ratio", landscape.imbalance_ratio)
    print_kv("imbalance_severity", landscape.imbalance_severity)
    print_kv("coverage_pct", round(landscape.coverage_pct * 100, 1))
    print_kv("cross_subject_consistency", round(landscape.cross_subject_consistency * 100, 1))

    print("\nTrialType stats (top 5):")
    for ts in landscape.trial_type_stats[:5]:
        print(f"  {ts.trial_type:25s}  count={ts.count:5d}  subjects={ts.n_subjects}")

    if landscape.isi_stats:
        print("\nISI stats (per task):")
        for task, isi in list(landscape.isi_stats.items())[:3]:
            print(
                f"  task={task:20s}  mean_isi={isi.mean_s:.2f}s  "
                f"jitter_cv={isi.jitter_cv:.3f}"
            )

    print("\nRecommendations:")
    for rec in landscape.recommendations[:5]:
        print(f"  • {rec}")

    print("\n" + landscape.summary())

    require(landscape.n_files_fetched > 0 or events_count == 0, "No events files were fetched")
    print("\nScenario 18 complete.")


if __name__ == "__main__":
    main()
