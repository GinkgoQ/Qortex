"""Scenario 19: SignalBudget — remote sidecar + NIfTI header acquisition analysis.

Fetches JSON sidecars for all signal files from CDN, falls back to NIfTI header
Range requests for fMRI files missing TR. Produces per-modality recording hour
estimates and window count projections. No bytes written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, print_kv, require  # noqa: E402

import qortex


def main() -> None:
    ds = qortex.Dataset(DATASET_ID)

    print("\n--- signal_budget() ---")
    # Cap sidecars to keep the scenario fast
    budget = ds.signal_budget(include_nifti_headers=True)

    print_kv("n_sidecars_fetched", budget.n_sidecars_fetched)
    print_kv("n_sidecars_failed", budget.n_sidecars_failed)
    print_kv("modalities", list(budget.modality_budgets.keys()))

    for modality, mb in budget.modality_budgets.items():
        print(
            f"\n  [{modality}]  subjects={mb.n_subjects}  "
            f"files={mb.n_files}  hours={mb.total_hours:.2f}  "
            f"avg_sfreq={mb.avg_sfreq:.1f}Hz"
        )

    # Window estimates
    print("\nWindow estimates at 2s / 50% overlap:")
    windows = budget.estimate_windows(window_duration_s=2.0, overlap=0.5)
    for modality, count in windows.items():
        print(f"  {modality}: {count:,} windows")

    print("\nPer-subject windows at 2s:")
    per_subject = budget.per_subject_windows(window_duration_s=2.0)
    for sub, count in list(per_subject.items())[:5]:
        print(f"  {sub}: {count:,}")

    if budget.adequacy_warnings:
        print("\nAdequacy warnings:")
        for w in budget.adequacy_warnings:
            print(f"  • {w}")

    if budget.n_sidecars_fetched > 0:
        require(len(budget.modality_budgets) > 0, "Expected at least one modality budget")
    print("\nScenario 19 complete.")


if __name__ == "__main__":
    main()
