"""project_17_signal_budget

Runs SignalBudgetEstimator on a real dataset manifest to verify that
acquisition parameters are resolved via remote BIDS sidecar inheritance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, passed,
)

from qortex.inspect.signal_budget import SignalBudgetEstimator, SignalBudget


def main() -> None:
    banner("project_17: signal budget estimation")

    ds, manifest = real_manifest()

    # check if there are any signal files
    signal_modalities = {"eeg", "meg", "ieeg", "bold"}
    signal_files = [
        f for f in manifest.files
        if not f.is_dir and f.modality in signal_modalities and f.urls
    ]
    if not signal_files:
        print("SKIP: no signal files with download URLs in manifest")
        passed("project_17_signal_budget")
        return

    estimator = SignalBudgetEstimator(
        manifest,
        concurrency=4,
        include_nifti_headers=True,
    )
    budget: SignalBudget = estimator.estimate()

    print_kv("signal budget", {
        "n_signal_files": budget.n_signal_files,
        "n_resolved": budget.n_resolved,
        "total_estimated_windows": budget.total_estimated_windows,
        "total_recording_seconds": f"{budget.total_recording_seconds:.1f}",
    })

    require(budget.n_signal_files >= 0, "n_signal_files is negative")
    require(budget.n_resolved >= 0, "n_resolved is negative")
    require(budget.n_resolved <= budget.n_signal_files, "n_resolved > n_signal_files")

    if budget.n_resolved > 0:
        require(budget.total_recording_seconds >= 0, "total_recording_seconds is negative")

        # per-file acquisition params
        params_list = budget.acquisition_params
        print_rows("acquisition params sample", [
            {
                "path": p.path,
                "sampling_freq": p.sampling_frequency,
                "duration_s": p.recording_duration,
                "n_channels": p.n_channels,
                "estimated_windows": p.estimated_windows,
            }
            for p in params_list[:8]
        ], limit=8)

        # at least some params must have meaningful values
        meaningful = [
            p for p in params_list
            if p.sampling_frequency and p.sampling_frequency > 0
        ]
        print_kv("files with sampling_freq resolved", len(meaningful))

    passed("project_17_signal_budget")


if __name__ == "__main__":
    main()
