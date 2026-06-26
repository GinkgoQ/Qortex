"""project_16_label_landscape

Runs LabelLandscapeAnalyzer on a real dataset manifest to check that class
balance, trial counts, and events coverage are computed from remote files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, passed,
)

from qortex.inspect.label_landscape import LabelLandscapeAnalyzer, LabelLandscape


def main() -> None:
    banner("project_16: label landscape from remote events")

    ds, manifest = real_manifest()

    events_files = [
        f for f in manifest.files
        if f.suffix == "events" and f.extension == ".tsv"
    ]
    if not events_files:
        print("SKIP: no events files in manifest")
        passed("project_16_label_landscape")
        return

    analyzer = LabelLandscapeAnalyzer()
    landscape: LabelLandscape = analyzer.analyze(
        manifest,
        label_column="trial_type",
        concurrency=4,
        max_events_files=20,
    )

    print_kv("landscape fields", {
        "dataset_id": landscape.dataset_id,
        "label_column": landscape.label_column,
        "n_events_files": landscape.n_events_files,
        "n_files_fetched": landscape.n_files_fetched,
        "n_files_failed": landscape.n_files_failed,
        "coverage_pct": f"{landscape.coverage_pct:.1%}",
        "imbalance_ratio": f"{landscape.imbalance_ratio:.2f}" if landscape.imbalance_ratio else None,
    })

    require(landscape.label_column == "trial_type", "label_column not preserved")
    require(landscape.n_files_fetched >= 0, "n_files_fetched is negative")
    require(0.0 <= landscape.coverage_pct <= 1.0, f"coverage_pct {landscape.coverage_pct!r} out of [0,1]")

    if landscape.trial_type_stats:
        print_rows("trial_type_stats", [
            {"trial_type": getattr(s, "trial_type", str(s)), "count": getattr(s, "count", None)}
            for s in landscape.trial_type_stats[:8]
        ], limit=8)

    if landscape.imbalance_ratio is not None:
        require(landscape.imbalance_ratio >= 1.0, f"imbalance_ratio {landscape.imbalance_ratio} < 1")

    if landscape.subject_profiles:
        print_kv("subjects with data", len(landscape.subject_profiles))

    if landscape.recommendations:
        print_kv("recommendations", landscape.recommendations[:2])

    passed("project_16_label_landscape")


if __name__ == "__main__":
    main()
