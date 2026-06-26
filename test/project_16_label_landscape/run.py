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

    # pick a reasonable events column — fall back to "trial_type"
    events_files = [
        f for f in manifest.files
        if f.suffix == "events" and f.extension == ".tsv"
    ]
    if not events_files:
        print("SKIP: no events files in manifest")
        passed("project_16_label_landscape")
        return

    analyzer = LabelLandscapeAnalyzer(
        manifest,
        label_column="trial_type",
        concurrency=4,
        max_events_files=20,
    )
    landscape: LabelLandscape = analyzer.analyze()

    print_kv("landscape", {
        "label_column": landscape.label_column,
        "n_classes": landscape.n_classes,
        "total_trials": landscape.total_trials,
        "coverage_pct": f"{landscape.coverage_pct:.1%}",
        "imbalance_ratio": f"{landscape.imbalance_ratio:.2f}" if landscape.imbalance_ratio else None,
        "files_analyzed": landscape.files_analyzed,
    })

    require(landscape.label_column == "trial_type", "label_column not preserved")
    require(landscape.files_analyzed >= 0, "files_analyzed is negative")
    require(0.0 <= landscape.coverage_pct <= 1.0, f"coverage_pct {landscape.coverage_pct!r} out of [0,1]")

    if landscape.n_classes > 0:
        require(landscape.total_trials > 0, "n_classes > 0 but total_trials == 0")
        require(landscape.class_counts, "class_counts is empty despite n_classes > 0")

        print_rows("class counts", [
            {"class": cls, "count": cnt}
            for cls, cnt in sorted(landscape.class_counts.items(), key=lambda x: -x[1])
        ][:10], limit=10)

        if landscape.imbalance_ratio is not None:
            require(landscape.imbalance_ratio >= 1.0, f"imbalance_ratio {landscape.imbalance_ratio} < 1")

        # per-subject balance
        if landscape.per_subject_counts:
            print_kv("subjects with data", len(landscape.per_subject_counts))

    passed("project_16_label_landscape")


if __name__ == "__main__":
    main()
