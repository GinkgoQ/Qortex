"""project_18_dataset_selector

Uses DatasetSelector with a ResearchGoal to rank EEG datasets from the
catalog and verify that fitness scores and grade assignments are correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, require, require_gt, passed,
)

from qortex.inspect.selector import DatasetSelector, ResearchGoal, DatasetFitness


def main() -> None:
    banner("project_18: dataset selection and fitness ranking")

    goal = ResearchGoal(
        modality="eeg",
        task_keywords=["motor", "imagery"],
        min_subjects=10,
        min_n_classes=2,
        max_imbalance_ratio=5.0,
        max_size_gb=50.0,
        license_must_be_open=True,
        min_downloads=5,
    )

    print_kv("research goal", {
        "modality": goal.modality,
        "task_keywords": goal.task_keywords,
        "min_subjects": goal.min_subjects,
        "min_n_classes": goal.min_n_classes,
        "max_size_gb": goal.max_size_gb,
        "license_must_be_open": goal.license_must_be_open,
    })

    # DatasetSelector takes no goal in __init__; goal is passed to rank()/find()
    selector = DatasetSelector()

    # ── find (ranked list, excluding hard-fails) ──────────────────────────────
    viable = selector.find(goal, include_failed=False, limit=10, tier2_api=False)
    print_kv("find() viable results", len(viable))
    require(isinstance(viable, list), "find() must return a list")
    for fit in viable:
        require(isinstance(fit, DatasetFitness), f"find() item is {type(fit).__name__}")
        require(not fit.hard_fail, f"find() returned hard-fail result: {fit.dataset_id}")

    # ── find with include_failed ──────────────────────────────────────────────
    all_results = selector.find(goal, include_failed=True, limit=10, tier2_api=False)
    require(isinstance(all_results, list), "find(include_failed=True) must return a list")
    require(len(all_results) >= len(viable), "include_failed=True returned fewer results than False")
    print_kv("find(include_failed=True)", len(all_results))

    # ── rank against known dataset IDs ────────────────────────────────────────
    test_ids = ["ds000001"]
    ranked = selector.rank(test_ids, goal, tier2_api=False)
    print_kv("rank() results", len(ranked))
    require(isinstance(ranked, list), "rank() must return a list")

    if ranked:
        for fit in ranked:
            require(isinstance(fit, DatasetFitness), f"rank() item is {type(fit).__name__}")
            require(0.0 <= fit.total_score <= 100.0, f"total_score {fit.total_score} out of [0,100]")
            require(fit.grade in {"A", "B", "C", "D", "F"}, f"unexpected grade {fit.grade!r}")

        top = ranked[0]
        print_kv("top result", {
            "dataset_id": top.dataset_id,
            "total_score": f"{top.total_score:.3f}",
            "grade": top.grade,
            "hard_fail": top.hard_fail,
        })

        if hasattr(top, "summary_line"):
            print_kv("summary_line", top.summary_line())

        if hasattr(top, "report"):
            report = top.report()
            require(isinstance(report, str) and report.strip(), "report() returned empty string")
            print_kv("report (first 300 chars)", report[:300])

        if hasattr(top, "dimensions") and top.dimensions:
            print_rows("dimensions", [
                {
                    "dimension": dim.name if hasattr(dim, "name") else str(dim),
                    "score": f"{dim.score:.2f}" if hasattr(dim, "score") else "?",
                    "met": dim.met if hasattr(dim, "met") else "?",
                }
                for dim in top.dimensions[:8]
            ])

    passed("project_18_dataset_selector")


if __name__ == "__main__":
    main()
