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

    selector = DatasetSelector(goal, max_datasets=20)

    # ── rank (returns all, including hard-fails) ──────────────────────────────
    ranked = selector.rank()
    print_kv("rank() results", len(ranked))
    require(isinstance(ranked, list), "rank() did not return a list")

    if ranked:
        for fit in ranked:
            require(isinstance(fit, DatasetFitness), f"rank() item is {type(fit).__name__}")
            require(0.0 <= fit.total_score <= 1.0, f"total_score {fit.total_score} out of [0,1]")
            require(fit.grade in {"A", "B", "C", "D", "F"}, f"unexpected grade {fit.grade!r}")

        # highest score first
        scores = [f.total_score for f in ranked]
        require(scores == sorted(scores, reverse=True), "rank() not sorted by score descending")

        top = ranked[0]
        print_kv("top result", {
            "dataset_id": top.dataset_id,
            "total_score": f"{top.total_score:.3f}",
            "grade": top.grade,
            "hard_fail": top.hard_fail,
            "summary": top.summary_line(),
        })

        # dimension breakdown on top result
        print_rows("top result dimensions", [
            {
                "dimension": name,
                "score": f"{dim.score:.2f}",
                "weight": dim.weight,
                "met": dim.met,
                "value": dim.value,
                "target": dim.target,
            }
            for name, dim in top.dimensions.items()
        ])

    # ── find (exclude hard-fails) ─────────────────────────────────────────────
    viable = selector.find(include_failed=False)
    print_kv("find() viable results", len(viable))
    for fit in viable:
        require(not fit.hard_fail, f"find() returned hard-fail result: {fit.dataset_id}")

    # ── find with include_failed ──────────────────────────────────────────────
    all_results = selector.find(include_failed=True)
    require(len(all_results) >= len(viable), "include_failed=True returned fewer results than False")
    print_kv("find(include_failed=True)", len(all_results))

    # ── report on first viable ────────────────────────────────────────────────
    if viable:
        report = viable[0].report()
        require(isinstance(report, str) and report.strip(), "report() returned empty string")
        print_kv("report (first 400 chars)", report[:400])

    passed("project_18_dataset_selector")


if __name__ == "__main__":
    main()
