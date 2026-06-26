"""Scenario 20: DatasetSelector — fitness ranking without downloading.

Tests ResearchGoal + DatasetSelector using the local catalog to pre-filter
and then the OpenNeuro API for rich metadata scoring. Runs rank() directly
against a small set of known dataset IDs so the test does not require a
populated local catalog.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, print_kv, require  # noqa: E402

from qortex.inspect.selector import DatasetFitness, DatasetSelector, ResearchGoal


def main() -> None:
    print("\n--- ResearchGoal construction ---")
    goal = ResearchGoal(
        modality=None,          # don't restrict modality for this test dataset
        min_subjects=1,
        license_must_be_open=False,
        max_size_gb=100.0,
    )
    print_kv("goal", goal.describe())

    # ── DatasetFitness without API ─────────────────────────────────────────
    print("\n--- DatasetFitness scoring (no API) ---")
    selector = DatasetSelector()
    results = selector.rank(
        dataset_ids=[DATASET_ID],
        goal=goal,
        tier2_api=False,    # no API call — pure manifest-scoring pass
        tier3_events=False,
    )
    require(len(results) == 1, "Expected one result")
    fit = results[0]
    print_kv("dataset_id", fit.dataset_id)
    print_kv("total_score", fit.total_score)
    print_kv("grade", fit.grade)
    print_kv("is_viable", fit.is_viable)
    print_kv("hard_fail", fit.hard_fail)
    print_kv("dimensions", len(fit.dimensions))
    print_kv("recommendation", fit.recommendation)

    # ── DatasetFitness with API ────────────────────────────────────────────
    print("\n--- DatasetFitness scoring (with API) ---")
    try:
        results_api = selector.rank(
            dataset_ids=[DATASET_ID],
            goal=goal,
            tier2_api=True,
            tier3_events=False,
        )
        fit_api = results_api[0]
        print_kv("score_with_api", fit_api.total_score)
        print_kv("grade_with_api", fit_api.grade)
        print_kv("has_rich_info", fit_api.rich_info is not None)
        print("\nDimension breakdown:")
        for d in fit_api.dimensions:
            met_str = "✓" if d.met else "✗"
            print(f"  {met_str} {d.name:30s}  score={d.score:.2f}  val={d.value}")
    except Exception as exc:
        print(f"  API ranking raised: {exc}  (API may not be available in this environment)")

    # ── summary_line and report ────────────────────────────────────────────
    print("\n--- Summary line ---")
    print(fit.summary_line())
    print("\n--- Full report ---")
    print(fit.report())

    print("\nScenario 20 complete.")


if __name__ == "__main__":
    main()
