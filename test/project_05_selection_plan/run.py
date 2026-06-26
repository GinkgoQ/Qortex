"""project_05_selection_plan

Builds a download plan from manifest filtering and companion expansion,
then exercises dry-run plan inspection without any actual file transfer.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, require_gt, passed,
)

from qortex.core.entities import SelectionSpec
from qortex.plan.planner import DownloadPlanner


def main() -> None:
    banner("project_05: selection planning and companion expansion")

    ds, manifest = real_manifest()
    subjects = manifest.summary.subjects or []
    require(subjects, "manifest has no subjects")
    print_kv("subjects available", len(subjects))

    test_subjects = subjects[:2]
    print_kv("test subjects", test_subjects)

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "download"
        target.mkdir()
        planner = DownloadPlanner(check_disk_space=False)

        # ── metadata-only plan ────────────────────────────────────────────────
        meta_spec = SelectionSpec(metadata_only=True)
        meta_plan = planner.plan(manifest, meta_spec, target)

        print_kv("metadata-only plan", {
            "n_files": meta_plan.n_files,
            "estimated_bytes": meta_plan.estimated_bytes,
            "warnings": len(meta_plan.warnings),
        })

        require(meta_plan.n_files > 0, "metadata-only plan has 0 files")
        require(meta_plan.dataset_id == manifest.dataset_id, "plan.dataset_id mismatch")

        # ── subject-filtered plan ─────────────────────────────────────────────
        subject_spec = SelectionSpec(subjects=test_subjects, metadata_only=True)
        subject_plan = planner.plan(manifest, subject_spec, target)

        print_kv("subject-filtered plan", {
            "subjects": test_subjects,
            "n_files": subject_plan.n_files,
            "estimated_bytes": subject_plan.estimated_bytes,
        })

        require(subject_plan.n_files > 0, f"subject plan for {test_subjects} has 0 files")

        # verify plan only contains files for the selected subjects (raw values)
        raw_subs = {s.removeprefix("sub-") for s in test_subjects}
        for fr in subject_plan.files:
            if fr.subject:
                require(
                    fr.subject in raw_subs,
                    f"plan contains file for subject {fr.subject!r} outside {raw_subs}",
                )

        print_rows("planned files sample", [
            {"path": fr.path, "subject": fr.subject, "size": fr.size, "modality": fr.modality}
            for fr in subject_plan.files[:10]
        ], limit=10)

        # ── plan summary + explain ────────────────────────────────────────────
        summary_text = subject_plan.summary()
        require(summary_text, "plan.summary() returned empty string")
        print_kv("plan summary", summary_text)

        explain_text = subject_plan.explain(limit=5)
        require(isinstance(explain_text, str), "plan.explain() did not return a string")

        # ── full plan should have at least as many files ───────────────────────
        full_spec = SelectionSpec(metadata_only=True)
        full_plan = planner.plan(manifest, full_spec, target)
        require(
            full_plan.n_files >= subject_plan.n_files,
            "full metadata plan has fewer files than subject-filtered plan",
        )

    passed("project_05_selection_plan")


if __name__ == "__main__":
    main()
