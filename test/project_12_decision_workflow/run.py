"""project_12_decision_workflow

Exercises the decision-workflow functions: doctor(), minimum_plan(),
can_train(), and first_batch() against a real manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, passed,
)

from qortex.decision import doctor, minimum_plan, can_train, first_batch


def main() -> None:
    banner("project_12: decision workflow")

    ds, manifest = real_manifest()

    # ── doctor ────────────────────────────────────────────────────────────────
    health = doctor(manifest)
    print_kv("doctor", {
        "status": health.status,
        "n_recordings": health.n_recordings,
        "n_event_complete": health.n_event_complete,
        "n_label_ready": health.n_label_ready,
        "can_download": health.can_download,
        "can_convert": health.can_convert,
        "next_actions": len(health.next_actions),
        "findings": len(health.findings),
    })
    require(health.status in {"possible", "uncertain", "not_possible"},
            f"unexpected doctor status {health.status!r}")
    require(health.n_recordings >= 0, "n_recordings < 0")
    require(isinstance(health.next_actions, list), "next_actions is not a list")

    doctor_text = health.to_text()
    require(isinstance(doctor_text, str) and doctor_text.strip(), "to_text() returned empty")
    print_kv("doctor text (first 300 chars)", doctor_text[:300])

    # ── minimum_plan ──────────────────────────────────────────────────────────
    for goal in ("metadata", "label-check"):
        min_report = minimum_plan(manifest, goal=goal)
        print_kv(f"minimum_plan(goal={goal!r})", {
            "status": min_report.status,
            "n_files": min_report.plan.n_files,
            "estimated_bytes": min_report.plan.estimated_bytes,
            "reason": min_report.reason[:80],
            "next_command": min_report.next_command,
        })
        require(min_report.status in {"possible", "uncertain", "not_possible"},
                f"unexpected status {min_report.status!r}")
        require(min_report.plan.n_files > 0, f"minimum_plan({goal}) has 0 files")

    # ── can_train ─────────────────────────────────────────────────────────────
    ct = can_train(manifest)
    print_kv("can_train", {
        "status": ct.status,
        "label_status": ct.label_status,
        "n_subjects": ct.n_subjects,
        "n_recordings": ct.n_recordings,
        "n_label_ready": ct.n_label_ready,
        "suggested_split": ct.suggested_split,
        "leakage_risks": len(ct.leakage_risks),
    })
    require(ct.status in {"possible", "uncertain", "not_possible"},
            f"unexpected can_train status {ct.status!r}")
    require(ct.label_status in {"confirmed", "candidate", "missing"},
            f"unexpected label_status {ct.label_status!r}")
    require(ct.n_subjects >= 0, "n_subjects < 0")

    ct_text = ct.to_text()
    require(isinstance(ct_text, str) and ct_text.strip(), "can_train to_text() returned empty")

    # ── first_batch (from manifest — no local data) ───────────────────────────
    fb = first_batch(manifest)
    print_kv("first_batch", {
        "status": fb.status,
        "source": fb.source,
        "n_rows": fb.n_rows,
        "message": (fb.message or "")[:80],
    })
    require(fb.status in {"possible", "uncertain", "not_possible"},
            f"unexpected first_batch status {fb.status!r}")

    passed("project_12_decision_workflow")


if __name__ == "__main__":
    main()
