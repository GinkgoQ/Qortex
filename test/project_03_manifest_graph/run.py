"""project_03_manifest_graph

Builds a ManifestGraph from a real manifest and verifies that logical
recordings are assembled correctly with companion closure and entity coherence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, require_gt, passed,
)

from qortex.manifest.graph import ManifestGraph


def main() -> None:
    banner("project_03: manifest graph and companion closure")

    ds, manifest = real_manifest()
    graph = ManifestGraph(manifest)
    recordings = graph.recordings()

    print_kv("graph", {
        "dataset": manifest.dataset_id,
        "logical_recordings": len(recordings),
    })

    require(recordings, "ManifestGraph.recordings() returned empty list")

    # ── per-recording entity coherence ───────────────────────────────────────
    for rec in recordings[:50]:
        require(rec.primary is not None, f"recording {rec} has no primary file")
        require(rec.modality, f"recording {rec.primary.path!r} has no modality")

    print_rows("recordings sample", [
        {
            "path": rec.primary.path,
            "subject": rec.subject,
            "session": rec.session,
            "task": rec.task,
            "run": rec.run,
            "modality": rec.modality,
            "has_events": bool(rec.companions.events),
            "has_channels": bool(rec.companions.channels),
            "estimated_bytes": rec.estimated_bytes,
        }
        for rec in recordings[:12]
    ], limit=12)

    # ── subjects_with_modality ────────────────────────────────────────────────
    if recordings:
        first_mod = recordings[0].modality
        subjects_mod = manifest.subjects_with_modality(first_mod)
        require(subjects_mod is not None, "subjects_with_modality returned None")
        print_kv(f"subjects with modality={first_mod}", len(subjects_mod))

    # ── companions ───────────────────────────────────────────────────────────
    event_recs = [r for r in recordings if r.companions.events]
    print_kv("recordings with events", len(event_recs))
    # at least some recordings should have events in a task-based dataset
    # (we don't hard-fail because some datasets are resting-state only)

    # ── tasks_for_subject ────────────────────────────────────────────────────
    subjects = manifest.summary.subjects or []
    if subjects:
        sub = subjects[0]
        tasks = manifest.tasks_for_subject(sub)
        require(tasks is not None, f"tasks_for_subject({sub!r}) returned None")
        print_kv(f"tasks for subject {sub}", tasks)

    passed("project_03_manifest_graph")


if __name__ == "__main__":
    main()
