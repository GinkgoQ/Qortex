from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from qortex.core.entities import SelectionSpec
from qortex.plan.planner import DownloadPlanner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import primary_recording_with_events, print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    _ds, manifest = real_manifest()
    recording = primary_recording_with_events(manifest)
    with tempfile.TemporaryDirectory() as tmp:
        plan = DownloadPlanner(check_disk_space=False).plan(
            manifest,
            SelectionSpec(
                include=[recording.primary.path],
                with_companions=True,
            ),
            Path(tmp) / manifest.dataset_id,
        )

    print_kv(
        "PROJECT 2: real primary-file selection with companion closure",
        {
            "selected primary": recording.primary.path,
            "selected files": plan.n_files,
            "estimated bytes": plan.estimated_bytes,
            "warnings": len(plan.warnings),
        },
    )
    print(plan.summary())
    print_rows(
        "Real selected files and reasons",
        [
            {
                "path": file.path,
                "size": file.size,
                "reason": "; ".join(reason.reason for reason in plan.selection_reasons.get(file.path, [])),
            }
            for file in plan.files
        ],
        limit=16,
    )

    paths = {file.path for file in plan.files}
    require(recording.primary.path in paths, "real selected primary missing from plan")
    require(recording.companions.events.path in paths, "real events companion missing from plan")
    require(any(file.filename == "dataset_description.json" for file in plan.files), "dataset description missing from plan")
    require(plan.estimated_bytes > 0, "real plan estimated size is zero")

    print("RESULT: real selection project passed")


if __name__ == "__main__":
    main()
