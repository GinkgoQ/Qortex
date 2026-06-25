from __future__ import annotations

import sys
from pathlib import Path

import qortex
from qortex.manifest.graph import ManifestGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import print_kv, print_rows, real_manifest, require  # noqa: E402


def main() -> None:
    ds, manifest = real_manifest()
    graph = ManifestGraph(manifest)
    recordings = graph.recordings()

    print_kv(
        "PROJECT 1: real OpenNeuro manifest and semantic recording graph",
        {
            "dataset": ds.dataset_id,
            "snapshot": manifest.snapshot,
            "doi": manifest.doi,
            "files": manifest.summary.file_count,
            "subjects": manifest.summary.n_subjects,
            "sessions": len(manifest.summary.sessions),
            "tasks": ", ".join(manifest.summary.tasks[:8]),
            "modalities": ", ".join(manifest.summary.modalities),
            "logical recordings": len(recordings),
        },
    )
    print_rows(
        "First real logical recordings",
        [
            {
                "primary": rec.primary.path,
                "modality": rec.modality,
                "subject": rec.subject,
                "task": rec.task,
                "events": bool(rec.companions.events),
                "channels": bool(rec.companions.channels),
                "bytes": rec.estimated_bytes,
            }
            for rec in recordings[:12]
        ],
        limit=12,
    )

    require(manifest.summary.file_count > 0, "real manifest returned no files")
    require(manifest.summary.n_subjects > 0, "real manifest returned no subjects")
    require(recordings, "real manifest graph produced no logical recordings")
    require(any(rec.primary.urls for rec in recordings), "recordings have no real download URLs")
    require(qortex.FileRecord is not None, "public FileRecord export is missing")

    print("RESULT: real manifest project passed")


if __name__ == "__main__":
    main()
