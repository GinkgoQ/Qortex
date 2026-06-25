from __future__ import annotations

import sys
from pathlib import Path

import qortex
from qortex.check import compute_readiness

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import print_kv, print_rows, real_metadata_root, require  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        report = compute_readiness(
            manifest,
            local_path=root,
            conversion_target="sklearn",
            inspect_loaders=False,
        )

        print_kv(
            "PROJECT 7: real dataset readiness analysis",
            {
                "dataset": ds.dataset_id,
                "snapshot": manifest.snapshot,
                "score": report.score,
                "recordings": report.n_recordings,
                "loadable": report.n_loadable,
                "event complete": f"{report.n_event_complete}/{report.n_recordings}",
                "label ready": f"{report.n_label_ready}/{report.n_recordings}",
                "can download": report.can_download,
                "can convert": report.can_convert,
            },
        )
        print(report.summary())
        print_rows(
            "Real readiness findings",
            [
                {
                    "severity": finding.severity,
                    "code": finding.code,
                    "path": finding.path,
                    "message": finding.message,
                }
                for finding in report.findings
            ],
            limit=16,
        )
        print_rows(
            "Readiness score accounting",
            [
                {
                    "metric": "logical recordings",
                    "count": report.n_recordings,
                    "interpretation": "all primary modality records represented in the manifest graph",
                },
                {
                    "metric": "event-complete recordings",
                    "count": report.n_event_complete,
                    "gap": report.n_recordings - report.n_event_complete,
                    "interpretation": "anatomical/non-task records usually do not carry events",
                },
                {
                    "metric": "label-ready recordings",
                    "count": report.n_label_ready,
                    "gap": report.n_recordings - report.n_label_ready,
                    "interpretation": "local events tables with detected label columns",
                },
                {
                    "metric": "estimated bytes",
                    "count": report.estimated_bytes,
                    "interpretation": "primary recordings plus required companions",
                },
            ],
        )

        require(report.n_recordings > 0, "readiness did not find real logical recordings")
        require(report.n_loadable > 0, "readiness found no downloadable real recordings")
        require(report.can_download is True, "real dataset should be downloadable")
        require(
            report.n_recordings - report.n_event_complete >= 0,
            "readiness event accounting is inconsistent",
        )
        require(qortex.ReadinessReport is not None, "public ReadinessReport export is missing")
    finally:
        tmp.cleanup()

    print("RESULT: real readiness project passed")


if __name__ == "__main__":
    main()
