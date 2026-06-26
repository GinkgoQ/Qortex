"""project_08_readiness_report

Runs compute_readiness() against a real manifest to verify that score
components are meaningful and findings are correctly categorised.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_manifest,
    require, require_gt, passed,
)

from qortex.check.readiness import compute_readiness


def main() -> None:
    banner("project_08: dataset readiness scoring")

    ds, manifest = real_manifest()
    report = compute_readiness(manifest)

    print_kv("readiness", {
        "dataset": report.dataset_id,
        "n_recordings": report.n_recordings,
        "n_loadable": report.n_loadable,
        "n_event_complete": report.n_event_complete,
        "n_label_ready": report.n_label_ready,
        "score": f"{report.score:.1f}/100",
        "can_download": report.can_download,
        "can_convert": report.can_convert,
        "findings": len(report.findings),
    })

    require(0.0 <= report.score <= 100.0, f"score {report.score} out of [0,100]")
    require_gt(report.n_recordings, 0, "n_recordings")
    require(isinstance(report.findings, list), "findings is not a list")

    # categorise findings
    errors = [f for f in report.findings if f.severity == "error"]
    warnings = [f for f in report.findings if f.severity == "warning"]
    infos = [f for f in report.findings if f.severity == "info"]

    print_kv("findings by severity", {
        "error": len(errors),
        "warning": len(warnings),
        "info": len(infos),
    })

    if report.findings:
        print_rows("findings sample", [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message[:80],
                "path": (f.path or "")[:50],
            }
            for f in report.findings[:8]
        ], limit=8)

    # verify finding fields are well-formed
    for finding in report.findings:
        require(finding.severity in {"info", "warning", "error"}, f"bad severity {finding.severity!r}")
        require(finding.code, f"finding.code is empty for {finding!r}")
        require(finding.message, f"finding.message is empty for {finding!r}")

    # summary string
    summary_text = report.summary()
    require(isinstance(summary_text, str) and summary_text.strip(), "summary() returned empty")
    print_kv("summary (first 300 chars)", summary_text[:300])

    passed("project_08_readiness_report")


if __name__ == "__main__":
    main()
