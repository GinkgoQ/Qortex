"""project_09_eda_report

Runs the EDA engine against a real downloaded metadata tree and verifies that
the quality scores, modality breakdown, and HTML output are produced correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, print_rows, real_metadata_root, artifact_dir,
    require, passed,
)

from qortex.eda.report import EDAEngine


def main() -> None:
    banner("project_09: EDA / QC report")

    ctx, ds, root = real_metadata_root()
    try:
        out_dir = artifact_dir(root, "eda_report")
        manifest = ds.manifest()

        engine = EDAEngine(manifest)
        report = engine.run(local_path=root)

        print_kv("EDA report", {
            "dataset_id": report.dataset_id,
            "snapshot": report.snapshot,
            "n_modalities": len(report.modality_summaries),
            "n_event_summaries": len(report.event_summaries),
            "has_html": report.html is not None,
        })

        require(report.dataset_id == manifest.dataset_id, "report.dataset_id mismatch")

        # quality scores
        q = report.quality
        if q is not None:
            print_kv("quality", {
                "bids_score": f"{q.bids_score:.1f}/100",
                "ml_readiness_score": f"{q.ml_readiness_score:.1f}/100",
                "loadability_score": f"{q.loadability_score:.1f}/100",
                "n_issues": len(q.issues),
                "n_risks": len(q.risks),
            })
            require(0.0 <= q.bids_score <= 100.0, f"bids_score {q.bids_score} out of [0,100]")

        # modality breakdown
        if report.modality_summaries:
            print_rows("modality breakdown", [
                {
                    "modality": m,
                    "n_files": ms.n_files,
                    "n_subjects": ms.n_subjects,
                    "total_size_mb": f"{ms.total_size / 1e6:.1f}",
                }
                for m, ms in sorted(report.modality_summaries.items())
            ])
            for m, ms in report.modality_summaries.items():
                require(ms.n_files >= 0, f"modality {m} n_files is negative")

        # local events summaries
        if report.event_summaries:
            print_rows("event label summaries", [
                {
                    "path": es.path,
                    "n_events": es.n_events,
                    "label_column": es.label_column or "n/a",
                    "n_classes": es.n_classes,
                }
                for es in report.event_summaries[:5]
            ], limit=5)

        # HTML output
        require(report.html is not None, "report.html is None — EDAEngine did not render HTML")
        html_path = out_dir / "eda_report.html"
        report.to_html(html_path)
        require(html_path.exists(), f"HTML report not written to {html_path}")
        require(html_path.stat().st_size > 200, "HTML report is too small (likely empty)")
        print_kv("HTML report", str(html_path))

    finally:
        ctx.cleanup()

    passed("project_09_eda_report")


if __name__ == "__main__":
    main()
