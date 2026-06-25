from __future__ import annotations

import sys
from pathlib import Path

import qortex
from qortex.eda.events import summarize_events
from qortex.eda.report import EDAEngine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import artifact_dir, print_kv, print_rows, real_metadata_root, require  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        summaries = summarize_events(manifest, root)
        report = EDAEngine(manifest).run(local_path=root)
        html_path = artifact_dir(root, "project5_eda") / "qortex_real_openneuro_eda.html"
        report.to_html(html_path)

        print_kv(
            "PROJECT 5: real downloaded OpenNeuro metadata EDA",
            {
                "dataset": ds.dataset_id,
                "snapshot": manifest.snapshot,
                "local metadata root": root,
                "event files summarized": len(summaries),
                "quality bids score": report.quality.bids_score,
                "quality ml score": report.quality.ml_readiness_score,
                "html report bytes": html_path.stat().st_size,
            },
        )
        print_rows(
            "Real event-label distributions",
            [
                {
                    "path": item.path,
                    "rows": item.n_events,
                    "label_column": item.label_column,
                    "classes": item.n_classes,
                    "counts": item.label_counts,
                    "imbalance": item.imbalance_ratio,
                }
                for item in summaries
            ],
            limit=12,
        )

        require(summaries, "no real downloaded events files were summarized")
        require(any(item.label_column for item in summaries), "no real label column was detected")
        require(html_path.exists() and html_path.stat().st_size > 1000, "real EDA HTML report was not written")
        require("Local Events and Labels" in report.html, "EDA HTML is missing the local labels section")
        require(qortex.EventLabelSummary is not None, "public EventLabelSummary export is missing")
    finally:
        tmp.cleanup()

    print("RESULT: real EDA project passed")


if __name__ == "__main__":
    main()
