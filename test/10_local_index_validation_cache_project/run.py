from __future__ import annotations

import sys
import shutil
from pathlib import Path

import qortex
from qortex.core.config import QortexConfig
from qortex.core.entities import ValidationIssue, ValidationReport
from qortex.indexing import index_local_bids
from qortex.validation import ValidationCache, diff_validation_reports, validate_bids

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import artifact_dir, print_kv, print_rows, real_metadata_root, require  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        index_report = index_local_bids(root, manifest=manifest, use_pybids=False)

        validator_available = shutil.which("bids-validator") is not None
        if validator_available:
            report = validate_bids(
                root,
                timeout_s=120.0,
                use_cache=True,
                refresh_cache=True,
            )
            validation_mode = "official bids-validator"
        else:
            report = ValidationReport(
                dataset_path=str(root),
                valid=False,
                issues=[
                    ValidationIssue(
                        severity="warning",
                        code="BIDS_VALIDATOR_NOT_INSTALLED",
                        path=str(root),
                        message=(
                            "Official BIDS Validator CLI is not installed on PATH; "
                            "Qortex did not fabricate validation results."
                        ),
                    )
                ],
            )
            validation_mode = "not run: bids-validator missing"

        out_dir = artifact_dir(root, "project10_validation")
        cache = ValidationCache(QortexConfig(cache_dir=out_dir / "cache"))
        key = cache.key(
            root,
            executable="bids-validator",
            config_path=None,
            ignore_warnings=False,
            ignore_nifti_headers=False,
        )
        cache.put(key, report)
        cached = cache.get(key)
        md_path = out_dir / "real_validation.md"
        html_path = out_dir / "real_validation.html"
        json_path = out_dir / "real_validation.json"
        report.to_markdown(md_path)
        report.to_html(html_path)
        report.to_json(json_path)

        before = ValidationReport(dataset_path="before", valid=False, issues=report.issues)
        after = ValidationReport(dataset_path="after", valid=report.valid, issues=report.issues[:1])
        diff = diff_validation_reports(before, after)

        print_kv(
            "PROJECT 10: real local metadata index and validation report artifacts",
            {
                "dataset": ds.dataset_id,
                "indexed files": index_report.n_files,
                "missing remote": index_report.n_missing,
                "extra local": index_report.n_extra,
                "validation mode": validation_mode,
                "cached report": cached is not None,
                "official validator": validator_available,
                "validation score": report.score,
                "diff resolved issues": diff.n_resolved,
            },
        )
        print(index_report.summary())
        print(report.summary())
        print_rows(
            "Real validation report exports",
            [
                {"path": str(path), "bytes": path.stat().st_size}
                for path in [json_path, md_path, html_path]
            ],
        )

        require(index_report.n_files > 0, "local index found no real downloaded metadata files")
        require(cached is not None, "validation report was not cached")
        require(
            validator_available or report.warnings[0].code == "BIDS_VALIDATOR_NOT_INSTALLED",
            "missing validator was not reported explicitly",
        )
        require(json_path.exists() and md_path.exists() and html_path.exists(), "validation exports missing")
        require(qortex.ValidationReport is not None, "public ValidationReport export is missing")
    finally:
        tmp.cleanup()

    print("RESULT: real index/validation project passed")


if __name__ == "__main__":
    main()
