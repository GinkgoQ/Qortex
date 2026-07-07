#!/usr/bin/env python
"""Validate tutorial/project coverage documentation.

This project keeps the tutorial coverage map connected to real files.  It is
not a substitute for tutorial review; it catches stale links and missing
scenario references before docs drift away from implemented workflows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import banner, passed, print_kv, print_rows, require  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
COVERAGE = DOCS / "tutorials" / "coverage.md"

EXPECTED_TUTORIALS = [
    "t01-eeg-motor-imagery.md",
    "t02-eeg-connectivity.md",
    "t03-eeg-sleep-staging.md",
    "t04-eeg-seizure-detection.md",
    "t05-mri-dementia-baseline.md",
    "t06-mri-age-sex-qc.md",
    "t07-fmri-design-readiness.md",
    "t08-brain-tumour-segmentation.md",
]

EXPECTED_TEST_FILES = [
    "tests/test_cli_neuro_classic.py",
    "tests/test_console_error_mapping.py",
    "tests/test_console_streaming.py",
    "tests/test_convert_format_writers.py",
    "tests/test_neuroai_pipeline.py",
    "tests/test_neuroai_transforms.py",
    "tests/test_neuroclassic.py",
    "tests/test_neuroclassic_advanced.py",
    "tests/test_search_engine.py",
    "tests/test_stream_nifti.py",
]


def main() -> None:
    banner("project_22: tutorial and scenario coverage map")
    text = COVERAGE.read_text(encoding="utf-8")

    tutorial_links = set(re.findall(r"\]\((t\d\d-[^)]+\.md)\)", text))
    project_refs = set(re.findall(r"`(test/[^`]+)`", text))
    test_refs = set(re.findall(r"`(tests/[^`]+\.py)`", text))
    scenario_dirs = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "test").iterdir()
        if p.is_dir() and (p / "run.py").exists()
    )

    for name in EXPECTED_TUTORIALS:
        path = DOCS / "tutorials" / name
        require(path.exists(), f"tutorial file missing: {path}")
        require(name in tutorial_links, f"tutorial missing from coverage map: {name}")

    for ref in scenario_dirs:
        path = ROOT / ref
        require(path.exists(), f"scenario reference missing on disk: {ref}")
        require(ref in project_refs, f"scenario missing from coverage map: {ref}")
        require((path / "run.py").exists(), f"scenario has no run.py: {ref}")

    for ref in EXPECTED_TEST_FILES:
        path = ROOT / ref
        require(path.exists(), f"test reference missing on disk: {ref}")
        require(ref in test_refs, f"test file missing from coverage map: {ref}")

    rows = [
        {
            "kind": "tutorials",
            "expected": len(EXPECTED_TUTORIALS),
            "covered": len(EXPECTED_TUTORIALS),
        },
        {
            "kind": "scenario_projects",
            "expected": len(scenario_dirs),
            "covered": len(scenario_dirs),
        },
        {
            "kind": "focused_pytests",
            "expected": len(EXPECTED_TEST_FILES),
            "covered": len(EXPECTED_TEST_FILES),
        },
    ]
    print_kv("coverage document", str(COVERAGE.relative_to(ROOT)))
    print_rows("coverage counts", rows, limit=10)
    passed("project_22_tutorial_coverage")


if __name__ == "__main__":
    main()
