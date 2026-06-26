"""Run the Qortex scenario suite.

Picks up directories whose name starts with a digit sequence followed by an
underscore (legacy ``NN_name`` form) or with ``project_NN_`` (current form).
Directories are run in numeric order.

Set environment variables to reuse a shared metadata download across projects:
  QORTEX_REAL_METADATA_ROOT   path where metadata is cached between projects
  QORTEX_REAL_ARTIFACT_ROOT   path where converted artifacts are stored
  QORTEX_REAL_TEST_DATASET    OpenNeuro dataset ID (default: ds000001)
  QORTEX_REAL_TEST_SNAPSHOT   snapshot tag (default: latest)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_LEGACY_RE = re.compile(r"^(\d+)_")
_PROJECT_RE = re.compile(r"^project_(\d+)_")


def _sort_key(path: Path) -> tuple[int, str]:
    m = _PROJECT_RE.match(path.name) or _LEGACY_RE.match(path.name)
    return (int(m.group(1)) if m else 9999, path.name)


def _is_scenario(path: Path) -> bool:
    if not path.is_dir():
        return False
    if _PROJECT_RE.match(path.name) or _LEGACY_RE.match(path.name):
        return (path / "run.py").exists()
    return False


def main() -> None:
    root = Path(__file__).resolve().parent
    scenarios = sorted(
        (p for p in root.iterdir() if _is_scenario(p)),
        key=_sort_key,
    )

    if not scenarios:
        print("No scenario directories found.")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="qortex-suite-") as tmp:
        env = dict(os.environ)
        env.setdefault("QORTEX_REAL_METADATA_ROOT", str(Path(tmp) / "metadata"))
        env.setdefault("QORTEX_REAL_ARTIFACT_ROOT", str(Path(tmp) / "artifacts"))

        passed = 0
        failed: list[str] = []

        for scenario in scenarios:
            print(f"\n{'=' * 60}", flush=True)
            print(f"  {scenario.name}", flush=True)
            print(f"{'=' * 60}", flush=True)
            result = subprocess.run(
                [sys.executable, str(scenario / "run.py")],
                env=env,
            )
            if result.returncode == 0:
                passed += 1
            else:
                failed.append(scenario.name)
                print(f"FAILED: {scenario.name}", file=sys.stderr)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {len(failed)} failed")
    if failed:
        print("Failed scenarios:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)
    else:
        print(f"All {passed} scenarios passed.")


if __name__ == "__main__":
    main()
