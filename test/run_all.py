"""Run all staged Qortex scenario projects without pytest."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    test_dirs = sorted(
        (
            path for path in root.iterdir()
            if path.is_dir() and path.name.split("_", 1)[0].isdigit()
        ),
        key=lambda path: (int(path.name.split("_", 1)[0]), path.name),
    )
    executed = 0
    with tempfile.TemporaryDirectory(prefix="qortex-real-suite-") as tmp:
        env = dict(os.environ)
        env["QORTEX_REAL_METADATA_ROOT"] = str(Path(tmp) / "metadata")
        env["QORTEX_REAL_ARTIFACT_ROOT"] = str(Path(tmp) / "artifacts")
        for directory in test_dirs:
            script = directory / "run.py"
            if not script.exists():
                continue
            executed += 1
            print(f"\n##### running {directory.name} #####", flush=True)
            subprocess.run([sys.executable, str(script)], check=True, env=env)
    print(f"\nall {executed} staged Qortex scenario projects passed")


if __name__ == "__main__":
    main()
