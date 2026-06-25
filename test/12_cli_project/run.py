from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import DATASET_ID, print_kv, require  # noqa: E402


def main() -> None:
    command = shutil.which("qortex")
    require(command is not None, "qortex console script is not installed on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        env["QORTEX_CACHE_DIR"] = str(Path(tmp) / "cache")
        help_result = subprocess.run(
            [command, "--help"],
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
        metadata_result = subprocess.run(
            [command, "metadata", DATASET_ID, "--limit", "8"],
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )

    print_kv(
        "PROJECT 12: installed CLI against real OpenNeuro metadata",
        {
            "command": command,
            "help return code": help_result.returncode,
            "metadata return code": metadata_result.returncode,
            "metadata output lines": len(metadata_result.stdout.splitlines()),
        },
    )
    print("qortex metadata output:")
    print(metadata_result.stdout)

    require("Qortex" in help_result.stdout, "CLI help did not identify Qortex")
    require("dataset_description.json" in metadata_result.stdout, "CLI metadata command did not list real dataset metadata")

    print("RESULT: real CLI project passed")


if __name__ == "__main__":
    main()
