"""project_13_cli

Exercises the Qortex CLI by invoking key subcommands as subprocesses and
verifying that their exit codes and output are sane.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, require, passed, DATASET_ID,
)


def _cli(*args: str, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["qortex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


def main() -> None:
    banner("project_13: CLI commands")

    # ── help (always available, no network) ──────────────────────────────────
    result = _cli("--help")
    require(result.returncode == 0, f"qortex --help failed: {result.stderr}")
    require("search" in result.stdout or "Usage" in result.stdout,
            "qortex --help did not show expected output")
    print_kv("--help exit code", result.returncode)

    # ── search (live, small limit) ────────────────────────────────────────────
    result = _cli("search", "eeg", "--limit", "3")
    require(result.returncode == 0, f"qortex search failed:\n{result.stderr}")
    output = result.stdout + result.stderr
    require(output.strip(), "search produced no output")
    print_kv("search output (first 300 chars)", output[:300])

    # ── inspect (fetches real manifest from API) ──────────────────────────────
    result = _cli("inspect", DATASET_ID)
    require(result.returncode == 0, f"inspect failed:\n{result.stderr}")
    output = result.stdout
    require(DATASET_ID in output, f"inspect output missing dataset ID {DATASET_ID!r}")
    require("Files" in output or "Subjects" in output,
            "inspect output missing Files/Subjects line")
    print_kv("inspect output (first 400 chars)", output[:400])

    # ── doctor ────────────────────────────────────────────────────────────────
    result = _cli("doctor", DATASET_ID)
    require(result.returncode == 0, f"doctor failed:\n{result.stderr}")
    output = result.stdout + result.stderr
    require(output.strip(), "doctor produced no output")
    print_kv("doctor output (first 300 chars)", output[:300])

    # ── can-train ────────────────────────────────────────────────────────────
    result = _cli("can-train", DATASET_ID)
    require(result.returncode == 0, f"can-train failed:\n{result.stderr}")
    output = result.stdout + result.stderr
    require(output.strip(), "can-train produced no output")
    print_kv("can-train output (first 300 chars)", output[:300])

    # ── plan (computes plan without downloading) ──────────────────────────────
    result = _cli("plan", DATASET_ID)
    require(result.returncode == 0, f"plan failed:\n{result.stderr}")
    output = result.stdout
    require("Files" in output or "plan" in output.lower(),
            "plan output missing expected content")
    print_kv("plan output (first 300 chars)", output[:300])

    passed("project_13_cli")


if __name__ == "__main__":
    main()
