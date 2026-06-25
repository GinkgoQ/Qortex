"""Dataset materialization modes.

Supports three strategies for making cached files available at a target path:
  - copy      : full copy (safe, portable, uses 2× disk)
  - hardlink  : hard link (saves disk, same filesystem required)
  - symlink   : symbolic link (saves disk, cross-filesystem, fragile on Windows)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

MaterializeMode = Literal["copy", "hardlink", "symlink"]


def materialize_file(
    source: Path,
    destination: Path,
    mode: MaterializeMode = "hardlink",
) -> None:
    """Make *source* available at *destination* using the given strategy."""
    if destination.exists() or destination.is_symlink():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            # Cross-filesystem or Windows restriction — fall back to copy
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def materialize_dataset(
    source_dir: Path,
    target_dir: Path,
    mode: MaterializeMode = "hardlink",
) -> int:
    """Materialise all files from *source_dir* into *target_dir*.

    Returns the number of files linked/copied.
    """
    count = 0
    for src in source_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(source_dir)
            dst = target_dir / rel
            materialize_file(src, dst, mode)
            count += 1
    return count
