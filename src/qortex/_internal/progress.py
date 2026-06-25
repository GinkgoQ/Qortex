"""Unified progress reporting for terminal and Jupyter environments.

All user-visible output goes through this module so the rest of the
codebase never imports tqdm directly.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from tqdm.auto import tqdm as _tqdm


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _stdout_is_utf8() -> bool:
    if hasattr(sys.stdout, "encoding"):
        return sys.stdout.encoding.lower() in ("utf-8", "utf8")
    return False


_UTF8 = _stdout_is_utf8()


def _u(text: str, emoji: str = "", fallback: str = "") -> str:
    """Return emoji-prefixed text when stdout supports Unicode."""
    if _UTF8:
        return f"{emoji} {text}" if emoji else text
    return f"{fallback} {text}" if fallback else text


def msg(text: str, emoji: str = "", fallback: str = "") -> None:
    """Write a tqdm-safe status line (won't break progress bars)."""
    _tqdm.write(_u(text, emoji=emoji, fallback=fallback))


# ── Progress bar factory ──────────────────────────────────────────────────────

def bytes_bar(
    total: int,
    desc: str = "Overall",
    *,
    leave: bool = True,
    position: int | None = None,
) -> _tqdm:
    """Return a bytes-unit tqdm bar."""
    return _tqdm(
        total=total,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        leave=leave,
        position=position,
    )


def file_bar(
    total: int | None,
    desc: str,
    initial: int = 0,
    *,
    leave: bool = False,
    position: int | None = None,
) -> _tqdm:
    """Return a per-file bytes-unit tqdm bar."""
    return _tqdm(
        total=total,
        desc=desc,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=leave,
        position=position,
    )


def count_bar(total: int, desc: str, *, leave: bool = True) -> _tqdm:
    """Return a simple item-count progress bar."""
    return _tqdm(total=total, desc=desc, unit="file", leave=leave)


@contextmanager
def spinner(desc: str, emoji: str = "⏳") -> Iterator[None]:
    """Context manager that displays a simple 'working…' message."""
    msg(desc, emoji=emoji)
    yield
    msg("Done", emoji="✅")
