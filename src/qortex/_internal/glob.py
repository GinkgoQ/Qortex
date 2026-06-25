"""Gitignore-style glob matching for BIDS file paths.

Semantics:
  - Bare patterns (no ``/``) match the basename at any depth (MATCHBASE).
    ``*.fif`` matches every ``.fif`` file everywhere.
  - Every pattern is also tried with ``/**`` appended so that directory-like
    patterns (``sub-01``, ``sub-0001/anat``) match all files underneath.
  - A leading ``/`` anchors the pattern to the dataset root and disables
    basename matching.
  - ``*`` and ``**`` do NOT match dot-prefixed filenames (gitignore behaviour).
    Use ``'**/.*'`` to explicitly include hidden files.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import get_close_matches
from fnmatch import fnmatch

try:
    from wcmatch import glob as _wc
except ImportError:  # pragma: no cover - exercised only in reduced envs
    _wc = None


def is_dotfile(path: str) -> bool:
    """Return True if any path component starts with a dot."""
    return any(part.startswith(".") for part in path.split("/"))


def glob_filter(
    all_paths: Iterable[str],
    patterns: Iterable[str],
) -> dict[str, set[str]]:
    """Match paths against glob patterns; return per-pattern match sets.

    Parameters
    ----------
    all_paths:
        The complete flat list of BIDS-relative file paths from the manifest.
    patterns:
        Include or exclude glob patterns (same syntax as ``.gitignore``).

    Returns
    -------
    dict mapping each original pattern to its set of matched paths.
    """
    paths = list(all_paths)
    results: dict[str, set[str]] = {}

    for pattern in patterns:
        original = pattern
        anchored = pattern.startswith("/")
        pattern = pattern.removeprefix("/")
        stripped = pattern.rstrip("/")
        bare = "/" not in stripped

        if _wc is not None:
            base_flags = _wc.GLOBSTAR
            flags = base_flags | _wc.MATCHBASE if bare and not anchored else base_flags
            matched: set[str] = {
                str(p) for p in _wc.globfilter(paths, pattern, flags=flags)
            }
            matched |= {
                str(p)
                for p in _wc.globfilter(paths, stripped + "/**", flags=base_flags)
            }
        else:
            matched = _fallback_match(paths, pattern, stripped, bare, anchored)

        results[original] = matched

    return results


def _fallback_match(
    paths: list[str],
    pattern: str,
    stripped: str,
    bare: bool,
    anchored: bool,
) -> set[str]:
    """Small stdlib fallback for common BIDS path globs."""
    matched: set[str] = set()
    for path in paths:
        if is_dotfile(path) and not pattern.startswith(".") and "/." not in pattern:
            continue
        basename = path.rsplit("/", 1)[-1]
        if bare and not anchored and fnmatch(basename, pattern):
            matched.add(path)
        if fnmatch(path, pattern):
            matched.add(path)
        prefix = stripped.rstrip("/")
        if prefix and (path == prefix or path.startswith(prefix + "/")):
            matched.add(path)
    return matched


def find_close_matches(
    pattern: str,
    all_paths: list[str],
    n: int = 4,
    cutoff: float = 0.6,
) -> list[str]:
    """Return filenames similar to *pattern* for error messages."""
    return get_close_matches(pattern, all_paths, n=n, cutoff=cutoff)


def apply_include_exclude(
    all_files: list,           # list[FileRecord] or list[str]
    include: list[str] | None,
    exclude: list[str] | None,
    *,
    path_attr: str = "path",
) -> tuple[list, set[str], set[str]]:
    """Apply include/exclude glob logic and return (kept, included_set, excluded_set).

    ``all_files`` can be either ``FileRecord`` objects (then ``path_attr`` is
    used to get the path string) or plain strings.
    """
    def get_path(f) -> str:
        return getattr(f, path_attr) if not isinstance(f, str) else f

    all_paths = [get_path(f) for f in all_files]

    if include:
        matched = glob_filter(all_paths, include)
        included_set: set[str] = {p for ms in matched.values() for p in ms}
    else:
        included_set = {p for p in all_paths if not is_dotfile(p)}

    if exclude:
        matched = glob_filter(all_paths, exclude)
        excluded_set: set[str] = {p for ms in matched.values() for p in ms}
    else:
        excluded_set = set()

    kept = [f for f in all_files if get_path(f) in (included_set - excluded_set)]
    return kept, included_set, excluded_set
