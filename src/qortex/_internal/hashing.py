"""Streaming hash and integrity utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 65_536  # 64 KiB


def md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a local file (synchronous)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a local file (synchronous)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


class StreamingHasher:
    """Feed bytes incrementally; call ``hexdigest()`` when done."""

    def __init__(self, algorithm: str = "md5") -> None:
        self._h = hashlib.new(algorithm)

    def update(self, data: bytes) -> None:
        self._h.update(data)

    def hexdigest(self) -> str:
        return self._h.hexdigest()

    def copy(self) -> "StreamingHasher":
        inst = StreamingHasher.__new__(StreamingHasher)
        inst._h = self._h.copy()
        return inst


async def feed_existing_file_async(path: Path, hasher: StreamingHasher) -> None:
    """Feed an already-downloaded file into a hasher (for resume verification)."""
    import aiofiles

    async with aiofiles.open(path, "rb") as f:
        while True:
            chunk = await f.read(_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)


def parse_etag_md5(etag: str | None) -> str | None:
    """Extract a plain MD5 hex string from an S3-style ETag if it looks like one.

    S3 ETags for non-multipart uploads are ``"<md5>"`` (32 hex chars in quotes).
    Multipart ETags look like ``"<md5>-<N>"`` and are NOT the full-file MD5.
    """
    if etag is None:
        return None
    stripped = etag.strip('"')
    # Reject multipart ETags
    if "-" in stripped:
        return None
    if len(stripped) == 32 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return stripped.lower()
    return None
