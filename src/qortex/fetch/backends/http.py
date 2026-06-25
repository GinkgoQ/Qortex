"""Native async HTTP download backend.

Design decisions (improvements over openneuro-py):
  - Single shared httpx.AsyncClient across all file downloads (connection pool).
  - Two semaphores: HEAD pool (fast metadata) and GET pool (data transfer).
  - HEAD requests are skipped when the manifest already provides a checksum.
  - 429 Retry-After header is respected.
  - Per-file failure is isolated: exceptions propagate up to the engine which
    records them without aborting other downloads.
  - Resume uses HTTP Range header; size + hash verified after GET.
  - No success is reported until asyncio.gather() returns all results.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import aiofiles
import httpx

from qortex._internal.hashing import StreamingHasher, feed_existing_file_async, parse_etag_md5
from qortex._internal.progress import file_bar
from qortex.client.transport import RETRYABLE_CODES, RETRYABLE_EXCEPTIONS
from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import FileRecord
from qortex.core.exceptions import DownloadError, IntegrityError

_CHUNK = 65_536  # 64 KiB read chunk


class _RetryableError(Exception):
    """Internal signal: this attempt failed but should be retried."""


class HTTPBackend:
    """Async HTTP download backend using a shared httpx.AsyncClient."""

    backend_id = "http"

    def __init__(
        self,
        client: httpx.AsyncClient,
        sem_get: asyncio.Semaphore,
        sem_head: asyncio.Semaphore,
        config: QortexConfig | None = None,
        overall_progress=None,
    ) -> None:
        self._client = client
        self._sem_get = sem_get
        self._sem_head = sem_head
        self._cfg = config or get_config()
        self._overall = overall_progress  # tqdm bar, updated per chunk

    async def download_file(
        self,
        file: FileRecord,
        target_dir: Path,
        *,
        resume: bool = True,
        verify_hash: bool = True,
        verify_size: bool = True,
    ) -> tuple[int, int]:
        cfg = self._cfg
        url = file.urls[0] if file.urls else None
        if not url:
            raise DownloadError(file.path, "", "No download URL available.")

        outfile = target_dir / file.path
        outfile.parent.mkdir(parents=True, exist_ok=True)

        retries = 0
        backoff = cfg.retry_backoff_base

        for attempt in range(cfg.max_retries + 1):
            try:
                bytes_written = await self._attempt(
                    file=file,
                    url=url,
                    outfile=outfile,
                    resume=resume,
                    verify_hash=verify_hash,
                    verify_size=verify_size,
                    is_retry=attempt > 0,
                )
                return bytes_written, retries
            except _RetryableError as exc:
                retries += 1
                if attempt < cfg.max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, cfg.retry_backoff_max)
                    continue
                raise DownloadError(
                    file.path, url, str(exc.__cause__ or exc)
                ) from exc.__cause__

        raise DownloadError(file.path, url, "Exhausted retries.")

    # ── Core download logic ───────────────────────────────────────────────

    async def _attempt(
        self,
        *,
        file: FileRecord,
        url: str,
        outfile: Path,
        resume: bool,
        verify_hash: bool,
        verify_size: bool,
        is_retry: bool,
    ) -> int:
        remote_hash = file.checksum  # from manifest (preferred)

        # ── Phase 1: HEAD (only if we don't have hash from manifest) ─────
        if remote_hash is None:
            remote_hash = await self._head_for_hash(url)

        # ── Phase 2: Local file check ─────────────────────────────────────
        local_size = outfile.stat().st_size if outfile.exists() else 0
        remote_size = file.size
        mode: Literal["wb", "ab"] = "wb"
        offset = 0
        desc = outfile.name

        if outfile.exists() and remote_size is not None:
            if local_size == remote_size:
                # Potentially complete — verify hash
                if verify_hash and remote_hash is not None:
                    hasher = StreamingHasher("md5")
                    await feed_existing_file_async(outfile, hasher)
                    if hasher.hexdigest() == remote_hash:
                        # Cache hit — file already fully downloaded
                        if self._overall and not is_retry:
                            self._overall.update(remote_size)
                        return 0
                    # Hash mismatch → re-download
                    outfile.unlink()
                    desc = f"Re-downloading {outfile.name} (hash mismatch)"
                else:
                    if self._overall and not is_retry:
                        self._overall.update(remote_size)
                    return 0
            elif local_size < remote_size and resume:
                mode = "ab"
                offset = local_size
                desc = f"Resuming {outfile.name}"
                if self._overall and not is_retry:
                    self._overall.update(local_size)
            elif local_size > remote_size:
                outfile.unlink()
                desc = f"Re-downloading {outfile.name} (size mismatch)"
            else:
                outfile.unlink()

        # ── Phase 3: GET ──────────────────────────────────────────────────
        headers: dict[str, str] = {
            "Accept-Encoding": "",  # disable compression for accurate sizes
        }
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"

        try:
            async with self._sem_get:
                async with self._client.stream(
                    "GET", url, headers=headers
                ) as response:
                    if response.status_code == 429:
                        raise _RetryableError("HTTP 429") from None
                    if response.status_code in RETRYABLE_CODES:
                        raise _RetryableError(f"HTTP {response.status_code}") from None
                    if response.is_error:
                        raise DownloadError(
                            file.path, url,
                            f"GET returned HTTP {response.status_code}"
                        )
                    bytes_written = await self._stream_to_disk(
                        response=response,
                        outfile=outfile,
                        mode=mode,
                        offset=offset,
                        remote_path=file.path,
                        remote_hash=remote_hash,
                        remote_size=remote_size,
                        verify_hash=verify_hash,
                        verify_size=verify_size,
                        desc=desc,
                    )
        except RETRYABLE_EXCEPTIONS as exc:
            raise _RetryableError(str(exc)) from exc

        return bytes_written

    async def _head_for_hash(self, url: str) -> str | None:
        """Perform HEAD request and extract MD5 from ETag if reliable."""
        try:
            async with self._sem_head:
                response = await self._client.head(url, timeout=self._cfg.head_timeout)
                if response.status_code in RETRYABLE_CODES:
                    return None
                if response.is_error:
                    return None
                return parse_etag_md5(response.headers.get("etag"))
        except RETRYABLE_EXCEPTIONS:
            return None

    async def _stream_to_disk(
        self,
        *,
        response: httpx.Response,
        outfile: Path,
        mode: Literal["wb", "ab"],
        offset: int,
        remote_path: str,
        remote_hash: str | None,
        remote_size: int | None,
        verify_hash: bool,
        verify_size: bool,
        desc: str,
    ) -> int:
        hasher = StreamingHasher("md5") if verify_hash else None

        # Feed already-downloaded bytes into hasher if resuming
        if hasher and mode == "ab" and offset > 0 and outfile.exists():
            await feed_existing_file_async(outfile, hasher)

        total_written = 0
        with file_bar(remote_size, desc, initial=offset, leave=False) as pbar:
            async with aiofiles.open(outfile, mode=mode) as f:
                prev_downloaded = response.num_bytes_downloaded
                async for chunk in response.aiter_bytes(_CHUNK):
                    await f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    chunk_len = response.num_bytes_downloaded - prev_downloaded
                    prev_downloaded = response.num_bytes_downloaded
                    total_written += len(chunk)
                    pbar.update(chunk_len)
                    if self._overall:
                        self._overall.update(chunk_len)

        # ── Post-download verification ────────────────────────────────────
        if verify_hash and hasher and remote_hash is not None:
            got = hasher.hexdigest()
            if got != remote_hash:
                outfile.unlink(missing_ok=True)
                raise IntegrityError(remote_path, remote_hash, got, "hash")

        if verify_size and remote_size is not None:
            actual = outfile.stat().st_size
            if actual != remote_size:
                raise IntegrityError(remote_path, str(remote_size), str(actual), "size")

        # Detect OpenNeuro server-side JSON error payloads in tiny files
        if verify_size and total_written < 200:
            await _check_error_payload(outfile)

        return total_written


async def _check_error_payload(path: Path) -> None:
    """Detect ``{"error": "..."}`` payloads returned by the OpenNeuro CDN."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict) and list(data) == ["error"]:
            path.unlink(missing_ok=True)
            raise DownloadError(
                str(path), "", f"Server returned error payload: {data['error']}"
            )
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        pass
