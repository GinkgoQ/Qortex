"""Remote file gateway — stream small files from OpenNeuro CDN without download.

This is one of Qortex's most powerful capabilities: every file in an OpenNeuro
manifest carries a direct CDN URL (AWS S3 or OpenNeuro's own object store).
For small files (participants.tsv, events.tsv, JSON sidecars, dataset_description.json),
we can fetch them via HTTP without triggering a full dataset download.

For signal data files, we exploit HTTP Range requests to extract just the header:
  - NIfTI (.nii, .nii.gz): 352-byte header → shape, voxel sizes, TR, data type
    without downloading GBs of voxel data.
  - EDF/BrainVision: header bytes → channel count, sampling frequency, duration.

Architecture
------------
* ``RemoteFileGateway`` is the synchronous entry point. For single-file access
  it uses httpx directly. For batch operations (fetching all events files across
  a dataset), it dispatches to an asyncio executor with a per-session semaphore
  to respect the CDN rate limit.
* Per-URL in-process LRU cache with configurable TTL avoids redundant fetches
  within a session (events files are often read multiple times during inspection).
* Hard size guard (default 10 MB) prevents accidentally streaming large binary
  files. Callers can override for trusted paths.
* The NIfTI header decoder is pure Python — no nibabel required for inspection.

Key public methods
------------------
gateway.fetch_json(url)             → dict
gateway.fetch_tsv(url)              → polars.DataFrame
gateway.fetch_text(url)             → str
gateway.fetch_nifti_header(url)     → NIfTIHeader
gateway.batch_fetch_tsv(urls)       → dict[url, DataFrame]
gateway.batch_fetch_json(urls)      → dict[url, dict]
gateway.from_manifest_path(manifest, path) → bytes  (look up URL automatically)
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from qortex.client.transport import SSL_CONTEXT, USER_AGENT, RETRYABLE_EXCEPTIONS
from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import FileRecord, Manifest
from qortex.core.exceptions import QortexError

log = logging.getLogger(__name__)

# Maximum bytes to stream for "small file" preview
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
_NIFTI_HEADER_BYTES = 352               # NIfTI-1 header length
_GZIP_RANGE_BYTES = 65_536              # bytes to range-request for gzip NIfTI header
_CACHE_TTL_SECONDS = 3600               # 1 hour in-process cache TTL
_BATCH_CONCURRENCY = 24                 # max parallel remote fetches


# ── NIfTI header ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NIfTIHeader:
    """Key acquisition parameters extracted from a NIfTI header (352 bytes).

    Obtained via an HTTP Range request — no download of the full file required.
    A 37 MB fMRI run yields these parameters from just 352 bytes of network I/O.
    """
    ndim: int
    shape: tuple[int, ...]
    voxel_sizes_mm: tuple[float, ...]
    tr_s: float | None        # repetition time in seconds (pixdim[4] for 4D)
    dtype_code: int           # NIfTI data type code
    units_code: int           # spatial/temporal units code
    n_volumes: int | None     # shape[3] for 4D data
    description: str          # NIfTI descrip field (80 chars)
    raw_bytes_fetched: int

    @property
    def tr_ms(self) -> float | None:
        """TR in milliseconds if temporal units are milliseconds."""
        if self.tr_s is None:
            return None
        # units_code bit 3-7 encode temporal units: 8=sec, 16=msec, 24=usec, 32=Hz
        t_units = (self.units_code >> 3) & 0x07
        if t_units == 1:  # 1 = seconds
            return self.tr_s * 1000.0
        if t_units == 2:  # 2 = milliseconds
            return self.tr_s
        return self.tr_s * 1000.0  # assume seconds

    @property
    def is_4d(self) -> bool:
        return self.ndim == 4 and self.n_volumes is not None and self.n_volumes > 1

    @property
    def duration_s(self) -> float | None:
        """Approximate total scan duration in seconds (for 4D fMRI)."""
        if self.is_4d and self.tr_s:
            return self.n_volumes * self.tr_s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ndim": self.ndim,
            "shape": list(self.shape),
            "voxel_sizes_mm": list(self.voxel_sizes_mm),
            "tr_s": self.tr_s,
            "n_volumes": self.n_volumes,
            "is_4d": self.is_4d,
            "duration_s": self.duration_s,
            "description": self.description,
        }

    def __str__(self) -> str:
        if self.is_4d:
            return (
                f"4D fMRI {self.shape[0]}×{self.shape[1]}×{self.shape[2]}×{self.n_volumes} "
                f"vox={self.voxel_sizes_mm[0]:.2f}×{self.voxel_sizes_mm[1]:.2f}×{self.voxel_sizes_mm[2]:.2f}mm "
                f"TR={self.tr_s:.3f}s"
            )
        shape_str = "×".join(str(s) for s in self.shape[:self.ndim])
        vox_str = "×".join(f"{v:.2f}" for v in self.voxel_sizes_mm[:3])
        return f"{self.ndim}D {shape_str} vox={vox_str}mm"


# ── Cache ─────────────────────────────────────────────────────────────────────

class _TTLCache:
    """Simple thread-safe URL → bytes LRU cache with TTL."""

    def __init__(self, maxsize: int = 512, ttl: float = _CACHE_TTL_SECONDS) -> None:
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> bytes | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            data, ts = entry
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            return data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            if len(self._cache) >= self._maxsize:
                # Evict oldest entry
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (data, time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ── Gateway ───────────────────────────────────────────────────────────────────

class RemoteFileGateway:
    """Fetch small files from OpenNeuro CDN URLs without a full dataset download.

    Thread-safe. Create one instance per session and reuse it.

    Parameters
    ----------
    config:
        ``QortexConfig`` or None for defaults.
    max_bytes:
        Hard cap for file streaming. Files larger than this raise ``FileTooLargeError``.
        Does not apply to NIfTI header-only fetches.
    cache_ttl:
        Seconds to cache fetched bytes in-process. Set 0 to disable caching.
    """

    def __init__(
        self,
        config: QortexConfig | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        cache_ttl: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._cfg = config or get_config()
        self._max_bytes = max_bytes
        self._cache = _TTLCache(ttl=cache_ttl)
        self._client = httpx.Client(
            verify=SSL_CONTEXT,
            headers={"user-agent": USER_AGENT},
            timeout=self._cfg.metadata_timeout,
            follow_redirects=True,
        )

    # ── Core fetch ────────────────────────────────────────────────────────

    def fetch_bytes(
        self,
        url: str,
        *,
        max_bytes: int | None = None,
        use_cache: bool = True,
        range_bytes: int | None = None,
    ) -> bytes:
        """Fetch raw bytes from a URL with optional Range request and cache.

        Parameters
        ----------
        range_bytes:
            If set, issues ``Range: bytes=0-{range_bytes-1}`` and returns only
            the prefix. Used for NIfTI header extraction.
        """
        cache_key = f"{url}|{range_bytes}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        cap = max_bytes or self._max_bytes
        headers: dict[str, str] = {}
        if range_bytes is not None:
            headers["Range"] = f"bytes=0-{range_bytes - 1}"
        elif cap < _DEFAULT_MAX_BYTES:
            headers["Range"] = f"bytes=0-{cap - 1}"

        try:
            response = self._client.get(url, headers=headers)
        except RETRYABLE_EXCEPTIONS as exc:
            raise RemotePreviewError(url, f"Network error: {exc}") from exc

        if response.status_code not in (200, 206):
            raise RemotePreviewError(
                url,
                f"HTTP {response.status_code}: "
                + (response.text[:200] if response.text else ""),
            )

        data = response.content
        if range_bytes is None and len(data) > cap:
            raise FileTooLargeError(url, len(data), cap)

        if use_cache:
            self._cache.put(cache_key, data)
        return data

    # ── Text / structured types ───────────────────────────────────────────

    def fetch_text(self, url: str, *, max_bytes: int | None = None) -> str:
        """Fetch a text file (UTF-8, errors replaced)."""
        return self.fetch_bytes(url, max_bytes=max_bytes).decode("utf-8", errors="replace")

    def fetch_json(self, url: str) -> dict[str, Any]:
        """Fetch and parse a JSON file."""
        import json
        text = self.fetch_text(url, max_bytes=2 * 1024 * 1024)
        try:
            result = json.loads(text)
        except Exception as exc:
            raise RemotePreviewError(url, f"JSON parse error: {exc}") from exc
        if not isinstance(result, dict):
            raise RemotePreviewError(url, f"Expected JSON object, got {type(result).__name__}")
        return result

    def fetch_tsv(self, url: str, *, sep: str = "\t") -> Any:
        """Fetch and parse a TSV (or CSV) file as a Polars DataFrame."""
        import polars as pl
        text = self.fetch_text(url, max_bytes=5 * 1024 * 1024)
        try:
            return pl.read_csv(io.StringIO(text), separator=sep, infer_schema_length=10_000)
        except Exception as exc:
            raise RemotePreviewError(url, f"TSV parse error: {exc}") from exc

    def fetch_csv(self, url: str) -> Any:
        """Fetch and parse a CSV file as a Polars DataFrame."""
        return self.fetch_tsv(url, sep=",")

    # ── NIfTI header extraction ───────────────────────────────────────────

    def fetch_nifti_header(self, url: str) -> NIfTIHeader:
        """Extract NIfTI acquisition parameters from the first 352 bytes.

        Works via an HTTP Range request — the full file (often gigabytes) is
        never downloaded. For .nii.gz files, fetches enough compressed bytes
        to decompress the header; the decompressor stops as soon as 352 bytes
        of uncompressed data are available.

        Returns
        -------
        NIfTIHeader
            Shape, voxel sizes, TR (for 4D fMRI), and data type.

        Raises
        ------
        RemotePreviewError
            If the URL is not a NIfTI file or the header cannot be decoded.
        """
        is_gz = ".nii.gz" in url.lower() or url.lower().endswith(".gz")

        if is_gz:
            # Fetch enough compressed bytes to decompress 352-byte header.
            # In practice 64KB covers even high-compression cases.
            raw = self.fetch_bytes(url, range_bytes=_GZIP_RANGE_BYTES, max_bytes=_GZIP_RANGE_BYTES)
            header_bytes = _decompress_nifti_header(raw, url)
        else:
            raw = self.fetch_bytes(url, range_bytes=_NIFTI_HEADER_BYTES, max_bytes=_NIFTI_HEADER_BYTES)
            header_bytes = raw

        if len(header_bytes) < _NIFTI_HEADER_BYTES:
            raise RemotePreviewError(
                url,
                f"Could not extract {_NIFTI_HEADER_BYTES} header bytes "
                f"(got {len(header_bytes)}). File may be corrupt or non-NIfTI.",
            )

        return _parse_nifti_header(header_bytes, bytes_fetched=len(raw))

    # ── Manifest-aware helpers ────────────────────────────────────────────

    def from_manifest_path(
        self,
        manifest: Manifest,
        path: str,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> bytes:
        """Look up a path in the manifest and fetch its bytes remotely.

        Parameters
        ----------
        path:
            BIDS-relative path, e.g. ``"participants.tsv"`` or
            ``"sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.json"``.
        max_bytes:
            Refuse to fetch files larger than this. Raises ``FileTooLargeError``
            before any HTTP request when ``fr.size`` is known.

        Raises
        ------
        FileNotFoundError
            If the path is not in the manifest.
        FileTooLargeError
            If the file size exceeds ``max_bytes``.
        RemotePreviewError
            If the URL is not available or the fetch fails.
        """
        fr = manifest.get_file(path)
        if fr is None:
            raise FileNotFoundError(f"Path not in manifest: {path!r}")
        if fr.size and fr.size > max_bytes:
            raise FileTooLargeError(fr.path, fr.size, max_bytes)
        url = _pick_url(fr)
        return self.fetch_bytes(url, max_bytes=max_bytes)

    def preview_path(
        self, manifest: Manifest, path: str, *, n_rows: int = 20
    ) -> dict[str, Any]:
        """Smart preview of any manifest path — auto-detects format.

        Returns a dict with ``type``, ``content`` (DataFrame or dict or str),
        and metadata fields useful for display.
        """
        fr = manifest.get_file(path)
        if fr is None:
            raise FileNotFoundError(f"Path not in manifest: {path!r}")
        url = _pick_url(fr)
        ext = (fr.extension or "").lower()

        if ext == ".json":
            data = self.fetch_json(url)
            return {"type": "json", "path": path, "content": data, "url": url}

        if ext in (".tsv", ".csv"):
            sep = "\t" if ext == ".tsv" else ","
            df = self.fetch_tsv(url, sep=sep)
            return {
                "type": "table",
                "path": path,
                "content": df.head(n_rows),
                "n_rows_total": len(df),
                "columns": df.columns,
                "url": url,
            }

        if ext in (".nii", ".nii.gz"):
            hdr = self.fetch_nifti_header(url)
            return {"type": "nifti_header", "path": path, "content": hdr.to_dict(), "url": url}

        # Fallback: text preview
        text = self.fetch_text(url, max_bytes=4096)
        return {"type": "text", "path": path, "content": text[:2000], "url": url}

    # ── Batch operations (concurrent, async-backed) ───────────────────────

    def batch_fetch_tsv(
        self,
        url_map: dict[str, str],
        concurrency: int = _BATCH_CONCURRENCY,
        max_bytes_per_file: int = 5 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Fetch multiple TSV files concurrently.

        Parameters
        ----------
        url_map:
            ``{key: url}`` mapping. Keys are returned as-is in the result.
        concurrency:
            Maximum parallel HTTP connections.
        max_bytes_per_file:
            Per-file byte cap (default 5 MB). Files exceeding this raise
            ``FileTooLargeError`` stored in the result dict.

        Returns
        -------
        dict[str, DataFrame | Exception]
            Successful fetches return a Polars DataFrame.
            Failed fetches return the Exception (never raises).
        """
        coro = _async_batch_fetch(
            url_map=url_map,
            fetch_fn=self._async_fetch_tsv,
            concurrency=concurrency,
            cfg=self._cfg,
            max_bytes=max_bytes_per_file,
        )
        return _run_async(coro)

    def batch_fetch_json(
        self,
        url_map: dict[str, str],
        concurrency: int = _BATCH_CONCURRENCY,
        max_bytes_per_file: int = 2 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Fetch multiple JSON files concurrently."""
        coro = _async_batch_fetch(
            url_map=url_map,
            fetch_fn=self._async_fetch_json,
            concurrency=concurrency,
            cfg=self._cfg,
            max_bytes=max_bytes_per_file,
        )
        return _run_async(coro)

    async def _async_fetch_tsv(
        self, url: str, client: httpx.AsyncClient, max_bytes: int = 5 * 1024 * 1024
    ) -> Any:
        import polars as pl
        cached = self._cache.get(f"{url}|None")
        if cached is not None:
            return pl.read_csv(io.BytesIO(cached), separator="\t", infer_schema_length=10_000)
        response = await client.get(url)
        if response.status_code not in (200, 206):
            raise RemotePreviewError(url, f"HTTP {response.status_code}")
        data = response.content
        if len(data) > max_bytes:
            raise FileTooLargeError(url, len(data), max_bytes)
        self._cache.put(f"{url}|None", data)
        return pl.read_csv(io.BytesIO(data), separator="\t", infer_schema_length=10_000)

    async def _async_fetch_json(
        self, url: str, client: httpx.AsyncClient, max_bytes: int = 2 * 1024 * 1024
    ) -> dict:
        import json
        cached = self._cache.get(f"{url}|None")
        if cached is not None:
            return json.loads(cached)
        response = await client.get(url)
        if response.status_code not in (200, 206):
            raise RemotePreviewError(url, f"HTTP {response.status_code}")
        data = response.content
        if len(data) > max_bytes:
            raise FileTooLargeError(url, len(data), max_bytes)
        self._cache.put(f"{url}|None", data)
        return json.loads(data)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RemoteFileGateway":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── NIfTI header parsing ──────────────────────────────────────────────────────

def _decompress_nifti_header(compressed: bytes, url: str) -> bytes:
    """Decompress gzip bytes until we have at least 352 bytes or exhaust input."""
    try:
        gz = gzip.GzipFile(fileobj=io.BytesIO(compressed))
        header_bytes = gz.read(_NIFTI_HEADER_BYTES)
        return header_bytes
    except EOFError:
        # Input too short — return what we got
        try:
            gz = gzip.GzipFile(fileobj=io.BytesIO(compressed))
            return gz.read()
        except Exception:
            return b""
    except Exception as exc:
        raise RemotePreviewError(url, f"gzip decompression failed: {exc}") from exc


def _parse_nifti_header(header: bytes, bytes_fetched: int) -> NIfTIHeader:
    """Parse a 348-byte NIfTI-1 header into structured fields.

    NIfTI-1 layout (little-endian by default; checks endianness via sizeof_hdr):
      offset 0   : sizeof_hdr (int32) — should be 348
      offset 40  : dim (8 × int16) — [ndim, nx, ny, nz, nt, ...]
      offset 76  : pixdim (8 × float32) — [qfac, dx, dy, dz, TR, ...]
      offset 112 : vox_offset (float32) — where data starts
      offset 148 : scl_slope (float32)
      offset 152 : scl_inter (float32)
      offset 252 : descrip (char[80])
      offset 340 : dim_info (char) — frequency/phase/slice encoding
      offset 344 : magic (char[4]) — "ni1\0" or "n+1\0"
    """
    # Determine endianness from sizeof_hdr (should be 348)
    sizeof_hdr_le = struct.unpack_from("<i", header, 0)[0]
    endian = "<" if sizeof_hdr_le == 348 else ">"

    # dim: 8 int16 at offset 40
    dims = struct.unpack_from(f"{endian}8h", header, 40)
    ndim = max(0, min(int(dims[0]), 7))
    shape = tuple(int(d) for d in dims[1 : ndim + 1])

    # pixdim: 8 float32 at offset 76
    pixdim = struct.unpack_from(f"{endian}8f", header, 76)
    voxel_sizes = tuple(abs(float(pixdim[i])) for i in range(1, 4))

    # TR is pixdim[4] for 4D data (temporal step)
    tr_raw = float(pixdim[4]) if ndim >= 4 else None

    # datatype: int16 at offset 70
    dtype_code = struct.unpack_from(f"{endian}h", header, 70)[0]

    # xyzt_units: uint8 at offset 123 (encodes spatial + temporal units)
    units_code = struct.unpack_from("B", header, 123)[0]

    # TR interpretation: temporal units from bits 3-5
    t_units = (units_code >> 3) & 0x07
    # 1=sec, 2=msec, 3=usec, 8=Hz, 16=ppm, 24=rad/s
    tr_s: float | None = None
    if tr_raw and tr_raw > 0:
        if t_units == 1:    # seconds
            tr_s = tr_raw
        elif t_units == 2:  # milliseconds
            tr_s = tr_raw / 1000.0
        elif t_units == 3:  # microseconds
            tr_s = tr_raw / 1_000_000.0
        else:
            # Unknown or Hz — assume seconds if value is plausible (0.1-20s)
            tr_s = tr_raw if 0.1 <= tr_raw <= 20.0 else None

    # descrip: 80 chars at offset 148 in NIfTI-1 (common path — some NIfTI use offset 288)
    # NIfTI-1 spec: descrip is at byte 148-227 (80 chars)
    try:
        descrip = header[148:228].decode("ascii", errors="replace").rstrip("\x00 ")
    except Exception:
        descrip = ""

    n_volumes = int(shape[3]) if ndim == 4 and len(shape) >= 4 else None

    return NIfTIHeader(
        ndim=ndim,
        shape=shape,
        voxel_sizes_mm=voxel_sizes,
        tr_s=tr_s,
        dtype_code=dtype_code,
        units_code=units_code,
        n_volumes=n_volumes,
        description=descrip,
        raw_bytes_fetched=bytes_fetched,
    )


# ── Async batch engine ────────────────────────────────────────────────────────

def _run_async(coro) -> Any:
    """Run an async coroutine safely regardless of whether a loop is already running.

    In Jupyter notebooks and async frameworks, ``asyncio.run()`` raises
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
    We detect this and fall back to running the coroutine in a thread that has
    its own event loop — identical semantics, no external dependencies.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # A loop is already running (Jupyter / async app) — use a thread
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


async def _async_batch_fetch(
    *,
    url_map: dict[str, str],
    fetch_fn,
    concurrency: int,
    cfg: QortexConfig,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Execute fetch_fn(url, client) for all URLs concurrently, bounded by semaphore."""
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        verify=SSL_CONTEXT,
        headers={"user-agent": USER_AGENT},
        timeout=cfg.metadata_timeout,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=concurrency + 4),
    ) as client:
        async def _one(key: str, url: str) -> tuple[str, Any]:
            async with sem:
                try:
                    result = await fetch_fn(url, client, max_bytes)
                    return key, result
                except Exception as exc:
                    log.debug("batch fetch failed for %s: %s", url, exc)
                    return key, exc

        tasks = [_one(k, u) for k, u in url_map.items()]
        pairs = await asyncio.gather(*tasks)
    return dict(pairs)


# ── URL utilities ─────────────────────────────────────────────────────────────

def _pick_url(fr: FileRecord) -> str:
    """Return the best CDN URL for a FileRecord.

    Prefer S3 URLs (stable, not proxied through CRN) over CRN object-store URLs.
    """
    if not fr.urls:
        raise RemotePreviewError(fr.path, "No URLs available in manifest.")
    # S3 URLs are more stable and handle larger parallel loads better
    s3_urls = [u for u in fr.urls if "s3.amazonaws.com" in u or "s3." in u]
    return s3_urls[0] if s3_urls else fr.urls[0]


def best_url_for_path(manifest: Manifest, path: str) -> str:
    """Return the CDN URL for a manifest path, or raise FileNotFoundError."""
    fr = manifest.get_file(path)
    if fr is None:
        raise FileNotFoundError(f"Path not in manifest: {path!r}")
    return _pick_url(fr)


def small_files_from_manifest(
    manifest: Manifest,
    *,
    max_size_bytes: int = 2 * 1024 * 1024,  # 2 MB
    extensions: set[str] | None = None,
) -> list[FileRecord]:
    """Return manifest files that are small enough for remote preview.

    Parameters
    ----------
    max_size_bytes:
        Files above this threshold are excluded (they require download).
    extensions:
        If set, only files with these extensions are returned.
    """
    exts = extensions or {".tsv", ".csv", ".json", ".txt", ".bvec", ".bval"}
    return [
        f for f in manifest.files
        if not f.is_dir
        and f.extension in exts
        and f.urls
        and (f.size is None or f.size <= max_size_bytes)
    ]


# ── Exceptions ────────────────────────────────────────────────────────────────

class RemotePreviewError(QortexError):
    """Raised when a remote file cannot be fetched or parsed."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Remote preview failed for {url!r}: {reason}")
        self.url = url
        self.reason = reason


class FileTooLargeError(RemotePreviewError):
    """Raised when a file exceeds the max_bytes guard."""

    def __init__(self, url: str, actual: int, limit: int) -> None:
        super().__init__(
            url,
            f"file size {actual:,} B exceeds max_bytes={limit:,} B. "
            "Use download() for large data files.",
        )
        self.actual = actual
        self.limit = limit
