"""Remote NIfTI streamer — byte-range access without full download.

Implements two access patterns:

  1. **Header-only** (< 65 KB transferred):
     Fetch and parse the NIfTI-1/2 header entirely via an HTTP Range request.
     Provides shape, voxel sizes, TR, affine, data type, and byte offset to the
     start of the voxel array.

  2. **Virtual array slicing** (proportional to slice size):
     For uncompressed ``.nii`` files: calculates the exact byte range for any
     2D slice or 3D volume, issues a single Range request, and returns a numpy
     array without reading the rest of the file.

     For gzip-compressed ``.nii.gz`` files: implements streaming gzip
     decompression that reads only enough compressed data to produce the
     requested output — up to 40× savings for thin axial slices of large
     fMRI volumes.

Design
------
* Pure-Python HTTP client (httpx via ``RemoteFileGateway``) — no fsspec dependency.
* Configurable LRU + disk cache via ``stream._cache``.
* Thread-safe: one streamer can be shared across workers.
* Lazy: header is fetched once and cached; subsequent slice requests reuse it.
"""

from __future__ import annotations

import io
import logging
import struct
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qortex.stream._cache import MemoryCache, make_cache

log = logging.getLogger(__name__)

# NIfTI-1 constants
_NIFTI1_HDR_SIZE = 348
_NIFTI2_HDR_SIZE = 540
_NIFTI1_MAGIC = b"ni1\x00"
_NIFTI2_MAGIC = b"ni2\x00"
_NIFTI_PAIR_MAGIC = b"n+1\x00"
_NIFTI2_PAIR_MAGIC = b"n+2\x00"
_GZIP_PREFETCH_BYTES = 1 << 17   # 128 KB — covers header + small volumes
_NIFTI_DTYPE_MAP = {
    2:  np.uint8,
    4:  np.int16,
    8:  np.int32,
    16: np.float32,
    32: np.complex64,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
    1280: np.uint64,
    1792: np.complex128,
}


@dataclass
class NiftiStreamHeader:
    """NIfTI header decoded from a byte-range fetch.

    Includes everything needed for virtual array slicing:
    offset to the voxel data, element dtype, and C-order strides.
    """
    ndim: int
    shape: tuple[int, ...]          # (x, y, z) or (x, y, z, t)
    voxel_sizes_mm: tuple[float, ...]
    affine: np.ndarray              # 4×4 float64
    tr_s: float | None              # repetition time in seconds (4D only)
    dtype: np.dtype
    vox_offset: int                 # byte offset in the file where data starts
    is_gz: bool
    is_nifti2: bool
    description: str
    source_url: str
    bytes_fetched: int
    scl_slope: float = 1.0          # NIfTI intensity calibration: value = raw * scl_slope + scl_inter
    scl_inter: float = 0.0          # (scl_slope == 0 means "no scaling", per the NIfTI-1/2 spec)

    @property
    def needs_scaling(self) -> bool:
        return self.scl_slope not in (0.0, 1.0) or self.scl_inter != 0.0

    @property
    def n_volumes(self) -> int | None:
        return self.shape[3] if self.ndim == 4 else None

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.shape[:3]

    @property
    def is_4d(self) -> bool:
        return self.ndim == 4

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def volume_bytes(self) -> int:
        """Bytes for one 3D volume."""
        return int(np.prod(self.spatial_shape)) * self.itemsize

    @property
    def slice_bytes(self) -> int:
        """Bytes for one 2D axial slice (z-plane) in the first two spatial dims."""
        return self.shape[0] * self.shape[1] * self.itemsize

    def strides(self) -> tuple[int, ...]:
        """C-order strides in bytes for the full spatial array."""
        shape = self.shape
        s = [self.itemsize]
        for dim in reversed(shape[1:self.ndim]):
            s.insert(0, s[0] * dim)
        return tuple(s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ndim": self.ndim,
            "shape": list(self.shape),
            "spatial_shape": list(self.spatial_shape),
            "voxel_sizes_mm": list(self.voxel_sizes_mm),
            "tr_s": self.tr_s,
            "dtype": str(self.dtype),
            "vox_offset": self.vox_offset,
            "is_gz": self.is_gz,
            "is_nifti2": self.is_nifti2,
            "description": self.description,
            # Previously computed internally (as properties/fields) but never
            # surfaced here — a console API consumer had no way to know a file
            # was 4D, needed scl_slope/scl_inter calibration, or to place a
            # voxel in scanner/world space, without a second, separate call.
            "is_4d": self.is_4d,
            "n_volumes": self.n_volumes,
            "scl_slope": self.scl_slope,
            "scl_inter": self.scl_inter,
            "needs_scaling": self.needs_scaling,
            "affine": self.affine.tolist(),
        }

    def __str__(self) -> str:
        shape_str = "×".join(str(s) for s in self.shape[:self.ndim])
        vox_str   = "×".join(f"{v:.2f}" for v in self.voxel_sizes_mm[:3])
        return f"NIfTI {self.ndim}D {shape_str} vox={vox_str}mm dtype={self.dtype}"


class NiftiStreamer:
    """Stream NIfTI data from a remote URL without downloading the full file.

    Parameters
    ----------
    url_or_path:
        HTTP/HTTPS URL (OpenNeuro CDN or S3) or local filesystem path.
        Detects ``.nii`` vs ``.nii.gz`` automatically from the URL.
    cache_dir:
        Directory for the persistent byte-range cache.  Defaults to
        ``~/.qortex/stream_cache``.  Pass ``None`` to disable disk caching.
    cache_backend:
        ``"memory"`` (default) for in-process LRU, ``"disk"`` for persistent cache.
    token:
        Optional API token for authenticated OpenNeuro URLs.

    Examples
    --------
    >>> streamer = NiftiStreamer("https://cdn.openneuro.org/.../T1w.nii.gz")
    >>> hdr = streamer.header()
    >>> print(hdr.shape, hdr.voxel_sizes_mm)
    >>> axial_slice = streamer.get_slice(axis=2, index=90)
    >>> vol_t50 = streamer.get_volume(t=50)
    >>> lazy = streamer.get_lazy_array()        # nibabel proxy, fetches on demand
    """

    def __init__(
        self,
        url_or_path: str | Path,
        *,
        cache_dir: Path | None = None,
        cache_backend: str = "memory",
        token: str | None = None,
    ) -> None:
        self._url = str(url_or_path)
        self._is_gz = ".nii.gz" in self._url.lower() or self._url.lower().endswith(".gz")
        self._is_local = not self._url.startswith(("http://", "https://", "s3://"))
        self._token = token
        self._cache = make_cache(backend=cache_backend, cache_dir=cache_dir)
        self._header: NiftiStreamHeader | None = None
        # Decompressed-volume LRU (native dtype, unscaled, F-ordered). A .nii.gz
        # cannot be randomly seeked, so extracting any single slice means
        # decompressing the whole volume from byte 0 — scrubbing the Z slider
        # over one compressed volume previously paid that full decode on *every*
        # slice (O(Z × decode)). Caching the last couple of decoded volumes
        # collapses that to one decode per volume; scaling and slicing are then
        # cheap views on top. Kept small (2) so a 4D scrub doesn't retain the
        # whole series in RAM.
        self._volume_lru: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._volume_lru_max = 2
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def header(self, *, force_refresh: bool = False) -> NiftiStreamHeader:
        """Fetch and parse the NIfTI header.  Cached after first call.

        Parameters
        ----------
        force_refresh:
            Re-fetch even if cached.
        """
        if self._header is not None and not force_refresh:
            return self._header
        raw = self._fetch_bytes(
            start=0,
            length=_GZIP_PREFETCH_BYTES,
            label="header",
        )
        if self._is_gz:
            try:
                raw = _stream_decompress(raw, max_bytes=_NIFTI2_HDR_SIZE + 32)
            except Exception as exc:
                raise ValueError(
                    f"Cannot decompress NIfTI header from {self._url!r}: {exc}"
                ) from exc
        self._header = _parse_nifti_header(raw, source_url=self._url, is_gz=self._is_gz, bytes_fetched=len(raw))
        return self._header

    def get_slice(
        self,
        axis: int,
        index: int,
        *,
        t: int = 0,
        dtype: np.dtype | None = None,
        canonical: bool = True,
    ) -> np.ndarray:
        """Stream one 2D slice from a remote NIfTI volume.

        Parameters
        ----------
        axis:
            Spatial axis to slice: 0 = sagittal, 1 = coronal, 2 = axial.
        index:
            Voxel index along the chosen axis.
        t:
            Volume time index (only for 4D files; default 0).
        dtype:
            Output dtype.  Defaults to the file's native dtype.
        canonical:
            For axis=2 (axial), flip to canonical orientation (default True).

        Returns
        -------
        np.ndarray
            2D array of shape depending on axis:
            axis=0 → (y, z), axis=1 → (x, z), axis=2 → (x, y).
        """
        hdr = self.header()
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        shape = hdr.spatial_shape
        if index < 0 or index >= shape[axis]:
            raise IndexError(
                f"Slice index {index} out of range [0, {shape[axis]}) for axis {axis}"
            )
        _validate_time_index(hdr, t)

        # For .nii.gz: stream-decompress up to the needed point (whole volume;
        # gzip can't be randomly seeked into, so single-slice extraction isn't
        # possible without decompressing from the start regardless).
        # For uncompressed .nii, axis=2 (axial) slices are contiguous in the
        # file's Fortran-ordered storage — fetch exactly that slice's bytes,
        # not the whole volume (see _fetch_slice_contiguous). axis=0/1 are not
        # contiguous (interleaved with a stride spanning most of the volume
        # regardless of which index is requested), so those still fall back
        # to a whole-volume fetch; this was previously true for *every* axis,
        # including axis=2, which made "one slice" cost as much bandwidth as
        # a full download for the single most commonly viewed plane.
        if self._is_gz:
            vol = self._volume_array(hdr, t)
            slc = np.take(vol, index, axis=axis).copy()
        else:
            fast = self._fetch_slice_contiguous(hdr, axis, index, t)
            if fast is not None:
                nx, ny = hdr.spatial_shape[0], hdr.spatial_shape[1]
                slc = np.frombuffer(fast, dtype=hdr.dtype).reshape((nx, ny), order="F")
            else:
                vol = self._volume_array(hdr, t)
                slc = np.take(vol, index, axis=axis).copy()
        if hdr.needs_scaling:
            slc = slc.astype(np.float32) * hdr.scl_slope + hdr.scl_inter
        if dtype is not None:
            slc = slc.astype(dtype)
        return slc

    def get_volume(
        self,
        t: int = 0,
        *,
        dtype: np.dtype | None = None,
        canonical: bool = True,
    ) -> np.ndarray:
        """Stream one complete 3D volume from a remote 4D NIfTI file.

        For 3D files, ``t`` must be 0.  Returns shape ``(x, y, z)``.

        Parameters
        ----------
        t:
            Time/volume index (0-based).
        dtype:
            Output dtype (default: file's native dtype → float32 for ML).
        canonical:
            Apply nibabel ``as_closest_canonical`` reorientation (requires nibabel).
        """
        hdr = self.header()
        _validate_time_index(hdr, t)
        vol = self._volume_array(hdr, t)
        if hdr.needs_scaling:
            vol = vol.astype(np.float32) * hdr.scl_slope + hdr.scl_inter
        elif dtype is None:
            # Return a writable copy — the cached array is read-only (shared
            # across callers). np.array always copies (unlike ascontiguousarray,
            # which can hand back the read-only input when it's already
            # contiguous), matching the previous .copy() contract.
            vol = np.array(vol)
        if dtype is not None:
            vol = vol.astype(dtype)
        return vol

    def get_lazy_array(self, *, canonical: bool = True) -> Any:
        """Return a nibabel proxy image — data is fetched on first ``.get_fdata()`` call.

        This is useful when you want nibabel-compatible code paths (transforms,
        affine access) without incurring the full download cost.  When ``.get_fdata()``
        is eventually called, nibabel reads from the URL if it is a local path,
        or raises if it is remote.  For remote URLs, use ``get_volume()`` instead.

        For local NIfTI paths, this returns a true nibabel lazy-loading proxy
        (data is memory-mapped, loaded on access).

        Returns
        -------
        nibabel.Nifti1Image
            If ``url_or_path`` is a local path.
        NiftiStreamHeader
            If remote (nibabel cannot memory-map HTTP URLs).
        """
        if self._is_local:
            try:
                import nibabel as nib
                img = nib.load(self._url)
                if canonical:
                    img = nib.as_closest_canonical(img)
                return img
            except ImportError:
                raise ImportError("get_lazy_array() requires nibabel: pip install 'qortex[mri]'")
        log.warning(
            "get_lazy_array() for remote URLs returns the NiftiStreamHeader only. "
            "Call get_volume() to stream the actual array data."
        )
        return self.header()

    def prefetch_slabs(
        self,
        axis: int = 2,
        *,
        t: int = 0,
        slab_size: int = 8,
    ) -> list[np.ndarray]:
        """Prefetch the full 3D volume in contiguous slabs and return a list of 2D arrays.

        Useful for progressive rendering or training on slice-level labels.

        Parameters
        ----------
        axis:
            Axis along which to slice.
        slab_size:
            Number of consecutive slices per HTTP request (increases efficiency).

        Returns
        -------
        list[np.ndarray]
            List of 2D arrays, one per slice along the chosen axis.
        """
        hdr = self.header()
        n_slices = hdr.spatial_shape[axis]
        # One decode for the whole volume (cached, scaled, writable), then cheap
        # views per slice — the same path for compressed and uncompressed files.
        vol = self.get_volume(t=t)
        return [np.take(vol, i, axis=axis) for i in range(n_slices)]

    def stream_stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        if hasattr(self._cache, "stats"):
            return self._cache.stats()
        return {}

    # ── Private ───────────────────────────────────────────────────────────

    def _fetch_bytes(self, start: int, length: int, label: str = "") -> bytes:
        """Fetch a byte range from the remote file, using the cache."""
        key = f"{self._url}|{start}|{length}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._is_local:
            with open(self._url, "rb") as fh:
                fh.seek(start)
                data = fh.read(length)
        else:
            data = self._remote_range_request(start, length)

        self._cache.put(key, data)
        return data

    def _remote_range_request(self, start: int, length: int) -> bytes:
        from qortex.client.remote import get_shared_gateway
        gw = get_shared_gateway()
        # A true bytes=start-end range — previously this fetched
        # bytes=0-{start+length-1} (the whole prefix up to the window) and
        # relied on the caller to discard everything before `start`, which
        # for a late volume/slice in a large uncompressed .nii file could
        # mean downloading hundreds of MB just to keep the last few.
        return gw.fetch_bytes(self._url, range_start=start, range_bytes=length)

    def _fetch_volume_raw(self, hdr: NiftiStreamHeader, t: int) -> bytes:
        """Fetch exactly one 3D volume from an uncompressed .nii file."""
        offset = hdr.vox_offset + t * hdr.volume_bytes
        return self._fetch_bytes(offset, hdr.volume_bytes, label=f"vol-t{t}")

    def _volume_array(self, hdr: NiftiStreamHeader, t: int) -> np.ndarray:
        """Return the decoded, native-dtype-preserving, *unscaled* 3D volume for
        time index ``t`` as a read-only F-ordered array, memoized in a small LRU.

        This is the single decode point for whole-volume access. For a
        compressed file it turns repeated slice/volume requests (Z-slider
        scrubbing, MIP, time scrubbing over the cached window) into one decode
        instead of one per request. The array is deliberately read-only so a
        caller can never corrupt the shared cache entry — ``get_slice`` copies
        the extracted plane, ``get_volume`` copies/scales into a fresh array.
        """
        with self._lock:
            hit = self._volume_lru.get(t)
            if hit is not None:
                self._volume_lru.move_to_end(t)
                return hit

        if self._is_gz:
            vol_data = self._stream_volume_gz(hdr, t)
        else:
            vol_data = self._fetch_volume_raw(hdr, t)
        # np.frombuffer over immutable bytes yields a read-only array; the
        # reshape preserves that, which is exactly the sharing guarantee we want.
        vol = np.frombuffer(vol_data, dtype=hdr.dtype).reshape(hdr.spatial_shape, order="F")

        with self._lock:
            self._volume_lru[t] = vol
            self._volume_lru.move_to_end(t)
            while len(self._volume_lru) > self._volume_lru_max:
                self._volume_lru.popitem(last=False)
        return vol

    def _fetch_slice_contiguous(
        self, hdr: NiftiStreamHeader, axis: int, index: int, t: int
    ) -> bytes | None:
        """Fetch just the bytes for one 2D slice with a single Range request,
        when the file's storage order makes that slice contiguous. Returns
        ``None`` when it isn't (caller falls back to a whole-volume fetch).

        NIfTI voxel data is stored in Fortran (column-major) order: element
        ``(x, y, z)`` sits at byte offset
        ``vox_offset + (x + y*nx + z*nx*ny) * itemsize``. Fixing the *last*
        spatial axis (``axis=2``, axial) makes every ``(x, y)`` element for
        that ``z`` contiguous — exactly ``nx*ny*itemsize`` bytes starting at
        ``vox_offset + z*nx*ny*itemsize`` (plus the volume offset for 4D
        files). Fixing axis 0 or 1 does not: the needed elements are
        interleaved with a stride of ``nx*itemsize`` or ``nx*ny*itemsize``
        respectively, spanning a byte range close to the size of the whole
        volume regardless of which index is requested — a "single range
        request" for those axes would fetch nearly as much as just fetching
        the volume, so there's no real optimization available there without
        issuing many small requests (one per row), which trades bandwidth for
        request-count/latency and isn't a clear win in general.
        """
        if axis != 2:
            return None
        nx, ny = hdr.spatial_shape[0], hdr.spatial_shape[1]
        itemsize = hdr.itemsize
        slice_bytes = nx * ny * itemsize
        volume_offset = hdr.vox_offset + t * hdr.volume_bytes
        slice_offset = volume_offset + index * slice_bytes
        return self._fetch_bytes(slice_offset, slice_bytes, label=f"slice-ax2-t{t}-i{index}")

    def _stream_volume_gz(self, hdr: NiftiStreamHeader, t: int) -> bytes:
        """Stream-decompress a .nii.gz file to extract one 3D volume.

        Reads compressed data in chunks, decompressing until we reach
        ``vox_offset + (t+1) * volume_bytes`` in decompressed coordinates.
        Only the requested volume is retained.
        """
        target_start = hdr.vox_offset + t * hdr.volume_bytes
        target_end   = target_start + hdr.volume_bytes

        # Estimate compressed bytes needed.  Neuroimaging data typically
        # achieves 2–8× compression; we fetch 1.5× the raw volume size as an
        # initial chunk, doubling if necessary.
        initial_chunk = max(int(target_end * 0.6), _GZIP_PREFETCH_BYTES)
        for attempt in range(6):
            fetch_len = min(initial_chunk << attempt, 512 * 1024 * 1024)  # cap at 512 MB
            raw_compressed = self._fetch_bytes(0, fetch_len, label=f"gz-vol-t{t}")
            try:
                decompressed = _stream_decompress(raw_compressed, max_bytes=target_end)
            except _NeedMoreData:
                log.debug(
                    "NiftiStreamer: need more data for t=%d (attempt %d, fetched %d B)",
                    t, attempt + 1, fetch_len,
                )
                continue
            if len(decompressed) >= target_end:
                return decompressed[target_start:target_end]
            # Not enough decompressed yet — fetch more
        raise IOError(
            f"Cannot stream volume t={t} from {self._url!r}: "
            f"fetched {fetch_len / 1e6:.1f} MB compressed but could not reach "
            f"decompressed offset {target_end / 1e6:.1f} MB."
        )


# ── NIfTI header parsing ──────────────────────────────────────────────────────

def _validate_time_index(hdr: NiftiStreamHeader, t: int) -> None:
    """Validate a requested 3D volume index for 3D and 4D NIfTI inputs."""
    if t < 0:
        upper = hdr.n_volumes if hdr.is_4d else 1
        raise IndexError(f"Volume index {t} out of range [0, {upper})")
    if hdr.is_4d:
        n_volumes = hdr.n_volumes or 0
        if t >= n_volumes:
            raise IndexError(f"Volume index {t} out of range [0, {n_volumes})")
    elif t != 0:
        raise IndexError(f"Volume index {t} out of range [0, 1)")


def _detect_byte_order(raw: bytes, *, is_nifti2: bool) -> str:
    """Return the struct/numpy byte-order prefix (``"<"`` or ``">"``) for this
    header.

    NIfTI does not carry an explicit endianness flag; the canonical detection
    (the one nibabel uses) reads ``sizeof_hdr`` — a known constant, 348 for
    NIfTI-1, 540 for NIfTI-2 — as a native int32/int64 at offset 0 and checks
    which byte order reproduces that constant. Big-endian NIfTI files are rare
    but valid and real (older FSL/AFNI exports, some PPC-era scanner output);
    before this, every multi-byte field (dims, dtype, vox_offset, scl, affine)
    *and* the voxel data itself was read as little-endian unconditionally,
    silently yielding byte-swapped garbage (e.g. a 4×5×6 volume parsed as
    1024×1280×1536) rather than an honest error.
    """
    expected = _NIFTI2_HDR_SIZE if is_nifti2 else _NIFTI1_HDR_SIZE
    if is_nifti2:
        # NIfTI-2 stores sizeof_hdr as an int32 at offset 0.
        (le,) = struct.unpack_from("<i", raw, 0)
        (be,) = struct.unpack_from(">i", raw, 0)
    else:
        (le,) = struct.unpack_from("<i", raw, 0)
        (be,) = struct.unpack_from(">i", raw, 0)
    if le == expected:
        return "<"
    if be == expected:
        return ">"
    # sizeof_hdr didn't match either order (some minimal writers leave it 0);
    # fall back to little-endian, the overwhelmingly common on-disk order.
    return "<"


def _parse_nifti_header(
    raw: bytes,
    *,
    source_url: str = "",
    is_gz: bool = False,
    bytes_fetched: int = 0,
) -> NiftiStreamHeader:
    """Parse NIfTI-1 or NIfTI-2 header from a byte buffer."""
    if len(raw) < _NIFTI1_HDR_SIZE:
        raise ValueError(
            f"Buffer too short to contain a NIfTI header: {len(raw)} < {_NIFTI1_HDR_SIZE}"
        )

    # Detect NIfTI version by magic string at offset 344 (NIfTI-1) or 4 (NIfTI-2)
    is_nifti2 = raw[4:8] in (_NIFTI2_MAGIC, _NIFTI2_PAIR_MAGIC)
    is_nifti1 = raw[344:348] in (_NIFTI1_MAGIC, _NIFTI_PAIR_MAGIC) or raw[344:347] == b"n+1"

    if is_nifti2 and len(raw) >= _NIFTI2_HDR_SIZE:
        endian = _detect_byte_order(raw, is_nifti2=True)
        return _parse_nifti2(raw, source_url=source_url, is_gz=is_gz, bytes_fetched=bytes_fetched, endian=endian)
    elif is_nifti1 and len(raw) >= _NIFTI1_HDR_SIZE:
        endian = _detect_byte_order(raw, is_nifti2=False)
        return _parse_nifti1(raw, source_url=source_url, is_gz=is_gz, bytes_fetched=bytes_fetched, endian=endian)
    else:
        raise ValueError(
            f"Buffer at {source_url!r} does not appear to be a valid NIfTI file "
            f"(magic bytes: {raw[344:348]!r})"
        )


def _parse_nifti1(
    raw: bytes,
    source_url: str,
    is_gz: bool,
    bytes_fetched: int,
    endian: str = "<",
) -> NiftiStreamHeader:
    """Parse a NIfTI-1 348-byte header. ``endian`` (``"<"``/``">"``) is
    detected from ``sizeof_hdr`` by the caller — see ``_detect_byte_order``."""
    # dim[0] = number of dimensions
    ndim = struct.unpack_from(f"{endian}h", raw, 40)[0]
    # dim[1..7] = sizes
    dims = struct.unpack_from(f"{endian}8h", raw, 40)
    ndim = max(min(int(dims[0]), 7), 1)
    shape = tuple(int(d) for d in dims[1: ndim + 1])

    # pixdim[1..7] = voxel sizes
    pixdims = struct.unpack_from(f"{endian}8f", raw, 76)
    voxel_sizes_mm = tuple(float(pixdims[i + 1]) for i in range(min(ndim, 3)))
    tr_s = float(pixdims[4]) if ndim == 4 else None

    # datatype
    datatype = struct.unpack_from(f"{endian}h", raw, 70)[0]
    dtype_np = _NIFTI_DTYPE_MAP.get(datatype, np.float32)

    # vox_offset (where data starts in the file)
    vox_offset_f = struct.unpack_from(f"{endian}f", raw, 108)[0]
    vox_offset = max(int(vox_offset_f), _NIFTI1_HDR_SIZE)

    # scl_slope / scl_inter: intensity calibration applied by nibabel's
    # get_fdata() but easy to miss in a hand-rolled reader — without this,
    # streamed intensities are meaningless raw scanner units for any file
    # that sets non-trivial scaling (common for real scanner exports).
    scl_slope = struct.unpack_from(f"{endian}f", raw, 112)[0]
    scl_inter = struct.unpack_from(f"{endian}f", raw, 116)[0]
    if not np.isfinite(scl_slope):
        scl_slope = 1.0
    if not np.isfinite(scl_inter):
        scl_inter = 0.0

    # sform / qform affine — build from sform if sform_code > 0
    sform_code = struct.unpack_from(f"{endian}h", raw, 254)[0]
    affine = _extract_affine_nifti1(raw, prefer_sform=(sform_code > 0), shape=shape, endian=endian)

    description = raw[148:228].split(b"\x00")[0].decode("latin-1", errors="replace").strip()

    return NiftiStreamHeader(
        ndim=ndim,
        shape=shape,
        voxel_sizes_mm=voxel_sizes_mm,
        affine=affine,
        tr_s=tr_s if (tr_s and tr_s > 0) else None,
        # Carry the on-disk byte order onto the voxel dtype so np.frombuffer
        # decodes the data array correctly for big-endian files too.
        dtype=np.dtype(dtype_np).newbyteorder(endian),
        vox_offset=vox_offset,
        is_gz=is_gz,
        is_nifti2=False,
        description=description,
        source_url=source_url,
        bytes_fetched=bytes_fetched,
        scl_slope=float(scl_slope),
        scl_inter=float(scl_inter),
    )


def _parse_nifti2(
    raw: bytes,
    source_url: str,
    is_gz: bool,
    bytes_fetched: int,
    endian: str = "<",
) -> NiftiStreamHeader:
    """Parse a NIfTI-2 540-byte header. ``endian`` is detected by the caller."""
    datatype = struct.unpack_from(f"{endian}h", raw, 12)[0]
    dtype_np = _NIFTI_DTYPE_MAP.get(datatype, np.float32)

    ndim = struct.unpack_from(f"{endian}q", raw, 16)[0]
    ndim = max(min(int(ndim), 7), 1)
    dims = struct.unpack_from(f"{endian}8q", raw, 16)
    shape = tuple(int(d) for d in dims[1: ndim + 1])

    pixdims = struct.unpack_from(f"{endian}8d", raw, 104)
    voxel_sizes_mm = tuple(float(pixdims[i + 1]) for i in range(min(ndim, 3)))
    tr_s = float(pixdims[4]) if ndim == 4 else None

    vox_offset = int(struct.unpack_from(f"{endian}q", raw, 168)[0])
    vox_offset = max(vox_offset, _NIFTI2_HDR_SIZE)

    scl_slope, scl_inter = struct.unpack_from(f"{endian}2d", raw, 176)
    if not np.isfinite(scl_slope):
        scl_slope = 1.0
    if not np.isfinite(scl_inter):
        scl_inter = 0.0

    # NIfTI-2 affine at offset 216 (4×4 float64)
    affine_vals = struct.unpack_from(f"{endian}16d", raw, 216)
    affine = np.array(affine_vals, dtype=np.float64).reshape(4, 4)

    description = raw[24:104].split(b"\x00")[0].decode("latin-1", errors="replace").strip()

    return NiftiStreamHeader(
        ndim=ndim,
        shape=shape,
        voxel_sizes_mm=voxel_sizes_mm,
        affine=affine,
        tr_s=tr_s if (tr_s and tr_s > 0) else None,
        dtype=np.dtype(dtype_np).newbyteorder(endian),
        vox_offset=vox_offset,
        is_gz=is_gz,
        is_nifti2=True,
        description=description,
        source_url=source_url,
        bytes_fetched=bytes_fetched,
        scl_slope=float(scl_slope),
        scl_inter=float(scl_inter),
    )


def _extract_affine_nifti1(
    raw: bytes,
    prefer_sform: bool,
    shape: tuple[int, ...],
    endian: str = "<",
) -> np.ndarray:
    """Extract a 4×4 affine from NIfTI-1 sform or qform parameters."""
    affine = np.eye(4, dtype=np.float64)
    try:
        if prefer_sform:
            # sform row vectors at offsets 280, 296, 312
            row = []
            for off in (280, 296, 312):
                row.append(struct.unpack_from(f"{endian}4f", raw, off))
            for i, r in enumerate(row):
                affine[i, :] = r
        else:
            # qform quaternion parameters
            qb, qc, qd = struct.unpack_from(f"{endian}3f", raw, 256)
            qx, qy, qz = struct.unpack_from(f"{endian}3f", raw, 268)
            pixdim = struct.unpack_from(f"{endian}8f", raw, 76)
            qfac = 1.0 if pixdim[0] >= 0 else -1.0
            qa = max(0.0, 1.0 - (qb**2 + qc**2 + qd**2)) ** 0.5
            b, c, d, a = qb, qc, qd, qa
            affine[:3, :3] = np.array([
                [a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d),        2*(b*d + a*c)],
                [2*(b*c + a*d),             a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b)],
                [qfac*(2*(b*d - a*c)),      qfac*(2*(c*d + a*b)),  qfac*(a**2 + d**2 - b**2 - c**2)],
            ]) * float(pixdim[1])
            affine[:3, 3] = [qx, qy, qz]
    except Exception:
        pass
    return affine


# ── Streaming gzip decompressor ───────────────────────────────────────────────

class _NeedMoreData(Exception):
    """Raised when the gzip stream ends before the requested offset is reached."""


def _stream_decompress(compressed: bytes, max_bytes: int) -> bytes:
    """Decompress as much gzip data as available, up to ``max_bytes``.

    Raises ``_NeedMoreData`` when the compressed stream ends before
    ``max_bytes`` of decompressed output are produced.
    """
    d = zlib.decompressobj(wbits=47)  # 47 = auto-detect gzip/zlib
    out = bytearray()
    chunk_size = 65536
    buf = memoryview(compressed)
    pos = 0
    while pos < len(buf):
        end = min(pos + chunk_size, len(buf))
        try:
            out.extend(d.decompress(bytes(buf[pos:end])))
        except zlib.error:
            break
        pos = end
        if len(out) >= max_bytes:
            break
    if len(out) < max_bytes and pos >= len(buf):
        raise _NeedMoreData(
            f"Compressed stream exhausted at {len(compressed)} B; "
            f"got {len(out)} decompressed B, need {max_bytes} B."
        )
    return bytes(out)
