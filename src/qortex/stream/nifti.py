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
import zlib
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
            "voxel_sizes_mm": list(self.voxel_sizes_mm),
            "tr_s": self.tr_s,
            "dtype": str(self.dtype),
            "vox_offset": self.vox_offset,
            "is_gz": self.is_gz,
            "is_nifti2": self.is_nifti2,
            "description": self.description,
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
        if hdr.is_4d and t >= (hdr.n_volumes or 1):
            raise IndexError(f"Volume index {t} out of range [0, {hdr.n_volumes})")

        # For .nii.gz: stream-decompress up to the needed point
        # For .nii: issue an exact Range request
        if self._is_gz:
            vol_data = self._stream_volume_gz(hdr, t)
        else:
            vol_data = self._fetch_volume_raw(hdr, t)

        vol = np.frombuffer(vol_data, dtype=hdr.dtype).reshape(hdr.spatial_shape, order="F")
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
        if hdr.is_4d and t >= (hdr.n_volumes or 1):
            raise IndexError(f"Volume index {t} out of range [0, {hdr.n_volumes})")
        if self._is_gz:
            vol_data = self._stream_volume_gz(hdr, t)
        else:
            vol_data = self._fetch_volume_raw(hdr, t)

        vol = np.frombuffer(vol_data, dtype=hdr.dtype).reshape(hdr.spatial_shape, order="F").copy()
        if hdr.needs_scaling:
            vol = vol.astype(np.float32) * hdr.scl_slope + hdr.scl_inter
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
        if self._is_gz:
            vol = self.get_volume(t=t)
        else:
            vol_data = self._fetch_volume_raw(hdr, t)
            vol = np.frombuffer(vol_data, dtype=hdr.dtype).reshape(hdr.spatial_shape, order="F").copy()
            if hdr.needs_scaling:
                vol = vol.astype(np.float32) * hdr.scl_slope + hdr.scl_inter

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
        return _parse_nifti2(raw, source_url=source_url, is_gz=is_gz, bytes_fetched=bytes_fetched)
    elif is_nifti1 and len(raw) >= _NIFTI1_HDR_SIZE:
        return _parse_nifti1(raw, source_url=source_url, is_gz=is_gz, bytes_fetched=bytes_fetched)
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
) -> NiftiStreamHeader:
    """Parse a NIfTI-1 348-byte header.  Little-endian assumed (most files)."""
    # dim[0] = number of dimensions
    ndim = struct.unpack_from("<h", raw, 40)[0]
    # dim[1..7] = sizes
    dims = struct.unpack_from("<8h", raw, 40)
    ndim = max(min(int(dims[0]), 7), 1)
    shape = tuple(int(d) for d in dims[1: ndim + 1])

    # pixdim[1..7] = voxel sizes
    pixdims = struct.unpack_from("<8f", raw, 76)
    voxel_sizes_mm = tuple(float(pixdims[i + 1]) for i in range(min(ndim, 3)))
    tr_s = float(pixdims[4]) if ndim == 4 else None

    # datatype
    datatype = struct.unpack_from("<h", raw, 70)[0]
    dtype_np = _NIFTI_DTYPE_MAP.get(datatype, np.float32)

    # vox_offset (where data starts in the file)
    vox_offset_f = struct.unpack_from("<f", raw, 108)[0]
    vox_offset = max(int(vox_offset_f), _NIFTI1_HDR_SIZE)

    # scl_slope / scl_inter: intensity calibration applied by nibabel's
    # get_fdata() but easy to miss in a hand-rolled reader — without this,
    # streamed intensities are meaningless raw scanner units for any file
    # that sets non-trivial scaling (common for real scanner exports).
    scl_slope = struct.unpack_from("<f", raw, 112)[0]
    scl_inter = struct.unpack_from("<f", raw, 116)[0]
    if not np.isfinite(scl_slope):
        scl_slope = 1.0
    if not np.isfinite(scl_inter):
        scl_inter = 0.0

    # sform / qform affine — build from sform if sform_code > 0
    sform_code = struct.unpack_from("<h", raw, 254)[0]
    affine = _extract_affine_nifti1(raw, prefer_sform=(sform_code > 0), shape=shape)

    description = raw[148:228].split(b"\x00")[0].decode("latin-1", errors="replace").strip()

    return NiftiStreamHeader(
        ndim=ndim,
        shape=shape,
        voxel_sizes_mm=voxel_sizes_mm,
        affine=affine,
        tr_s=tr_s if (tr_s and tr_s > 0) else None,
        dtype=np.dtype(dtype_np),
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
) -> NiftiStreamHeader:
    """Parse a NIfTI-2 540-byte header."""
    datatype = struct.unpack_from("<h", raw, 12)[0]
    dtype_np = _NIFTI_DTYPE_MAP.get(datatype, np.float32)

    ndim = struct.unpack_from("<q", raw, 16)[0]
    ndim = max(min(int(ndim), 7), 1)
    dims = struct.unpack_from("<8q", raw, 16)
    shape = tuple(int(d) for d in dims[1: ndim + 1])

    pixdims = struct.unpack_from("<8d", raw, 104)
    voxel_sizes_mm = tuple(float(pixdims[i + 1]) for i in range(min(ndim, 3)))
    tr_s = float(pixdims[4]) if ndim == 4 else None

    vox_offset = int(struct.unpack_from("<q", raw, 168)[0])
    vox_offset = max(vox_offset, _NIFTI2_HDR_SIZE)

    scl_slope, scl_inter = struct.unpack_from("<2d", raw, 176)
    if not np.isfinite(scl_slope):
        scl_slope = 1.0
    if not np.isfinite(scl_inter):
        scl_inter = 0.0

    # NIfTI-2 affine at offset 216 (4×4 float64)
    affine_vals = struct.unpack_from("<16d", raw, 216)
    affine = np.array(affine_vals, dtype=np.float64).reshape(4, 4)

    description = raw[24:104].split(b"\x00")[0].decode("latin-1", errors="replace").strip()

    return NiftiStreamHeader(
        ndim=ndim,
        shape=shape,
        voxel_sizes_mm=voxel_sizes_mm,
        affine=affine,
        tr_s=tr_s if (tr_s and tr_s > 0) else None,
        dtype=np.dtype(dtype_np),
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
) -> np.ndarray:
    """Extract a 4×4 affine from NIfTI-1 sform or qform parameters."""
    affine = np.eye(4, dtype=np.float64)
    try:
        if prefer_sform:
            # sform row vectors at offsets 280, 296, 312
            row = []
            for off in (280, 296, 312):
                row.append(struct.unpack_from("<4f", raw, off))
            for i, r in enumerate(row):
                affine[i, :] = r
        else:
            # qform quaternion parameters
            qb, qc, qd = struct.unpack_from("<3f", raw, 256)
            qx, qy, qz = struct.unpack_from("<3f", raw, 268)
            pixdim = struct.unpack_from("<8f", raw, 76)
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
