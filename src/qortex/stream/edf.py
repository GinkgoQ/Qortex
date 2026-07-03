"""Remote EDF/BDF/GDF streamer — byte-range channel and epoch access.

Implements byte-level random access into European Data Format (EDF/EDF+)
and BDF (24-bit variant) files without downloading the full recording.

EDF structure (all offsets are deterministic from the fixed 256-byte header):

  Global header  : 256 bytes  (recording info, start time, n_records, record_dur, n_channels)
  Channel headers: 256 * n_channels bytes  (labels, physical/digital ranges, sampling rates)
  Data records   : n_records × Σ(n_samples_per_record[ch]) × bytes_per_sample

Since EDF/BDF data records have fixed, predictable sizes, any (channel, time)
window maps to an exact byte range — enabling single Range requests for
individual channels or epochs without any sequential decompression.

For BDF (24-bit samples) the byte width is 3 instead of 2.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qortex.stream._cache import MemoryCache, make_cache

log = logging.getLogger(__name__)

_EDF_GLOBAL_HDR  = 256     # bytes
_EDF_CHANNEL_HDR = 256     # bytes per channel


@dataclass
class EDFStreamHeader:
    """EDF/BDF header decoded from the first ``256 + 256*n_channels`` bytes."""

    version: str                       # "0" for EDF, "24BIT" for BDF
    patient_id: str
    recording_id: str
    start_date: str                    # DD.MM.YY
    start_time: str                    # HH.MM.SS
    n_header_bytes: int
    reserved: str                      # "EDF+C", "EDF+D", "" etc.
    n_data_records: int                # -1 if unknown (online recording)
    record_duration_s: float           # seconds per data record
    n_channels: int
    channel_labels: list[str]
    physical_min: list[float]
    physical_max: list[float]
    digital_min: list[int]
    digital_max: list[int]
    samples_per_record: list[int]      # n_samples in each data record
    prefilters: list[str]
    transducer: list[str]
    units: list[str]
    is_bdf: bool
    source_url: str
    bytes_per_sample: int = field(init=False)

    def __post_init__(self) -> None:
        self.bytes_per_sample = 3 if self.is_bdf else 2

    @property
    def sampling_rates(self) -> list[float]:
        """Sampling rate per channel in Hz."""
        if self.record_duration_s <= 0:
            return [0.0] * self.n_channels
        return [
            s / self.record_duration_s
            for s in self.samples_per_record
        ]

    @property
    def duration_s(self) -> float | None:
        """Total recording duration in seconds (None if n_data_records is unknown)."""
        if self.n_data_records < 0:
            return None
        return self.n_data_records * self.record_duration_s

    @property
    def record_size_bytes(self) -> int:
        """Bytes in one complete data record across all channels."""
        return sum(self.samples_per_record) * self.bytes_per_sample

    def channel_index(self, label: str) -> int:
        """Return 0-based channel index for a label (case-insensitive prefix match)."""
        label_lower = label.lower().strip()
        for i, ch in enumerate(self.channel_labels):
            if ch.lower().strip() == label_lower:
                return i
        # prefix match
        for i, ch in enumerate(self.channel_labels):
            if ch.lower().strip().startswith(label_lower):
                return i
        raise KeyError(
            f"Channel {label!r} not found. Available: {self.channel_labels}"
        )

    def gain(self, ch_idx: int) -> float:
        """Physical gain for channel ``ch_idx``: maps digital units → physical units."""
        drange = self.digital_max[ch_idx] - self.digital_min[ch_idx]
        prange = self.physical_max[ch_idx] - self.physical_min[ch_idx]
        return prange / drange if drange != 0 else 1.0

    def offset(self, ch_idx: int) -> float:
        """Physical offset for channel ``ch_idx``."""
        g = self.gain(ch_idx)
        return self.physical_max[ch_idx] / g - self.digital_max[ch_idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "patient_id": self.patient_id,
            "n_channels": self.n_channels,
            "n_data_records": self.n_data_records,
            "record_duration_s": self.record_duration_s,
            "duration_s": self.duration_s,
            "channel_labels": self.channel_labels,
            "sampling_rates": self.sampling_rates,
            "units": self.units,
            "is_bdf": self.is_bdf,
        }

    def __str__(self) -> str:
        fmt = "BDF" if self.is_bdf else "EDF"
        dur = f"{self.duration_s:.1f}s" if self.duration_s else "?"
        srs = sorted(set(f"{r:.0f}" for r in self.sampling_rates))
        return (
            f"{fmt} {self.n_channels}ch duration={dur} "
            f"srate={'/'.join(srs)} Hz"
        )


class EDFStreamer:
    """Stream EDF/BDF data from a remote URL using byte-range requests.

    Parameters
    ----------
    url_or_path:
        HTTP/HTTPS URL or local filesystem path to an EDF or BDF file.
    cache_dir:
        Directory for byte-range cache.  Defaults to ``~/.qortex/stream_cache``.
    cache_backend:
        ``"memory"`` or ``"disk"``.

    Examples
    --------
    >>> edf = EDFStreamer("https://cdn.openneuro.org/.../sub-01_eeg.edf")
    >>> hdr = edf.header()
    >>> print(hdr)   # "EDF 64ch duration=480.0s srate=256 Hz"
    >>> epoch = edf.get_epoch(tmin=10.0, tmax=20.0, channels=["Cz", "Pz"])
    >>> print(epoch.shape)   # (2, 2560)  — 2 channels × 2560 samples @ 256 Hz
    >>> raw_all = edf.get_channel("Cz")   # full channel vector
    """

    def __init__(
        self,
        url_or_path: str | Path,
        *,
        cache_dir: Path | None = None,
        cache_backend: str = "memory",
    ) -> None:
        self._url = str(url_or_path)
        self._is_local = not self._url.startswith(("http://", "https://", "s3://"))
        self._cache = make_cache(backend=cache_backend, cache_dir=cache_dir)
        self._header: EDFStreamHeader | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def header(self, *, force_refresh: bool = False) -> EDFStreamHeader:
        """Fetch and decode the EDF/BDF header.  Cached after first call."""
        if self._header is not None and not force_refresh:
            return self._header
        raw = self._fetch_bytes(0, _EDF_GLOBAL_HDR, label="global-header")
        n_channels = int(raw[252:256].strip())
        channel_hdr = self._fetch_bytes(
            _EDF_GLOBAL_HDR,
            _EDF_CHANNEL_HDR * n_channels,
            label="channel-headers",
        )
        self._header = _parse_edf_header(raw + channel_hdr, source_url=self._url)
        return self._header

    def get_epoch(
        self,
        tmin: float,
        tmax: float,
        *,
        channels: list[str] | None = None,
        apply_gain: bool = True,
    ) -> np.ndarray:
        """Stream a time-bounded epoch for selected channels.

        Parameters
        ----------
        tmin, tmax:
            Epoch boundaries in seconds from recording start.
        channels:
            Channel labels to include.  ``None`` = all channels.
        apply_gain:
            Convert digital units to physical units (µV, mV, etc.) using
            the EDF header gain/offset (default True).

        Returns
        -------
        np.ndarray
            Shape ``(n_channels, n_samples)`` in physical units if
            ``apply_gain=True``, or in raw digital units otherwise.
        """
        hdr = self.header()
        if tmin < 0 or (hdr.duration_s and tmax > hdr.duration_s):
            raise ValueError(
                f"Epoch [{tmin}, {tmax}]s out of recording range "
                f"[0, {hdr.duration_s}]s."
            )

        ch_indices = (
            [hdr.channel_index(c) for c in channels]
            if channels is not None
            else list(range(hdr.n_channels))
        )

        # Which data records overlap our epoch?
        rec_dur = hdr.record_duration_s
        first_rec = max(0, int(tmin / rec_dur))
        last_rec = min(
            hdr.n_data_records - 1 if hdr.n_data_records >= 0 else 10**9,
            int(np.ceil(tmax / rec_dur)),
        )
        n_records = last_rec - first_rec + 1

        # Fetch the exact byte range covering the needed data records
        rec_size = hdr.record_size_bytes
        fetch_start = hdr.n_header_bytes + first_rec * rec_size
        fetch_len   = n_records * rec_size
        raw_data = self._fetch_bytes(fetch_start, fetch_len, label=f"epoch-{tmin:.1f}-{tmax:.1f}")

        # Decode records channel by channel
        bps = hdr.bytes_per_sample
        channel_data: list[list[int]] = [[] for _ in range(len(ch_indices))]

        for rec_i in range(n_records):
            rec_offset = rec_i * rec_size
            ch_read_offset = rec_offset
            for ch_i in range(hdr.n_channels):
                n_s = hdr.samples_per_record[ch_i]
                ch_bytes = n_s * bps
                if ch_i in ch_indices:
                    local_idx = ch_indices.index(ch_i)
                    chunk = raw_data[ch_read_offset: ch_read_offset + ch_bytes]
                    samples = _decode_samples(chunk, n_s, bps)
                    channel_data[local_idx].extend(samples)
                ch_read_offset += ch_bytes

        # Trim to exact requested time range
        result_rows: list[np.ndarray] = []
        for local_idx, ch_i in enumerate(ch_indices):
            srate = hdr.sampling_rates[ch_i]
            samples = np.array(channel_data[local_idx], dtype=np.float32)
            # Convert from record time to global sample indices
            rec_start_time = first_rec * rec_dur
            start_sample = int((tmin - rec_start_time) * srate)
            end_sample   = start_sample + int((tmax - tmin) * srate)
            slc = samples[start_sample: end_sample]
            if apply_gain:
                g = hdr.gain(ch_i)
                off = hdr.offset(ch_i)
                slc = slc * g + off * g
            result_rows.append(slc)

        # Pad or trim to common length (different srates may give different lengths)
        min_len = min(len(r) for r in result_rows)
        return np.stack([r[:min_len] for r in result_rows], axis=0)

    def get_channel(
        self,
        channel: str,
        *,
        tmin: float = 0.0,
        tmax: float | None = None,
        apply_gain: bool = True,
    ) -> np.ndarray:
        """Stream the full time series for one channel.

        For large recordings this still requires reading the entire file in
        channel-interleaved chunks.  Use ``get_epoch`` with a time window for
        efficiency.
        """
        hdr = self.header()
        t_end = tmax if tmax is not None else (hdr.duration_s or 3600.0)
        epoch = self.get_epoch(tmin=tmin, tmax=t_end, channels=[channel], apply_gain=apply_gain)
        return epoch[0]

    def channel_info(self) -> list[dict[str, Any]]:
        """Return a list of per-channel metadata dicts."""
        hdr = self.header()
        return [
            {
                "index": i,
                "label": hdr.channel_labels[i],
                "unit": hdr.units[i],
                "srate_hz": hdr.sampling_rates[i],
                "physical_range": [hdr.physical_min[i], hdr.physical_max[i]],
                "digital_range": [hdr.digital_min[i], hdr.digital_max[i]],
                "prefilter": hdr.prefilters[i],
                "transducer": hdr.transducer[i],
            }
            for i in range(hdr.n_channels)
        ]

    # ── Private ───────────────────────────────────────────────────────────

    def _fetch_bytes(self, start: int, length: int, label: str = "") -> bytes:
        key = f"{self._url}|{start}|{length}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._is_local:
            with open(self._url, "rb") as fh:
                fh.seek(start)
                data = fh.read(length)
        else:
            from qortex.client.remote import get_shared_gateway
            gw = get_shared_gateway()
            # A true bytes=start-end range — not "fetch the whole prefix up
            # to here and slice off the part before `start` locally", which
            # for an epoch read far into a long recording could mean
            # downloading everything before it just to discard it.
            data = gw.fetch_bytes(self._url, range_start=start, range_bytes=length)
        self._cache.put(key, data)
        return data


# ── EDF parsing ───────────────────────────────────────────────────────────────

def _parse_edf_header(raw: bytes, source_url: str) -> EDFStreamHeader:
    """Parse the concatenated global + channel header block."""
    g = raw[:_EDF_GLOBAL_HDR]
    version      = g[0:8].decode("ascii", errors="replace").strip()
    patient_id   = g[8:88].decode("ascii", errors="replace").strip()
    recording_id = g[88:168].decode("ascii", errors="replace").strip()
    start_date   = g[168:176].decode("ascii", errors="replace").strip()
    start_time   = g[176:184].decode("ascii", errors="replace").strip()
    n_hdr_bytes  = int(g[184:192].strip())
    reserved     = g[192:236].decode("ascii", errors="replace").strip()
    n_records    = int(g[236:244].strip())
    rec_dur      = float(g[244:252].strip())
    n_channels   = int(g[252:256].strip())

    is_bdf = version.startswith("24BIT") or version.startswith("-1  ") or (
        len(source_url) > 0 and source_url.lower().endswith(".bdf")
    )

    def _field(block: bytes, field_len: int, start: int) -> list[str]:
        out = []
        for i in range(n_channels):
            off = start + i * field_len
            out.append(block[off: off + field_len].decode("ascii", errors="replace").strip())
        return out

    ch = raw[_EDF_GLOBAL_HDR:]
    labels    = _field(ch, 16, 0)
    transducer= _field(ch, 80, 16 * n_channels)
    units     = _field(ch, 8,  96 * n_channels)
    phys_min  = [float(x) for x in _field(ch, 8,  104 * n_channels)]
    phys_max  = [float(x) for x in _field(ch, 8,  112 * n_channels)]
    dig_min   = [int(x)   for x in _field(ch, 8,  120 * n_channels)]
    dig_max   = [int(x)   for x in _field(ch, 8,  128 * n_channels)]
    prefilter = _field(ch, 80, 136 * n_channels)
    spr       = [int(x)   for x in _field(ch, 8,  216 * n_channels)]

    return EDFStreamHeader(
        version=version,
        patient_id=patient_id,
        recording_id=recording_id,
        start_date=start_date,
        start_time=start_time,
        n_header_bytes=n_hdr_bytes,
        reserved=reserved,
        n_data_records=n_records,
        record_duration_s=rec_dur,
        n_channels=n_channels,
        channel_labels=labels,
        physical_min=phys_min,
        physical_max=phys_max,
        digital_min=dig_min,
        digital_max=dig_max,
        samples_per_record=spr,
        prefilters=prefilter,
        transducer=transducer,
        units=units,
        is_bdf=is_bdf,
        source_url=source_url,
    )


def _decode_samples(chunk: bytes, n_samples: int, bps: int) -> list[int]:
    """Decode ``n_samples`` EDF (2-byte) or BDF (3-byte) signed integers."""
    if bps == 2:
        return list(struct.unpack_from(f"<{n_samples}h", chunk))
    # BDF: 3-byte little-endian signed integers
    out = []
    for i in range(n_samples):
        b0, b1, b2 = chunk[i*3], chunk[i*3 + 1], chunk[i*3 + 2]
        val = b0 | (b1 << 8) | (b2 << 16)
        if val >= 0x800000:
            val -= 0x1000000
        out.append(val)
    return out
