"""Measured byte-cache and bounded stream telemetry contracts."""

from qortex.console.stream_telemetry import StreamTelemetry
from qortex.stream._cache import MemoryCache


def test_memory_cache_reports_measured_byte_counters() -> None:
    cache = MemoryCache(maxsize=2, ttl=60)

    assert cache.get("slice") is None
    cache.put("slice", b"1234")
    assert cache.get("slice") == b"1234"

    assert cache.stats() == {
        "size": 1,
        "maxsize": 2,
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
        "hit_bytes": 4,
        "bytes_inserted": 4,
        "resident_bytes": 4,
    }


def test_stream_telemetry_uses_recorded_bytes_and_latency() -> None:
    telemetry = StreamTelemetry(max_events=2)
    telemetry.record({
        "elapsed_seconds": 0.2,
        "response_data_bytes": 100,
        "cache_bytes_inserted_delta": 80,
        "cache_hit_bytes_delta": 0,
    })
    telemetry.record({
        "elapsed_seconds": 0.01,
        "response_data_bytes": 100,
        "cache_bytes_inserted_delta": 0,
        "cache_hit_bytes_delta": 80,
    })

    report = telemetry.report()

    assert report["event_count"] == 2
    assert report["summary"]["response_data_bytes"] == 200
    assert report["summary"]["cache_byte_efficiency"] == 0.5

