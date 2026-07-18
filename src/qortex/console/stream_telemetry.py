"""Bounded measured telemetry for Atlas imaging stream requests."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


class StreamTelemetry:
    def __init__(self, max_events: int = 1000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **event,
            })

    def report(self, *, limit: int = 100) -> dict[str, Any]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1, 1000]")
        with self._lock:
            events = list(self._events)
        selected = events[-limit:]
        elapsed = [float(event["elapsed_seconds"]) for event in selected]
        output_bytes = sum(int(event.get("response_data_bytes", 0)) for event in selected)
        fetched_bytes = sum(int(event.get("cache_bytes_inserted_delta", 0)) for event in selected)
        hit_bytes = sum(int(event.get("cache_hit_bytes_delta", 0)) for event in selected)
        decoded_hits = sum(int(event.get("decoded_volume_hits_delta", 0)) for event in selected)
        decoded_misses = sum(int(event.get("decoded_volume_misses_delta", 0)) for event in selected)
        return {
            "events": selected,
            "event_count": len(selected),
            "total_recorded": len(events),
            "truncated": len(events) > len(selected),
            "summary": {
                "median_latency_seconds": sorted(elapsed)[len(elapsed) // 2] if elapsed else None,
                "mean_latency_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
                "response_data_bytes": output_bytes,
                "cache_bytes_inserted": fetched_bytes,
                "cache_hit_bytes": hit_bytes,
                "cache_byte_efficiency": hit_bytes / (hit_bytes + fetched_bytes) if hit_bytes + fetched_bytes else None,
                "decoded_volume_hits": decoded_hits,
                "decoded_volume_misses": decoded_misses,
                "decoded_volume_hit_rate": decoded_hits / (decoded_hits + decoded_misses) if decoded_hits + decoded_misses else None,
            },
            "measurement_scope": (
                "Process-local NIfTI slice-data requests only. Cache byte counters measure bytes inserted "
                "and served from the streamer's byte-range cache. Decoded-volume hits separately measure "
                "warm compressed-volume reuse; these are not whole-dataset download totals."
            ),
        }


stream_telemetry = StreamTelemetry()

__all__ = ["StreamTelemetry", "stream_telemetry"]
