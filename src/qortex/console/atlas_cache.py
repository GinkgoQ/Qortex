"""Thread-safe TTL cache with request coalescing (single-flight) for the Atlas API.

The Atlas frontend is a single-page app: opening one dataset workspace remounts
several SPA routes in quick succession (Overview, Evidence, Readiness, Files …),
and each one independently asks the backend for the *same* underlying artifact —
the snapshot manifest or the dataset profile. Every one of those is a multi-second
OpenNeuro round-trip on a cold cache.

A plain ``dict`` cache (the previous approach) has two problems under that access
pattern:

* **No coalescing.** If three tabs mount at once on a cold key, all three miss,
  and all three launch the full fetch in parallel — three identical
  multi-second OpenNeuro round-trips (and 3× the rate-limit pressure) to produce
  one result. This is the classic cache-stampede / thundering-herd failure.
* **Unbounded growth.** Nothing evicts, so a long-lived server accumulates every
  manifest it ever built.

``TTLCache`` fixes both: a per-key lock serializes concurrent misses so exactly
one caller computes while the rest wait and receive that same result, and a bounded
LRU-by-timestamp store caps memory. Cache *hits* take only the short guard lock,
never the per-key lock, so a warm key never blocks behind an in-flight miss for a
different key.

This is a synchronous, blocking cache by design: the Atlas API runs every Qortex
call in a threadpool (``fastapi.concurrency.run_in_threadpool``), so blocking on a
per-key lock parks a worker thread — it never stalls the event loop.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Hashable


class _Entry:
    __slots__ = ("lock", "ts", "value", "ready")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ts = 0.0
        self.value: Any = None
        self.ready = False


class TTLCache:
    """A bounded, thread-safe TTL cache that coalesces concurrent misses.

    Parameters
    ----------
    ttl:
        Seconds a computed value stays fresh. After this it is recomputed on the
        next access.
    maxsize:
        Maximum number of distinct keys retained. When exceeded, the entry with
        the oldest write timestamp is evicted.
    """

    def __init__(self, ttl: float, maxsize: int = 256) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._entries: dict[Hashable, _Entry] = {}
        self._guard = threading.Lock()

    def _slot(self, key: Hashable) -> _Entry:
        """Return the (possibly new) entry for *key*, evicting if over capacity.

        Only the short guard lock is held here — never a compute — so this stays
        fast even while another key's value is being fetched.
        """
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self._maxsize:
                    oldest = min(self._entries, key=lambda k: self._entries[k].ts)
                    del self._entries[oldest]
                entry = _Entry()
                self._entries[key] = entry
            return entry

    def peek(self, key: Hashable) -> Any | None:
        """Return a fresh cached value without ever computing, else ``None``.

        Lets an async route short-circuit a hit without even hopping to the
        threadpool. Returns ``None`` for a miss *or* a stale/in-flight entry.
        """
        with self._guard:
            entry = self._entries.get(key)
        if entry is not None and entry.ready and (time.monotonic() - entry.ts) < self._ttl:
            return entry.value
        return None

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value for *key*, or compute it under single-flight.

        On a hit within TTL, returns immediately. On a miss, exactly one caller
        runs ``compute()`` while any concurrent callers for the same key block on
        the per-key lock and then observe the freshly-stored result — the fetch
        happens once, not once per caller.
        """
        entry = self._slot(key)
        if entry.ready and (time.monotonic() - entry.ts) < self._ttl:
            return entry.value
        with entry.lock:
            # Re-check under the per-key lock: whoever held it before us may have
            # just populated the value, in which case we must not fetch again.
            if entry.ready and (time.monotonic() - entry.ts) < self._ttl:
                return entry.value
            value = compute()
            entry.value = value
            entry.ts = time.monotonic()
            entry.ready = True
            return value

    def invalidate(self, key: Hashable) -> None:
        with self._guard:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._guard:
            self._entries.clear()

    def fresh_values(self) -> list[Any]:
        """Return a point-in-time list of fresh ready values without computing."""
        now = time.monotonic()
        with self._guard:
            return [
                entry.value
                for entry in self._entries.values()
                if entry.ready and (now - entry.ts) < self._ttl
            ]
