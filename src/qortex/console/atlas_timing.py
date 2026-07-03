"""Real, evidence-based ETA estimation for the Atlas console API.

Some Qortex operations are genuinely slow (a full manifest fetch for a
1000+ file dataset, a live OpenNeuro search) for reasons outside the UI's
control — network round-trips and real computation. Rather than a fabricated
progress percentage or a literal network-speed probe (unreliable, and
answers the wrong question — what matters is *this operation, on datasets
like this one*, not raw bandwidth), this tracks how long each operation
has actually taken on this machine, per dataset, and reports back a median/
p90 estimate from real observed history. No estimate is offered until at
least one real sample exists; nothing here is guessed.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator

_MAX_SAMPLES = 12
_history: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))
_lock = Lock()


def record(operation: str, key: str, duration_s: float) -> None:
    """Record a real observed duration for *operation* on *key* (usually a dataset id)."""
    with _lock:
        _history[(operation, key)].append(duration_s)
        # Also roll into a global, per-operation bucket ("*") so a dataset
        # seen for the first time still gets an estimate, from how this
        # operation has gone on *other* datasets.
        _history[(operation, "*")].append(duration_s)


@contextmanager
def timed(operation: str, key: str) -> Iterator[None]:
    """Context manager: time a block and record it — one line at each call site."""
    t0 = time.monotonic()
    try:
        yield
    finally:
        record(operation, key, time.monotonic() - t0)


def estimate(operation: str, key: str) -> dict[str, Any]:
    """Return a real historical estimate, preferring this exact dataset's
    own history and falling back to the operation's global history."""
    with _lock:
        specific = list(_history.get((operation, key), ()))
        general = list(_history.get((operation, "*"), ()))
    samples = specific if len(specific) >= 2 else general
    if not samples:
        return {"has_estimate": False}
    ordered = sorted(samples)
    n = len(ordered)
    median = ordered[n // 2]
    p90 = ordered[min(n - 1, max(0, round(n * 0.9) - 1))]
    return {
        "has_estimate": True,
        "median_s": round(median, 1),
        "p90_s": round(p90, 1),
        "n_samples": len(specific),
        "n_samples_used": n,
        "scope": "dataset" if samples is specific else "operation",
    }
