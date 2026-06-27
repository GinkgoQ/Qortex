"""Latency profiler for NeuroAI pipelines.

Measures end-to-end and per-stage latency across N windows/batches and
produces a ``LatencyReport`` with p50/p95/p99 statistics.

Tracks:
  - source read latency
  - preprocessing latency
  - inference latency
  - postprocessing / output write latency
  - end-to-end latency
  - dropped windows (timeout / error)

Usage::

    profiler = PipelineProfiler(budget_ms=100.0)
    profiler.start_source_read()
    ...
    profiler.end_source_read()
    profiler.start_inference()
    ...
    profiler.end_inference()
    profiler.commit_window()

    report = profiler.report()
    print(report.summary())
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from qortex.neuroai.contracts import LatencyBreakdown, LatencyReport

log = logging.getLogger(__name__)


@dataclass
class _WindowTiming:
    source_read_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    output_write_ms: float = 0.0
    total_ms: float = 0.0
    dropped: bool = False
    error: str | None = None


class PipelineProfiler:
    """Accumulate per-stage timings and produce a LatencyReport.

    Parameters
    ----------
    budget_ms:
        Target end-to-end latency budget.  When the p95 exceeds this value
        the report status is ``FAIL``.
    """

    def __init__(self, budget_ms: float | None = None) -> None:
        self._budget = budget_ms
        self._windows: list[_WindowTiming] = []
        self._current: _WindowTiming = _WindowTiming()
        self._stage_start: float = 0.0

    # ── Stage markers ─────────────────────────────────────────────────────────

    def start_source_read(self) -> None:
        self._stage_start = time.perf_counter()

    def end_source_read(self) -> None:
        self._current.source_read_ms += _elapsed_ms(self._stage_start)

    def start_preprocess(self) -> None:
        self._stage_start = time.perf_counter()

    def end_preprocess(self) -> None:
        self._current.preprocess_ms += _elapsed_ms(self._stage_start)

    def start_inference(self) -> None:
        self._stage_start = time.perf_counter()

    def end_inference(self) -> None:
        self._current.inference_ms += _elapsed_ms(self._stage_start)

    def start_postprocess(self) -> None:
        self._stage_start = time.perf_counter()

    def end_postprocess(self) -> None:
        self._current.postprocess_ms += _elapsed_ms(self._stage_start)

    def start_output_write(self) -> None:
        self._stage_start = time.perf_counter()

    def end_output_write(self) -> None:
        self._current.output_write_ms += _elapsed_ms(self._stage_start)

    def commit_window(self, *, dropped: bool = False, error: str | None = None) -> None:
        """Finalize the current window's timing and start a new one."""
        w = self._current
        w.dropped = dropped
        w.error = error
        w.total_ms = (
            w.source_read_ms
            + w.preprocess_ms
            + w.inference_ms
            + w.postprocess_ms
            + w.output_write_ms
        )
        self._windows.append(w)
        self._current = _WindowTiming()

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self) -> LatencyReport:
        """Build a ``LatencyReport`` from accumulated timings."""
        if not self._windows:
            return LatencyReport(status="UNKNOWN")

        all_totals = [w.total_ms for w in self._windows if not w.dropped]
        if not all_totals:
            return LatencyReport(n_windows=len(self._windows),
                                  n_dropped=len(self._windows),
                                  status="FAIL")

        all_totals_sorted = sorted(all_totals)
        n = len(all_totals_sorted)
        p50 = _percentile(all_totals_sorted, 50)
        p95 = _percentile(all_totals_sorted, 95)
        p99 = _percentile(all_totals_sorted, 99)
        mean = statistics.mean(all_totals)
        n_dropped = sum(1 for w in self._windows if w.dropped)

        breakdown = LatencyBreakdown(
            source_read_ms=statistics.mean(w.source_read_ms for w in self._windows if not w.dropped),
            preprocess_ms=statistics.mean(w.preprocess_ms for w in self._windows if not w.dropped),
            inference_ms=statistics.mean(w.inference_ms for w in self._windows if not w.dropped),
            postprocess_ms=statistics.mean(w.postprocess_ms for w in self._windows if not w.dropped),
            output_write_ms=statistics.mean(w.output_write_ms for w in self._windows if not w.dropped),
            total_ms=mean,
        )

        status: str
        if self._budget is None:
            status = "UNKNOWN"
        elif p95 <= self._budget:
            status = "PASS"
        else:
            status = "FAIL"

        return LatencyReport(
            n_windows=len(self._windows),
            n_dropped=n_dropped,
            budget_ms=self._budget,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            mean_ms=mean,
            breakdown=breakdown,
            status=status,
        )

    def reset(self) -> None:
        self._windows.clear()
        self._current = _WindowTiming()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
