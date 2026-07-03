"""In-process background job registry for the Qortex Atlas console API.

Several real Qortex operations are synchronous, network-bound, and slow
enough that they must never run inline inside a FastAPI request handler:
``Dataset.download()``, ``DatasetInspector.inspect(level="deep")``,
``DatasetSelector.find(tier3_events=True)``, and ``catalog.refresh()`` all do
blocking ``httpx`` calls (potentially dozens of them) on the calling thread.

This module is a minimal, dependency-free job tracker: submit a callable,
get a job id back immediately, poll ``/jobs/{id}`` for progress. It runs
work in a thread pool (not asyncio tasks) because the wrapped Qortex calls
are themselves synchronous/blocking, not async.
"""

from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qortex-atlas-job")


@dataclass
class Job:
    id: str
    label: str
    status: str = "running"  # running | done | error
    progress: int = 0
    result: Any = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    log: list[str] = field(default_factory=list)
    _future: Future | None = field(default=None, repr=False)


_JOBS: dict[str, Job] = {}


def submit(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    report_progress: bool = False,
    **kwargs: Any,
) -> Job:
    """Submit *fn* to run in the background.

    ``report_progress=True`` opts a caller into live progress: *fn* is then
    called with an extra ``on_progress(done, total)`` keyword, which updates
    ``job.progress``/``job.log`` in real time as the wrapped Qortex call
    reports it (e.g. ``Dataset.download()``'s own ``on_progress`` hook) —
    instead of a job only ever reporting 0% then jumping straight to 100%.
    Default False keeps every existing job kind (catalog refresh, deep
    inspect, ...) working unchanged, since their wrapped functions don't
    accept an ``on_progress`` keyword.
    """
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, label=label)
    job.log.append(f"queued: {label}")
    _JOBS[job_id] = job

    def _on_progress(done: int, total: int) -> None:
        pct = max(0, min(100, round(done / total * 100))) if total else 0
        # `progress` updates every call (cheap int); `log` only on a new
        # percentage point so a several-thousand-file download doesn't
        # append several thousand near-duplicate lines.
        if pct != job.progress:
            job.progress = pct
            job.log.append(f"{done}/{total} files ({pct}%)")

    def _run() -> None:
        job.log.append("started")
        try:
            if report_progress:
                job.result = fn(*args, on_progress=_on_progress, **kwargs)
            else:
                job.result = fn(*args, **kwargs)
            job.status = "done"
            job.progress = 100
            job.log.append("completed")
        except Exception as exc:  # noqa: BLE001 - surfaced to the API caller verbatim
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(traceback.format_exc(limit=6))
        finally:
            job.finished_at = time.time()

    job._future = _POOL.submit(_run)
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    return sorted(_JOBS.values(), key=lambda j: j.started_at, reverse=True)[:limit]


def to_public(job: Job) -> dict[str, Any]:
    return {
        "id": job.id, "label": job.label, "status": job.status, "progress": job.progress,
        "started_at": job.started_at, "finished_at": job.finished_at, "log": job.log,
        "error": job.error,
    }
