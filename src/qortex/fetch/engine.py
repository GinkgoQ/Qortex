"""Download engine — orchestrates async parallel file downloads.

Key design:
  - One shared ``httpx.AsyncClient`` for the entire download session.
  - Two semaphores: HEAD pool (large) and GET pool (user-controlled).
  - ``asyncio.gather(..., return_exceptions=True)`` collects ALL results
    before reporting success; no file is marked done until the gather returns.
  - Per-file failures are isolated: one broken file does not abort others.
  - The overall progress bar is updated per chunk, not per file.
  - Never prints "success" before all tasks have settled.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from qortex._internal.progress import bytes_bar, msg
from qortex.client.transport import build_async_client
from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import (
    DownloadPlan,
    DownloadRecord,
    DownloadResult,
    FailedRecord,
    FileRecord,
)
from qortex.fetch.backends.http import HTTPBackend
from qortex.lake.registry import LocalRegistry
from qortex.plan.lock import LockFile


class DownloadEngine:
    """Execute a DownloadPlan and return a DownloadResult."""

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()

    # ── Sync entry point (wraps async) ────────────────────────────────────

    def execute(self, plan: DownloadPlan) -> DownloadResult:
        """Execute *plan* synchronously.  Compatible with Jupyter notebooks."""
        try:
            asyncio.get_running_loop()
            # Already inside an event loop (Jupyter) — schedule as a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._run(plan))
                return future.result()
        except RuntimeError:
            return asyncio.run(self._run(plan))

    # ── Async core ────────────────────────────────────────────────────────

    async def _run(self, plan: DownloadPlan) -> DownloadResult:
        cfg = self._cfg
        target_dir = plan.target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialise lockfile
        lock = LockFile.from_plan(plan)
        if lock.path.exists():
            lock = LockFile.load(lock.path)
        lock.save()

        # Filter to files that still need downloading
        downloadable = [
            f for f in plan.files
            if not f.is_dir and f.urls and not lock.is_present(f.path)
        ]

        total_bytes = sum(f.size or 0 for f in downloadable)

        msg(
            f"Downloading {plan.dataset_id} (snapshot {plan.snapshot}): "
            f"{len(downloadable)} files, ~{total_bytes / 1e9:.2f} GB",
            emoji="📥",
        )

        t_start = time.monotonic()
        downloaded: list[DownloadRecord] = []
        skipped: list[DownloadRecord] = []
        failed: list[FailedRecord] = []

        # Files already present (lock says so)
        for f in plan.files:
            if not f.is_dir and lock.is_present(f.path):
                local = target_dir / f.path
                skipped.append(DownloadRecord(
                    file=f,
                    local_path=local,
                    bytes_written=f.size or 0,
                    elapsed=0.0,
                    from_cache=True,
                ))

        if not downloadable:
            msg("All files already present.", emoji="✅")
            return DownloadResult(
                plan=plan,
                downloaded=downloaded,
                skipped=skipped,
                failed=failed,
                bytes_downloaded=0,
                elapsed=time.monotonic() - t_start,
            )

        sem_get = asyncio.Semaphore(cfg.max_concurrent_downloads)
        sem_head = asyncio.Semaphore(cfg.max_concurrent_heads)

        async with build_async_client(cfg) as client:
            with bytes_bar(total_bytes, desc="Overall") as overall_bar:
                backend = HTTPBackend(
                    client=client,
                    sem_get=sem_get,
                    sem_head=sem_head,
                    config=cfg,
                    overall_progress=overall_bar,
                )

                tasks = [
                    self._download_one(backend, f, target_dir, lock)
                    for f in downloadable
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Collect results ── (only after ALL tasks complete) ────────────
        for file, result in zip(downloadable, results):
            t_file = time.monotonic() - t_start
            if isinstance(result, Exception):
                error_str = str(result)
                failed.append(FailedRecord(file=file, error=error_str,
                                           attempts=cfg.max_retries + 1))
                lock.mark_failed(file.path, error_str)
            else:
                bytes_written, retries = result
                downloaded.append(DownloadRecord(
                    file=file,
                    local_path=target_dir / file.path,
                    bytes_written=bytes_written,
                    elapsed=t_file,
                    retries=retries,
                    from_cache=False,
                ))

        elapsed = time.monotonic() - t_start
        total_dl = sum(r.bytes_written for r in downloaded)

        if failed:
            msg(
                f"Completed with {len(failed)} failed file(s). "
                f"Run again to retry.",
                emoji="⚠️",
            )
        else:
            msg(
                f"Finished {plan.dataset_id}. "
                f"{len(downloaded)} files, {total_dl / 1e6:.1f} MB in {elapsed:.1f}s.",
                emoji="✅",
            )

        try:
            registry = LocalRegistry(cfg)
            try:
                registry.register(
                    dataset_id=plan.dataset_id,
                    snapshot=plan.snapshot,
                    n_files=len(downloaded) + len(skipped),
                    n_failed=len(failed),
                    total_bytes=sum(
                        (target_dir / f.path).stat().st_size
                        for f in plan.files
                        if (target_dir / f.path).exists()
                    ),
                    data_dir=target_dir,
                )
            finally:
                registry.close()
        except Exception as exc:
            plan.warnings.append(
                f"Download finished, but local registry update failed: {exc}"
            )

        return DownloadResult(
            plan=plan,
            downloaded=downloaded,
            skipped=skipped,
            failed=failed,
            bytes_downloaded=total_dl,
            elapsed=elapsed,
        )

    async def _download_one(
        self,
        backend: HTTPBackend,
        file: FileRecord,
        target_dir: Path,
        lock: LockFile,
    ) -> tuple[int, int]:
        """Download a single file and update the lockfile on success."""
        bytes_written, retries = await backend.download_file(
            file,
            target_dir,
            verify_hash=self._cfg.verify_hash,
            verify_size=self._cfg.verify_size,
        )
        lock.mark_present(file.path, checksum=file.checksum)
        return bytes_written, retries
