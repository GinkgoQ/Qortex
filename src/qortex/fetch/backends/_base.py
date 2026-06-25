"""Base protocol for download backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from qortex.core.entities import FileRecord


@runtime_checkable
class DownloadBackend(Protocol):
    """Protocol every download backend must implement."""

    backend_id: str

    async def download_file(
        self,
        file: FileRecord,
        target_dir: Path,
        *,
        resume: bool = True,
        verify_hash: bool = True,
        verify_size: bool = True,
    ) -> tuple[int, int]:
        """Download *file* into *target_dir*.

        Returns
        -------
        (bytes_written, retries)
        """
        ...
