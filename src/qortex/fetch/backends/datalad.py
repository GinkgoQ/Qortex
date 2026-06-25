"""DataLad-based download backend (optional).

Requires DataLad to be installed and the dataset to be a git-annex repository.
Falls back with a clear error message if DataLad is not available.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from qortex.core.entities import FileRecord
from qortex.core.exceptions import DownloadError


class DataLadBackend:
    """Use DataLad ``datalad get`` for partial dataset retrieval.

    This backend is particularly useful for very large datasets where
    only a subset of files is needed, as DataLad avoids downloading
    unrequested annexed content.
    """

    backend_id = "datalad"

    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path
        self._check_datalad()

    @staticmethod
    def _check_datalad() -> None:
        try:
            subprocess.run(
                ["datalad", "--version"],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise ImportError(
                "DataLad backend requires DataLad to be installed. "
                "Install it with: pip install datalad"
            ) from exc

    async def download_file(
        self,
        file: FileRecord,
        target_dir: Path,
        *,
        resume: bool = True,
        verify_hash: bool = True,
        verify_size: bool = True,
    ) -> tuple[int, int]:
        relative_path = file.path

        # datalad get is synchronous; run in a thread to avoid blocking the loop
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["datalad", "get", relative_path],
                    cwd=self._dataset_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            )
        except subprocess.CalledProcessError as exc:
            raise DownloadError(
                file.path, "", f"datalad get failed: {exc.stderr.strip()}"
            ) from exc

        local_file = self._dataset_path / relative_path
        bytes_written = local_file.stat().st_size if local_file.exists() else 0
        return bytes_written, 0
