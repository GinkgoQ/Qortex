"""Download planner — converts a Manifest + SelectionSpec into a DownloadPlan."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from qortex.core.entities import DownloadPlan, Manifest, SelectionSpec
from qortex.core.exceptions import StorageError
from qortex.plan.selector import Selector


class DownloadPlanner:
    """Resolve selection logic and produce a ``DownloadPlan``."""

    def __init__(self, check_disk_space: bool = True) -> None:
        self._check_disk = check_disk_space
        self._selector = Selector()

    def plan(
        self,
        manifest: Manifest,
        spec: SelectionSpec,
        target_dir: Path,
    ) -> DownloadPlan:
        files, essential, warnings, reasons, recordings = self._selector.resolve_with_reasons(
            manifest, spec
        )

        estimated_bytes = sum(f.size or 0 for f in files)

        if self._check_disk:
            disk = shutil.disk_usage(target_dir.parent if not target_dir.exists() else target_dir)
            if estimated_bytes > 0 and disk.free < estimated_bytes * 1.05:
                raise StorageError(
                    f"Insufficient disk space: need ~{estimated_bytes / 1e9:.2f} GB, "
                    f"have {disk.free / 1e9:.2f} GB free at {target_dir}."
                )

        # Warn if no modality-specific files were found
        non_essential = [f for f in files if not f.is_essential]
        if not non_essential:
            warnings.append(
                "The selected files contain only essential metadata files. "
                "Check your subject/modality/include filters."
            )

        return DownloadPlan(
            dataset_id=manifest.dataset_id,
            snapshot=manifest.snapshot,
            target_dir=target_dir,
            selection=spec,
            files=files,
            essential_files=essential,
            estimated_bytes=estimated_bytes,
            warnings=warnings,
            selection_reasons=reasons,
            recordings=recordings,
            created_at=datetime.now(timezone.utc),
        )
