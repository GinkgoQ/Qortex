"""CSV output adapter.

Writes model outputs as a CSV file with one row per prediction.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)

_HEADER = [
    "timestamp",
    "output_type",
    "class_name",
    "class_index",
    "confidence",
    "regression_value",
    "source_id",
    "model_id",
    "window_index",
    "extra_json",
]


class CSVOutputAdapter(OutputAdapter):
    """Output adapter that writes predictions to a CSV file.

    Parameters
    ----------
    path:
        Output CSV file path.
    append:
        If True, append to existing file instead of overwriting.
    pipeline_ref:
        Short pipeline reference.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._append = append
        self._pipeline_ref = pipeline_ref
        self._file = None
        self._writer = None
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        write_header = not self._append or not self._path.exists()
        self._file = self._path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_HEADER, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
        log.info("CSV output ready: %s (mode=%s)", self._path, mode)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("CSVOutputAdapter: call open() first")
        meta = metadata or {}
        import json

        confidence = max(output.probabilities.values()) if output.probabilities else None

        self._writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_type": output.output_type,
            "class_name": output.class_name,
            "class_index": output.class_index,
            "confidence": confidence,
            "regression_value": output.regression_value,
            "source_id": meta.get("source_id"),
            "model_id": meta.get("model_id"),
            "window_index": meta.get("window_index"),
            "extra_json": json.dumps(meta.get("extra", {})) if meta.get("extra") else "",
        })
        self._n_written += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        log.info("CSV output closed (%d rows written)", self._n_written)
