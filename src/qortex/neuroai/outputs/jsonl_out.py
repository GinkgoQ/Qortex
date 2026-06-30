"""JSONL (newline-delimited JSON) output adapter.

Every prediction is written as one JSON object per line, including:
  - timestamp
  - window index
  - predicted class + probabilities
  - full provenance reference
  - any trigger events fired

JSONL is the canonical streaming output format — it is appendable,
grep-able, and directly consumable by Pandas, Polars, or jq.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class JSONLOutputAdapter(OutputAdapter):
    """Write model predictions to a JSONL file.

    Parameters
    ----------
    path:
        Output file path.  Parent directories are created automatically.
    append:
        When True, append to an existing file instead of overwriting.
    pipeline_ref:
        Optional pipeline name/hash included in every record for provenance.
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
        self._n_written = 0
        self._n_marker_records = 0

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        self._file = open(self._path, mode, encoding="utf-8")
        log.info("JSONLOutput: opened %s (append=%s)", self._path, self._append)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._file is None:
            raise RuntimeError("Call open() before write().")

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "index": self._n_written,
            "output_type": output.output_type,
        }

        if output.class_name is not None:
            record["class"] = output.class_name
        if output.class_index is not None:
            record["class_index"] = output.class_index
        if output.probabilities:
            record["probabilities"] = {k: round(v, 6) for k, v in output.probabilities.items()}
        if output.regression_value is not None:
            record["value"] = output.regression_value
        if output.bbox is not None:
            record["bbox"] = output.bbox
        if output.metadata:
            record["meta"] = output.metadata
        if metadata:
            record.update(metadata)
        if self._pipeline_ref:
            record["pipeline"] = self._pipeline_ref

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._n_written += 1

    def write_marker(self, marker: "Any") -> None:
        """Write a structured EventMarkerOutput as a dedicated JSONL marker record.

        Called by the runtime when a trigger fires.  The record has
        ``"record_type": "event_marker"`` so it can be filtered separately
        from normal prediction records.
        """
        if self._file is None:
            return
        from datetime import datetime, timezone
        record: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_type": "event_marker",
            "event_type": getattr(marker, "event_type", "trigger"),
            "label": getattr(marker, "label", None),
            "confidence": getattr(marker, "confidence", None),
            "window_index": getattr(marker, "window_index", None),
            "source_id": getattr(marker, "source_id", None),
            "timestamp_utc": getattr(marker, "timestamp_utc", None),
            "emit_payload": getattr(marker, "emit_payload", {}),
        }
        if self._pipeline_ref:
            record["pipeline"] = self._pipeline_ref
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._n_marker_records += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            log.info("JSONLOutput: closed — %d records written to %s",
                     self._n_written, self._path)

    @property
    def n_written(self) -> int:
        return self._n_written

    @property
    def n_marker_records(self) -> int:
        return self._n_marker_records
