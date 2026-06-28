"""CSV output adapter.

Writes model outputs as an appendable, analytics-friendly CSV file with a
stable schema. Complex values are preserved as compact JSON strings so the
file remains compatible with spreadsheet tools, Polars, Pandas, and DuckDB.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qortex.core.exceptions import OutputAdapterError
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)

_HEADER = [
    "timestamp",
    "index",
    "pipeline",
    "output_type",
    "class",
    "class_index",
    "top_probability",
    "regression_value",
    "bbox_json",
    "probabilities_json",
    "metadata_json",
    "output_metadata_json",
    "raw_summary_json",
    "mask_summary_json",
    "embedding_summary_json",
    "window_index",
    "trigger_fired",
    "source",
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
        try:
            self._file = self._path.open(mode, newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=_HEADER, extrasaction="ignore")
            if write_header:
                self._writer.writeheader()
        except OSError as exc:
            raise OutputAdapterError(
                f"Cannot open CSV output: {exc}",
                output_type="csv",
                path=str(self._path),
            ) from exc
        log.info("CSV output ready: %s (mode=%s)", self._path, mode)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("CSVOutputAdapter: call open() first")
        meta = metadata or {}

        confidence = max(output.probabilities.values()) if output.probabilities else None

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "index": self._n_written,
            "pipeline": self._pipeline_ref,
            "output_type": output.output_type,
            "class": output.class_name,
            "class_index": output.class_index,
            "top_probability": confidence,
            "regression_value": output.regression_value,
            "bbox_json": _json_or_empty(output.bbox),
            "probabilities_json": _json_or_empty(output.probabilities),
            "metadata_json": _json_or_empty(meta),
            "output_metadata_json": _json_or_empty(output.metadata),
            "raw_summary_json": _json_or_empty(_summarize_payload(output.raw)),
            "mask_summary_json": _json_or_empty(_summarize_payload(output.mask)),
            "embedding_summary_json": _json_or_empty(_summarize_payload(output.embedding)),
            "window_index": meta.get("window_index"),
            "trigger_fired": meta.get("trigger_fired"),
            "source": meta.get("source"),
        }
        try:
            self._writer.writerow(row)
        except (csv.Error, OSError, TypeError) as exc:
            raise OutputAdapterError(
                f"Cannot write CSV prediction row: {exc}",
                output_type="csv",
                path=str(self._path),
            ) from exc
        self._n_written += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        log.info("CSV output closed (%d rows written)", self._n_written)


def _json_or_empty(value: Any) -> str:
    if value in (None, {}, [], ()):
        return ""
    return json.dumps(_to_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _summarize_payload(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    return repr(value)


def _summarize_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.array([])
            if finite.size:
                summary.update({
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                })
        return summary
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")) or None,
        }
    return {"type": type(value).__name__, "value": _to_jsonable(value)}
