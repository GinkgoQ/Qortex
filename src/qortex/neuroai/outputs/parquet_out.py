"""Parquet output adapter.

Batches model outputs in memory and writes them as a typed Parquet file when
``close()`` is called.  Parquet is the canonical offline artifact format —
typed, columnar, and directly readable by Polars, Pandas, DuckDB, and Spark.

Provenance columns are always included so the artifact can be traced back
to the source data, model, and pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class ParquetOutputAdapter(OutputAdapter):
    """Write model predictions to a Parquet file.

    Parameters
    ----------
    path:
        Output file path (e.g. ``predictions.parquet``).
    pipeline_ref:
        Optional pipeline identifier included in provenance column.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._pipeline_ref = pipeline_ref
        self._rows: list[dict[str, Any]] = []
        self._n_written = 0

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = []
        log.info("ParquetOutput: destination %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "index": self._n_written,
            "output_type": output.output_type,
            "class": output.class_name,
            "class_index": output.class_index,
            "top_probability": max(output.probabilities.values()) if output.probabilities else None,
            "regression_value": output.regression_value,
            "pipeline": self._pipeline_ref,
        }
        # Flatten probabilities as per-class columns
        for cls, prob in (output.probabilities or {}).items():
            row[f"prob_{cls}"] = prob

        if metadata:
            row.update(metadata)

        self._rows.append(row)
        self._n_written += 1

    def close(self) -> None:
        if not self._rows:
            log.warning("ParquetOutput: no rows to write — skipping %s", self._path)
            return
        try:
            import polars as pl
            df = pl.DataFrame(self._rows)
            df.write_parquet(str(self._path))
            log.info("ParquetOutput: wrote %d rows to %s", len(self._rows), self._path)
        except ImportError:
            try:
                import pandas as pd
                pd.DataFrame(self._rows).to_parquet(str(self._path), index=False)
                log.info("ParquetOutput (pandas): wrote %d rows to %s",
                         len(self._rows), self._path)
            except ImportError:
                raise ImportError(
                    "Parquet output requires polars or pandas. "
                    "Install with: pip install 'qortex[neuroai]'"
                )

    @property
    def n_written(self) -> int:
        return self._n_written
