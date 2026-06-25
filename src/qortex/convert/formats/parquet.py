"""Write SampleRecords as sharded Parquet files."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

from qortex.core.entities import SampleRecord


class ParquetWriter:
    format_name = "parquet"
    file_extension = ".parquet"

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        written = 0
        buffer: list[SampleRecord] = []

        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= shard_size:
                self._flush(buffer, output_dir, shard_idx)
                shard_idx += 1
                written += len(buffer)
                buffer = []

        if buffer:
            self._flush(buffer, output_dir, shard_idx)
            written += len(buffer)

        if metadata:
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, default=str, indent=2)
            )
        (output_dir / "_SUCCESS").write_text(
            f"shards={shard_idx + (1 if written else 0)}\nsamples={written}\n"
        )
        return output_dir

    def _flush(self, buffer: list[SampleRecord], output_dir: Path, idx: int) -> None:
        import numpy as np

        rows = []
        for s in buffer:
            row: dict[str, Any] = {
                "subject": s.subject,
                "session": s.session,
                "task": s.task,
                "run": s.run,
                "modality": s.modality,
                "label": s.label,
                "label_name": s.label_name,
                "sfreq": s.sfreq,
                "onset": s.onset,
                "duration": s.duration,
                "split": s.split,
                "source_path": s.provenance.get("source_path"),
            }
            arr = _numeric_array_or_none(s.data, np)
            if arr is not None:
                row["signal_bytes"] = arr.astype(np.float32).tobytes()
                row["signal_dtype"] = str(arr.dtype)
                row["signal_shape"] = list(arr.shape)
            elif s.data is not None:
                row["data_json"] = json.dumps(s.data, default=str, sort_keys=True)
            rows.append(row)

        out_path = output_dir / f"shard_{idx:05d}.parquet"
        try:
            import polars as pl
        except ImportError:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(rows), out_path)
        else:
            pl.DataFrame(rows).write_parquet(out_path)

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 0.5)


def _numeric_array_or_none(data: Any, np: Any) -> Any | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        return None
    if isinstance(data, (str, bytes, bytearray)):
        return None
    if isinstance(data, Sequence) and not data:
        return None
    arr = np.asarray(data)
    if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
        return None
    return arr
