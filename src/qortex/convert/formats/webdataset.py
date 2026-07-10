"""Write SampleRecords as WebDataset tar shards."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import SampleRecord
from qortex.convert.formats.parquet import _numeric_array_or_none


class WebDatasetWriter:
    format_name = "webdataset"
    file_extension = ".tar"

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        import numpy as np
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        shards_written = 0
        buffer: list[SampleRecord] = []
        total = 0

        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= shard_size:
                self._write_shard(buffer, output_dir, shard_idx, np)
                shard_idx += 1
                shards_written += 1
                total += len(buffer)
                buffer = []

        if buffer:
            self._write_shard(buffer, output_dir, shard_idx, np)
            shards_written += 1
            total += len(buffer)

        if metadata:
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, default=str, indent=2)
            )
        (output_dir / "_index.json").write_text(
            json.dumps({"n_shards": shards_written, "n_samples": total})
        )
        return output_dir

    def _write_shard(
        self,
        buffer: list[SampleRecord],
        output_dir: Path,
        idx: int,
        np: Any,
    ) -> None:
        shard_path = output_dir / f"shard_{idx:05d}.tar"
        with tarfile.open(shard_path, "w") as tf:
            for i, sample in enumerate(buffer):
                key = f"{idx:05d}_{i:06d}"
                meta = {
                    "subject": sample.subject,
                    "session": sample.session,
                    "task": sample.task,
                    "run": sample.run,
                    "modality": sample.modality,
                    "label": sample.label,
                    "label_name": sample.label_name,
                    "sfreq": sample.sfreq,
                    "onset": sample.onset,
                    "duration": sample.duration,
                    "split": sample.split,
                }
                arr = _numeric_array_or_none(sample.data, np)
                if arr is None and sample.data is not None:
                    meta["data_json"] = sample.data
                meta_bytes = json.dumps(meta, default=str).encode()
                self._add_bytes(tf, f"{key}.json", meta_bytes)

                if arr is not None:
                    buf = io.BytesIO()
                    np.save(buf, arr.astype(np.float32))
                    self._add_bytes(tf, f"{key}.npy", buf.getvalue())

    @staticmethod
    def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 1.05)
