"""Write SampleRecords as TFRecord files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import SampleRecord
from qortex.convert.formats.parquet import _numeric_array_or_none


class TFRecordWriter:
    format_name = "tfrecord"
    file_extension = ".tfrecord"

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        try:
            import tensorflow as tf
            import numpy as np
        except ImportError:
            raise ImportError(
                "TFRecord output requires TensorFlow: pip install tensorflow"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        total = 0
        buffer: list[SampleRecord] = []
        writer = self._open_shard(output_dir, shard_idx, tf)

        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= shard_size:
                for s in buffer:
                    writer.write(self._make_example(s, tf, np).SerializeToString())
                writer.close()
                total += len(buffer)
                buffer = []
                shard_idx += 1
                writer = self._open_shard(output_dir, shard_idx, tf)

        for s in buffer:
            writer.write(self._make_example(s, tf, np).SerializeToString())
        total += len(buffer)
        writer.close()

        if metadata:
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, default=str, indent=2)
            )
        (output_dir / "_index.json").write_text(
            json.dumps({"n_shards": shard_idx + 1, "n_samples": total})
        )
        return output_dir

    @staticmethod
    def _open_shard(output_dir: Path, idx: int, tf: Any) -> Any:
        return tf.io.TFRecordWriter(str(output_dir / f"shard_{idx:05d}.tfrecord"))

    @staticmethod
    def _make_example(sample: SampleRecord, tf: Any, np: Any) -> Any:
        def _bytes(v: str) -> Any:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode()]))

        def _int(v: int) -> Any:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

        def _float(v: float) -> Any:
            return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))

        feat: dict[str, Any] = {
            "subject": _bytes(sample.subject or ""),
            "session": _bytes(sample.session or ""),
            "task": _bytes(sample.task or ""),
            "run": _bytes(sample.run or ""),
            "modality": _bytes(sample.modality or ""),
            "label": _int(sample.label if sample.label is not None else -1),
            "label_name": _bytes(sample.label_name or ""),
            "sfreq": _float(sample.sfreq or 0.0),
            "onset": _float(sample.onset or 0.0),
            "duration": _float(sample.duration or 0.0),
            "split": _bytes(sample.split or ""),
        }

        arr = _numeric_array_or_none(sample.data, np)
        if arr is not None:
            arr = arr.astype(np.float32)
            feat["signal_bytes"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[arr.tobytes()])
            )
            feat["signal_shape"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(arr.shape))
            )
        elif sample.data is not None:
            feat["data_json"] = _bytes(json.dumps(sample.data, default=str, sort_keys=True))

        return tf.train.Example(features=tf.train.Features(feature=feat))

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 1.1)
