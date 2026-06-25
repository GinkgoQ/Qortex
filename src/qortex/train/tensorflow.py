"""TensorFlow tf.data.Dataset adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorFlowAdapter:
    """Build a tf.data.Dataset from Qortex Parquet shards."""

    framework = "tensorflow"

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow adapter requires tensorflow: pip install tensorflow")

        import numpy as np
        import polars as pl

        shards = sorted(data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")

        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns:
            df = df.filter(pl.col("split") == split)

        signals: list[np.ndarray] = []
        labels: list[int] = []

        for row in df.iter_rows(named=True):
            raw = row.get("signal_bytes")
            if raw is not None:
                dtype = row.get("signal_dtype", "float32")
                shape = row.get("signal_shape")
                arr = np.frombuffer(raw, dtype=np.dtype(dtype))
                if shape:
                    arr = arr.reshape(shape)
                signals.append(arr.astype(np.float32))
            labels.append(int(row.get("label") or -1))

        if not signals:
            raise ValueError("No signal data found in shards.")

        shapes = [s.shape for s in signals]
        if len(set(shapes)) != 1:
            raise ValueError(
                "Signals have inconsistent shapes — cannot build a batched tf.data.Dataset. "
                "Use fixed windows first."
            )

        X = np.stack(signals, axis=0)
        y = np.array(labels, dtype=np.int64)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        return ds
