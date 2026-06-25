"""Scikit-learn / NumPy adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SklearnAdapter:
    """Return (X, y) numpy arrays from Qortex Parquet shards."""

    framework = "sklearn"

    def from_dir(
        self, data_dir: Path, split: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays suitable for sklearn estimators.

        X shape: (n_samples, n_channels * n_timepoints) — flattened.
        y shape: (n_samples,) — integer class labels.
        """
        import polars as pl

        shards = sorted(data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")

        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns:
            df = df.filter(pl.col("split") == split)

        Xs: list[np.ndarray] = []
        ys: list[int] = []

        for row in df.iter_rows(named=True):
            raw = row.get("signal_bytes")
            if raw is not None:
                dtype = row.get("signal_dtype", "float32")
                shape = row.get("signal_shape")
                arr = np.frombuffer(raw, dtype=np.dtype(dtype))
                if shape:
                    arr = arr.reshape(shape)
                Xs.append(arr.flatten())
            ys.append(int(row.get("label") or -1))

        if not Xs:
            raise ValueError("No signal data found in shards.")

        X = np.stack(Xs, axis=0)
        y = np.array(ys, dtype=np.int64)
        return X, y

    def as_dataframe(
        self, data_dir: Path, split: str | None = None
    ):
        """Return a Polars DataFrame without deserialising signal bytes."""
        import polars as pl
        shards = sorted(data_dir.glob("shard_*.parquet"))
        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns:
            df = df.filter(pl.col("split") == split)
        return df
