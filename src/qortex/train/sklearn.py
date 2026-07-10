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

        Only rows with signal_bytes are included.  Rows that carry only table
        data (behavior, events) are skipped so that len(X) == len(y) is always
        guaranteed.  Use as_dataframe() for table-only artifacts.
        """
        import polars as pl

        shards = _parquet_shards(data_dir, split=split)
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")

        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns and not (data_dir / split).exists():
            df = df.filter(pl.col("split") == split)

        Xs: list[np.ndarray] = []
        ys: list[int] = []
        n_skipped = 0

        for row in df.iter_rows(named=True):
            raw = row.get("signal_bytes")
            if raw is None:
                # Table-only row — skip to preserve X/y alignment.
                n_skipped += 1
                continue
            dtype = row.get("signal_dtype", "float32")
            shape = row.get("signal_shape")
            arr = np.frombuffer(raw, dtype=np.dtype(dtype))
            if shape:
                arr = arr.reshape(shape)
            Xs.append(arr.flatten())
            ys.append(int(row.get("label") or -1))

        if not Xs:
            raise ValueError(
                "No signal data found in shards. "
                "For event/table artifacts use as_dataframe() instead of from_dir(). "
                f"({n_skipped} non-signal row(s) were skipped.)"
            )

        X = np.stack(Xs, axis=0)
        y = np.array(ys, dtype=np.int64)
        assert len(X) == len(y), "Internal error: X/y length mismatch after alignment"
        return X, y

    def as_dataframe(
        self, data_dir: Path, split: str | None = None
    ):
        """Return a Polars DataFrame without deserialising signal bytes.

        Suitable for table/event/behavior artifacts and for metadata inspection
        of signal artifacts without loading arrays into memory.
        """
        import polars as pl
        shards = _parquet_shards(data_dir, split=split)
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")
        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns and not (data_dir / split).exists():
            df = df.filter(pl.col("split") == split)
        return df


def _parquet_shards(data_dir: Path, split: str | None = None) -> list[Path]:
    if split:
        split_dir = data_dir / split
        if split_dir.exists():
            shards = sorted(split_dir.glob("*.parquet"))
            if shards:
                return shards
    return sorted(data_dir.glob("shard_*.parquet")) or sorted(data_dir.glob("**/*.parquet"))
