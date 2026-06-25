"""PyTorch Dataset and IterableDataset adapters for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_shards(data_dir: Path, split: str | None) -> list[dict[str, Any]]:
    """Load all parquet shards in data_dir, filter by split if given."""
    import polars as pl

    shards = sorted(data_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")

    frames = [pl.read_parquet(s) for s in shards]
    df = pl.concat(frames)

    if split is not None and "split" in df.columns:
        df = df.filter(pl.col("split") == split)

    return df.to_dicts()


class QortexTorchDataset:
    """Map-style torch.utils.data.Dataset over Qortex Parquet shards.

    The entire dataset is loaded into memory on construction.  For large
    datasets use QortexIterableTorchDataset instead.
    """

    framework = "torch"

    def __init__(self, data_dir: Path, split: str | None = None) -> None:
        self._rows = _load_shards(data_dir, split)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        item: dict[str, Any] = {
            "subject": row.get("subject"),
            "session": row.get("session"),
            "task": row.get("task"),
            "run": row.get("run"),
            "modality": row.get("modality"),
            "label": row.get("label", -1),
            "label_name": row.get("label_name"),
            "sfreq": row.get("sfreq"),
        }
        raw = row.get("signal_bytes")
        if raw is not None:
            dtype = row.get("signal_dtype", "float32")
            shape = row.get("signal_shape")
            arr = np.frombuffer(raw, dtype=np.dtype(dtype))
            if shape:
                arr = arr.reshape(shape)
            try:
                import torch
                item["signal"] = torch.from_numpy(arr.copy())
            except ImportError:
                item["signal"] = arr
        else:
            item["signal"] = None
        return item

    def from_dir(self, data_dir: Path, split: str | None = None) -> "QortexTorchDataset":
        return QortexTorchDataset(data_dir, split)


class QortexIterableTorchDataset:
    """Iterable torch.utils.data.IterableDataset — streams shards lazily.

    Suitable for datasets that don't fit in memory.
    """

    framework = "torch"

    def __init__(self, data_dir: Path, split: str | None = None) -> None:
        self._data_dir = data_dir
        self._split = split
        self._shards = sorted(data_dir.glob("shard_*.parquet"))

    def __iter__(self):
        import polars as pl
        try:
            import torch
            _has_torch = True
        except ImportError:
            _has_torch = False

        for shard in self._shards:
            df = pl.read_parquet(shard)
            if self._split is not None and "split" in df.columns:
                df = df.filter(pl.col("split") == self._split)
            for row in df.iter_rows(named=True):
                item: dict[str, Any] = {
                    "subject": row.get("subject"),
                    "session": row.get("session"),
                    "task": row.get("task"),
                    "label": row.get("label", -1),
                    "label_name": row.get("label_name"),
                    "sfreq": row.get("sfreq"),
                }
                raw = row.get("signal_bytes")
                if raw is not None:
                    dtype = row.get("signal_dtype", "float32")
                    shape = row.get("signal_shape")
                    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
                    if shape:
                        arr = arr.reshape(shape)
                    if _has_torch:
                        import torch
                        item["signal"] = torch.from_numpy(arr.copy())
                    else:
                        item["signal"] = arr
                else:
                    item["signal"] = None
                yield item

    def from_dir(
        self, data_dir: Path, split: str | None = None
    ) -> "QortexIterableTorchDataset":
        return QortexIterableTorchDataset(data_dir, split)
