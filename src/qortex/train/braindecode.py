"""Braindecode WindowsDataset adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BraindecodeAdapter:
    """Build a braindecode.datasets.BaseConcatDataset from Qortex shards.

    Each shard row becomes an MNE RawArray + Epochs → WindowsDataset.
    This requires mne and braindecode to be installed.
    """

    framework = "braindecode"

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        try:
            import mne
            import numpy as np
            import polars as pl
            from braindecode.datasets import BaseDataset, BaseConcatDataset
        except ImportError as e:
            raise ImportError(
                "Braindecode adapter requires mne and braindecode: "
                "pip install mne braindecode"
            ) from e

        shards = sorted(data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")

        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns:
            df = df.filter(pl.col("split") == split)

        mne.set_log_level("WARNING")
        base_datasets: list[BaseDataset] = []

        for row in df.iter_rows(named=True):
            raw_bytes = row.get("signal_bytes")
            if raw_bytes is None:
                continue

            dtype = row.get("signal_dtype", "float32")
            shape = row.get("signal_shape")
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(dtype))
            if shape:
                arr = arr.reshape(shape)  # (n_ch, n_times)

            sfreq = float(row.get("sfreq") or 256.0)
            n_ch = arr.shape[0] if arr.ndim > 1 else 1
            ch_names = [f"ch{i}" for i in range(n_ch)]
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(arr if arr.ndim == 2 else arr[np.newaxis, :], info)

            description = {
                "subject": row.get("subject"),
                "session": row.get("session"),
                "task": row.get("task"),
                "label": row.get("label"),
                "label_name": row.get("label_name"),
                "split": row.get("split"),
            }
            base_datasets.append(BaseDataset(raw, description=description))

        if not base_datasets:
            raise ValueError("No valid signal rows found in shards.")

        return BaseConcatDataset(base_datasets)
