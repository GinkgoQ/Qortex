"""Dask adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DaskAdapter:
    """Build a dask.dataframe from Qortex Parquet shards."""

    framework = "dask"

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        try:
            import dask.dataframe as dd
        except ImportError:
            raise ImportError("Dask adapter requires dask: pip install dask[dataframe]")

        pattern = str(data_dir / "shard_*.parquet")
        ddf = dd.read_parquet(pattern, engine="pyarrow")

        if split:
            ddf = ddf[ddf["split"] == split]

        return ddf
