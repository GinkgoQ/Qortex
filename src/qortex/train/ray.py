"""Ray Data adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RayAdapter:
    """Build a ray.data.Dataset from Qortex Parquet shards."""

    framework = "ray"

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        try:
            import ray.data as rd
        except ImportError:
            raise ImportError("Ray adapter requires ray[data]: pip install 'ray[data]'")

        shards = sorted(str(s) for s in data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found in {data_dir}")

        ds = rd.read_parquet(shards)

        if split:
            ds = ds.filter(lambda row: row.get("split") == split)

        return ds
