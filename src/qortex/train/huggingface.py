"""HuggingFace Datasets adapter for Qortex artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class HuggingFaceAdapter:
    """Load a Qortex HuggingFace artifact or build one from Parquet shards."""

    framework = "huggingface"

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        try:
            import datasets
        except ImportError:
            raise ImportError(
                "HuggingFace adapter requires datasets: pip install datasets"
            )

        # Try native HF artifact first
        if (data_dir / "dataset_info.json").exists():
            ds = datasets.load_from_disk(str(data_dir))
            if split and hasattr(ds, "filter"):
                ds = ds.filter(lambda x: x.get("split") == split)
            return ds

        # Fall back to building from Parquet shards
        import polars as pl
        shards = sorted(data_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No artifact found in {data_dir}")

        frames = [pl.read_parquet(s) for s in shards]
        df = pl.concat(frames)
        if split and "split" in df.columns:
            df = df.filter(pl.col("split") == split)

        rows = df.to_dicts()
        return datasets.Dataset.from_list(rows)
