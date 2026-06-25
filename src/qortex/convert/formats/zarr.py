"""Write SampleRecords to a Zarr store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import SampleRecord


class ZarrWriter:
    format_name = "zarr"
    file_extension = ".zarr"

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        try:
            import numpy as np
            import zarr
            import polars as pl
        except ImportError:
            raise ImportError("zarr output requires zarr, numpy: pip install zarr numpy")

        output_dir.mkdir(parents=True, exist_ok=True)
        store_path = output_dir / "data.zarr"
        store = zarr.open(str(store_path), mode="w")

        all_samples = list(samples)
        signals = [np.asarray(s.data) for s in all_samples if s.data is not None]

        if signals:
            shapes = [sig.shape for sig in signals]
            if len(set(shapes)) == 1:
                arr = np.stack(signals, axis=0)
                store.create_dataset(
                    "signals",
                    data=arr,
                    chunks=(min(shard_size, len(arr)), *arr.shape[1:]),
                    overwrite=True,
                )
            else:
                grp = store.require_group("signals_ragged")
                for i, sig in enumerate(signals):
                    grp.create_dataset(str(i), data=sig, overwrite=True)

        meta_rows = [
            {
                "subject": s.subject or "",
                "session": s.session or "",
                "task": s.task or "",
                "run": s.run or "",
                "modality": s.modality or "",
                "label": s.label if s.label is not None else -1,
                "label_name": s.label_name or "",
                "sfreq": s.sfreq or 0.0,
                "onset": s.onset or 0.0,
                "duration": s.duration or 0.0,
                "split": s.split or "",
            }
            for s in all_samples
        ]
        pl.DataFrame(meta_rows).write_parquet(output_dir / "metadata.parquet")

        if metadata:
            store.attrs.update(metadata)

        return store_path

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 0.6)
