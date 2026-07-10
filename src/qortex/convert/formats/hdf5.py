"""Write SampleRecords to an HDF5 file via h5py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import SampleRecord
from qortex.convert.formats.parquet import _numeric_array_or_none


class HDF5Writer:
    format_name = "hdf5"
    file_extension = ".h5"

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        try:
            import h5py
            import numpy as np
        except ImportError:
            raise ImportError("HDF5 output requires h5py: pip install h5py")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "data.h5"
        all_samples = list(samples)
        n = len(all_samples)

        with h5py.File(out_path, "w") as f:
            if metadata:
                for k, v in metadata.items():
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        f.attrs[k] = str(v)

            meta_dt = np.dtype([
                ("subject", h5py.string_dtype()),
                ("session", h5py.string_dtype()),
                ("task", h5py.string_dtype()),
                ("run", h5py.string_dtype()),
                ("modality", h5py.string_dtype()),
                ("label", np.int64),
                ("label_name", h5py.string_dtype()),
                ("sfreq", np.float64),
                ("onset", np.float64),
                ("duration", np.float64),
                ("split", h5py.string_dtype()),
                ("data_json", h5py.string_dtype()),
                ("signal_index", np.int64),
            ])
            signal_items: list[tuple[int, Any]] = []
            data_json_values: list[str] = []
            signal_indices: list[int] = []
            for s in all_samples:
                arr = _numeric_array_or_none(s.data, np)
                if arr is not None:
                    signal_indices.append(len(signal_items))
                    signal_items.append((len(data_json_values), arr.astype(np.float32)))
                    data_json_values.append("")
                elif s.data is not None:
                    signal_indices.append(-1)
                    import json
                    data_json_values.append(json.dumps(s.data, default=str, sort_keys=True))
                else:
                    signal_indices.append(-1)
                    data_json_values.append("")

            meta_arr = np.array(
                [
                    (
                        s.subject or "",
                        s.session or "",
                        s.task or "",
                        s.run or "",
                        s.modality or "",
                        s.label if s.label is not None else -1,
                        s.label_name or "",
                        s.sfreq or 0.0,
                        s.onset or 0.0,
                        s.duration or 0.0,
                        s.split or "",
                        data_json_values[i],
                        signal_indices[i],
                    )
                    for i, s in enumerate(all_samples)
                ],
                dtype=meta_dt,
            )
            f.create_dataset("metadata", data=meta_arr)

            signals = [sig for _row_idx, sig in signal_items]
            if signals:
                shapes = [sig.shape for sig in signals]
                if len(set(shapes)) == 1:
                    arr = np.stack(signals, axis=0)
                    f.create_dataset(
                        "signals",
                        data=arr,
                        chunks=(min(shard_size, len(arr)), *arr.shape[1:]),
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    grp = f.create_group("signals_ragged")
                    for i, sig in enumerate(signals):
                        grp.create_dataset(str(i), data=sig)

        return out_path

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 0.4)
