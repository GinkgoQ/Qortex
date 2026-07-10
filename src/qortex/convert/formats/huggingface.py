"""Write SampleRecords as a HuggingFace datasets.Dataset on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import SampleRecord
from qortex.convert.formats.parquet import _numeric_array_or_none


class HuggingFaceWriter:
    format_name = "huggingface"
    file_extension = ""  # directory-based

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        try:
            import datasets
            import numpy as np
        except ImportError:
            raise ImportError(
                "HuggingFace output requires `datasets`: pip install datasets"
            )

        all_samples = list(samples)
        rows: dict[str, list] = {
            "subject": [],
            "session": [],
            "task": [],
            "run": [],
            "modality": [],
            "label": [],
            "label_name": [],
            "sfreq": [],
            "onset": [],
            "duration": [],
            "split": [],
            "signal": [],
            "data_json": [],
        }

        for s in all_samples:
            rows["subject"].append(s.subject or "")
            rows["session"].append(s.session or "")
            rows["task"].append(s.task or "")
            rows["run"].append(s.run or "")
            rows["modality"].append(s.modality or "")
            rows["label"].append(s.label if s.label is not None else -1)
            rows["label_name"].append(s.label_name or "")
            rows["sfreq"].append(s.sfreq or 0.0)
            rows["onset"].append(s.onset or 0.0)
            rows["duration"].append(s.duration or 0.0)
            rows["split"].append(s.split or "")
            arr = _numeric_array_or_none(s.data, np)
            if arr is not None:
                rows["signal"].append(arr.astype(np.float32).tolist())
                rows["data_json"].append("")
            elif s.data is not None:
                rows["signal"].append(None)
                rows["data_json"].append(json.dumps(s.data, default=str, sort_keys=True))
            else:
                rows["signal"].append(None)
                rows["data_json"].append("")

        ds = datasets.Dataset.from_dict(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))

        if metadata:
            (output_dir / "qortex_metadata.json").write_text(
                json.dumps(metadata, default=str, indent=2)
            )
        return output_dir

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        n = 1
        for d in sample_shape:
            n *= d
        return int(n_samples * n * 4 * 0.55)
