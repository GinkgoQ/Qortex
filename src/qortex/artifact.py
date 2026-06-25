"""User-facing handle for converted Qortex artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qortex.core.entities import ArtifactManifest


class Artifact:
    """Open and use a converted Qortex artifact."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        manifest_path = self.path / "artifact_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No artifact_manifest.json found in {self.path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = ArtifactManifest(**data)

    @classmethod
    def open(cls, path: Path | str) -> "Artifact":
        return cls(path)

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.manifest.artifact_id,
            "dataset_id": self.manifest.dataset_id,
            "snapshot": self.manifest.snapshot,
            "format": self.manifest.output_format,
            "n_samples": self.manifest.n_samples,
            "n_subjects": self.manifest.n_subjects,
            "splits": self.manifest.splits,
        }

    def torch(self, split: str | None = "train", *, iterable: bool = False):
        from qortex.train.torch import QortexIterableTorchDataset, QortexTorchDataset

        if self.manifest.output_format != "parquet":
            raise ValueError("Torch adapter currently expects a Parquet Qortex artifact.")
        if iterable:
            return QortexIterableTorchDataset(self.path, split=split)
        return QortexTorchDataset(self.path, split=split)

    def sklearn(self, split: str | None = None):
        from qortex.train.sklearn import SklearnAdapter

        if self.manifest.output_format != "parquet":
            raise ValueError("sklearn adapter currently expects a Parquet Qortex artifact.")
        return SklearnAdapter().from_dir(self.path, split=split)
