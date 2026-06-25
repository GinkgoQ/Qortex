"""Cache directory layout conventions.

All paths under the Qortex cache are managed here — nothing else should
construct cache paths from scratch.

Layout:
    {cache_dir}/
        catalog.duckdb              ← searchable dataset catalog
        registry.duckdb             ← local snapshot/download registry
        objects/                    ← content-addressed file store
            {xx}/{yyyyyy...}        ← two-level sharding
        datasets/
            {dataset_id}/
                {snapshot_tag}/
                    manifest.parquet
                    manifest.json
                    download.lock.yaml (under .qortex/)
                    data/           ← BIDS files
                    exports/        ← converted outputs
                    reports/        ← validation + EDA reports
"""

from __future__ import annotations

from pathlib import Path

from qortex.core.config import QortexConfig, get_config


class LakeLayout:
    """Central authority for cache path resolution."""

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()
        self._root = self._cfg.cache_dir

    @property
    def root(self) -> Path:
        return self._root

    @property
    def catalog_db(self) -> Path:
        return self._root / "catalog.duckdb"

    @property
    def registry_db(self) -> Path:
        return self._root / "registry.duckdb"

    @property
    def objects_dir(self) -> Path:
        return self._root / "objects"

    def dataset_root(self, dataset_id: str) -> Path:
        return self._root / "datasets" / dataset_id

    def snapshot_root(self, dataset_id: str, snapshot: str) -> Path:
        return self.dataset_root(dataset_id) / snapshot

    def manifest_dir(self, dataset_id: str, snapshot: str) -> Path:
        return self.snapshot_root(dataset_id, snapshot)

    def manifest_parquet(self, dataset_id: str, snapshot: str) -> Path:
        return self.manifest_dir(dataset_id, snapshot) / "manifest.parquet"

    def manifest_json(self, dataset_id: str, snapshot: str) -> Path:
        return self.manifest_dir(dataset_id, snapshot) / "manifest.json"

    def data_dir(self, dataset_id: str, snapshot: str) -> Path:
        return self.snapshot_root(dataset_id, snapshot) / "data"

    def exports_dir(self, dataset_id: str, snapshot: str) -> Path:
        return self.snapshot_root(dataset_id, snapshot) / "exports"

    def reports_dir(self, dataset_id: str, snapshot: str) -> Path:
        return self.snapshot_root(dataset_id, snapshot) / "reports"

    def lock_file(self, dataset_id: str, snapshot: str) -> Path:
        return self.data_dir(dataset_id, snapshot) / ".qortex" / "download.lock.yaml"

    def ensure_snapshot(self, dataset_id: str, snapshot: str) -> None:
        """Create all required directories for a snapshot."""
        for d in (
            self.manifest_dir(dataset_id, snapshot),
            self.data_dir(dataset_id, snapshot),
            self.exports_dir(dataset_id, snapshot),
            self.reports_dir(dataset_id, snapshot),
            self.objects_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def has_manifest(self, dataset_id: str, snapshot: str) -> bool:
        return (
            self.manifest_parquet(dataset_id, snapshot).exists()
            and self.manifest_json(dataset_id, snapshot).exists()
        )

    def has_data(self, dataset_id: str, snapshot: str) -> bool:
        d = self.data_dir(dataset_id, snapshot)
        return d.exists() and any(d.iterdir())

    def list_datasets(self) -> list[str]:
        datasets_dir = self._root / "datasets"
        if not datasets_dir.exists():
            return []
        return [p.name for p in datasets_dir.iterdir() if p.is_dir()]

    def list_snapshots(self, dataset_id: str) -> list[str]:
        ds_root = self.dataset_root(dataset_id)
        if not ds_root.exists():
            return []
        return [p.name for p in ds_root.iterdir() if p.is_dir()]
