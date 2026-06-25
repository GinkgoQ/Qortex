"""Local dataset/snapshot registry.

The registry tracks which datasets and snapshots have been downloaded,
when, how many files, and total size on disk.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.core.config import QortexConfig, get_config
from qortex.lake.layout import LakeLayout


class SnapshotEntry:
    __slots__ = ("dataset_id", "snapshot", "doi", "downloaded_at",
                 "n_files", "n_failed", "total_bytes", "data_dir")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return (
            f"SnapshotEntry(dataset_id={self.dataset_id!r}, "
            f"snapshot={self.snapshot!r}, "
            f"n_files={self.n_files}, "
            f"total_bytes={self.total_bytes})"
        )


class LocalRegistry:
    """Registry of downloaded datasets, backed by DuckDB or SQLite fallback."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS snapshots (
        dataset_id   VARCHAR NOT NULL,
        snapshot     VARCHAR NOT NULL,
        doi          VARCHAR,
        downloaded_at TIMESTAMPTZ,
        n_files      INTEGER DEFAULT 0,
        n_failed     INTEGER DEFAULT 0,
        total_bytes  BIGINT  DEFAULT 0,
        data_dir     VARCHAR,
        PRIMARY KEY (dataset_id, snapshot)
    );
    """

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()
        self._layout = LakeLayout(config)
        self._db_path = self._layout.registry_db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect(self._db_path)
        self._con.execute(self._SCHEMA)

    # ── Write ──────────────────────────────────────────────────────────────

    def register(
        self,
        *,
        dataset_id: str,
        snapshot: str,
        doi: str | None = None,
        n_files: int = 0,
        n_failed: int = 0,
        total_bytes: int = 0,
        data_dir: Path | None = None,
    ) -> None:
        """Upsert a snapshot record."""
        self._con.execute(
            """
            INSERT INTO snapshots
                (dataset_id, snapshot, doi, downloaded_at, n_files, n_failed, total_bytes, data_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (dataset_id, snapshot) DO UPDATE SET
                doi          = EXCLUDED.doi,
                downloaded_at = EXCLUDED.downloaded_at,
                n_files      = EXCLUDED.n_files,
                n_failed     = EXCLUDED.n_failed,
                total_bytes  = EXCLUDED.total_bytes,
                data_dir     = EXCLUDED.data_dir
            """,
            [
                dataset_id, snapshot, doi,
                datetime.now(timezone.utc).isoformat(),
                n_files, n_failed, total_bytes,
                str(data_dir) if data_dir else None,
            ],
        )
        _commit(self._con)

    def remove(self, dataset_id: str, snapshot: str) -> None:
        self._con.execute(
            "DELETE FROM snapshots WHERE dataset_id = ? AND snapshot = ?",
            [dataset_id, snapshot],
        )
        _commit(self._con)

    # ── Read ───────────────────────────────────────────────────────────────

    def get(self, dataset_id: str, snapshot: str) -> SnapshotEntry | None:
        rows = self._con.execute(
            "SELECT * FROM snapshots WHERE dataset_id = ? AND snapshot = ?",
            [dataset_id, snapshot],
        ).fetchall()
        if not rows:
            return None
        return self._row_to_entry(rows[0])

    def list_all(self) -> list[SnapshotEntry]:
        rows = self._con.execute(
            "SELECT * FROM snapshots ORDER BY downloaded_at DESC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def list_dataset(self, dataset_id: str) -> list[SnapshotEntry]:
        rows = self._con.execute(
            "SELECT * FROM snapshots WHERE dataset_id = ? ORDER BY snapshot",
            [dataset_id],
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def is_registered(self, dataset_id: str, snapshot: str) -> bool:
        return self.get(dataset_id, snapshot) is not None

    def total_disk_usage(self) -> int:
        result = self._con.execute(
            "SELECT COALESCE(SUM(total_bytes), 0) FROM snapshots"
        ).fetchone()
        return int(result[0]) if result else 0

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_entry(row: tuple) -> SnapshotEntry:
        cols = ["dataset_id", "snapshot", "doi", "downloaded_at",
                "n_files", "n_failed", "total_bytes", "data_dir"]
        return SnapshotEntry(**dict(zip(cols, row)))

    def close(self) -> None:
        self._con.close()

    def __del__(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass


def _connect(path: Path) -> Any:
    try:
        import duckdb
    except ImportError:
        return sqlite3.connect(str(path))
    return duckdb.connect(str(path))


def _commit(connection: Any) -> None:
    commit = getattr(connection, "commit", None)
    if commit is not None:
        commit()
