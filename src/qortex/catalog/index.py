"""Persistent dataset catalog index."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sqlite3


_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    TEXT PRIMARY KEY,
    name          TEXT,
    authors       TEXT,
    description   TEXT,
    doi           TEXT,
    license       TEXT,
    n_subjects    INTEGER,
    n_sessions    INTEGER,
    n_tasks       INTEGER,
    modalities    TEXT,    -- JSON array
    snapshot      TEXT,
    n_files       INTEGER,
    total_bytes   BIGINT,
    updated_at    TEXT
);
"""


class CatalogIndex:
    """Persistent index of OpenNeuro datasets available for download."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect(db_path)
        self._con.execute(_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, row: dict[str, Any]) -> None:
        import json
        self._con.execute(
            """
            INSERT OR REPLACE INTO datasets
                (dataset_id, name, authors, description, doi, license,
                 n_subjects, n_sessions, n_tasks, modalities, snapshot,
                 n_files, total_bytes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                _to_text(row.get("dataset_id", "")) or "",
                _to_text(row.get("name")),
                _to_text(row.get("authors")),
                _to_text(row.get("description")),
                _to_text(row.get("doi")),
                _to_text(row.get("license")),
                _to_int(row.get("n_subjects")),
                _to_int(row.get("n_sessions")),
                _to_int(row.get("n_tasks")),
                json.dumps(row.get("modalities", [])),
                _to_text(row.get("snapshot")),
                _to_int(row.get("n_files")),
                _to_int(row.get("total_bytes")),
                _to_text(row.get("updated_at")),
            ],
        )
        _commit(self._con)

    def upsert_many(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.upsert(row)

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str | None = None,
        modality: str | None = None,
        min_subjects: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        import json

        clauses = []
        params: list[Any] = []

        if query:
            clauses.append(
                "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(authors) LIKE ?)"
            )
            q = f"%{query.lower()}%"
            params.extend([q, q, q])
        if modality:
            clauses.append("LOWER(modalities) LIKE ?")
            params.append(f"%{modality.lower()}%")
        if min_subjects is not None:
            clauses.append("n_subjects >= ?")
            params.append(min_subjects)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM datasets {where} ORDER BY n_subjects IS NULL, n_subjects DESC LIMIT ?"
        params.append(limit)

        cursor = self._con.execute(sql, params)
        rows = cursor.fetchall()
        cols = _columns(cursor, self._con)
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            try:
                d["modalities"] = json.loads(d["modalities"] or "[]")
            except Exception:
                d["modalities"] = []
            result.append(d)
        return result

    def get(self, dataset_id: str) -> dict[str, Any] | None:
        import json
        cursor = self._con.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", [dataset_id]
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        cols = _columns(cursor, self._con)
        d = dict(zip(cols, rows[0]))
        try:
            d["modalities"] = json.loads(d["modalities"] or "[]")
        except Exception:
            d["modalities"] = []
        return d

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]

    def close(self) -> None:
        self._con.close()


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


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    import json

    return json.dumps(value, sort_keys=True, default=str)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _columns(cursor: Any, connection: Any) -> list[str]:
    description = getattr(cursor, "description", None)
    if description is None:
        description = getattr(connection, "description", None)
    return [item[0] for item in description or []]
