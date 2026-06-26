"""Persistent OpenNeuro catalog index.

The catalog is intentionally normalized: dataset-level metadata stays in the
``datasets`` table, while repeatable semantic fields such as modalities, tasks,
authors, keywords, and file-type summaries are indexed in child tables. Search
uses structured filters first and then a transparent weighted text score over
the indexed metadata fields.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


_DATASETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id          TEXT PRIMARY KEY,
    name                TEXT,
    description         TEXT,
    authors             TEXT,
    doi                 TEXT,
    license             TEXT,
    n_subjects          INTEGER,
    n_sessions          INTEGER,
    n_tasks             INTEGER,
    modalities          TEXT,
    tasks               TEXT,
    keywords            TEXT,
    snapshot            TEXT,
    snapshot_created    TEXT,
    n_files             INTEGER,
    total_bytes         BIGINT,
    has_events          INTEGER,
    has_derivatives     INTEGER,
    n_event_files       INTEGER,
    n_derivative_files  INTEGER,
    n_primary_files     INTEGER,
    n_metadata_files    INTEGER,
    raw_metadata        TEXT,
    raw_description     TEXT,
    updated_at          TEXT
);
"""

_CHILD_SCHEMAS = [
    """
    CREATE TABLE IF NOT EXISTS dataset_modalities (
        dataset_id TEXT NOT NULL,
        modality   TEXT NOT NULL,
        PRIMARY KEY (dataset_id, modality)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_tasks (
        dataset_id TEXT NOT NULL,
        task       TEXT NOT NULL,
        PRIMARY KEY (dataset_id, task)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_authors (
        dataset_id TEXT NOT NULL,
        author     TEXT NOT NULL,
        PRIMARY KEY (dataset_id, author)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_keywords (
        dataset_id TEXT NOT NULL,
        keyword    TEXT NOT NULL,
        PRIMARY KEY (dataset_id, keyword)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_file_summaries (
        dataset_id TEXT NOT NULL,
        category   TEXT NOT NULL,
        value      TEXT NOT NULL,
        n_files    INTEGER NOT NULL,
        bytes      BIGINT,
        PRIMARY KEY (dataset_id, category, value)
    );
    """,
]

_DATASET_COLUMNS: dict[str, str] = {
    "description": "TEXT",
    "tasks": "TEXT",
    "keywords": "TEXT",
    "snapshot_created": "TEXT",
    "has_events": "INTEGER",
    "has_derivatives": "INTEGER",
    "n_event_files": "INTEGER",
    "n_derivative_files": "INTEGER",
    "n_primary_files": "INTEGER",
    "n_metadata_files": "INTEGER",
    "raw_metadata": "TEXT",
    "raw_description": "TEXT",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*")


class CatalogIndex:
    """Persistent index of OpenNeuro datasets available for discovery."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect(db_path)
        self._init_schema()

    # ── Schema ───────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._con.execute(_DATASETS_SCHEMA)
        for column, sql_type in _DATASET_COLUMNS.items():
            _ensure_column(self._con, "datasets", column, sql_type)
        for schema in _CHILD_SCHEMAS:
            self._con.execute(schema)
        _commit(self._con)

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, row: dict[str, Any]) -> None:
        """Insert or update a normalized dataset metadata row."""
        dataset_id = _to_text(row.get("dataset_id", "")) or ""
        if not dataset_id:
            return

        authors = _as_list(row.get("authors"))
        modalities = _as_list(row.get("modalities"))
        tasks = _as_list(row.get("tasks"))
        keywords = _as_list(row.get("keywords"))
        description = _to_text(row.get("description")) or _to_text(row.get("name"))
        has_events = _to_bool_int(row.get("has_events"))
        has_derivatives = _to_bool_int(row.get("has_derivatives"))

        self._con.execute(
            """
            INSERT OR REPLACE INTO datasets
                (dataset_id, name, description, authors, doi, license,
                 n_subjects, n_sessions, n_tasks, modalities, tasks, keywords,
                 snapshot, snapshot_created, n_files, total_bytes, has_events,
                 has_derivatives, n_event_files, n_derivative_files,
                 n_primary_files, n_metadata_files, raw_metadata,
                 raw_description, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                dataset_id,
                _to_text(row.get("name")),
                description,
                json.dumps(authors),
                _to_text(row.get("doi")),
                _to_text(row.get("license")),
                _to_int(row.get("n_subjects")),
                _to_int(row.get("n_sessions")),
                _to_int(row.get("n_tasks")) or len(tasks) or None,
                json.dumps(modalities),
                json.dumps(tasks),
                json.dumps(keywords),
                _to_text(row.get("snapshot")),
                _to_text(row.get("snapshot_created")),
                _to_int(row.get("n_files")),
                _to_int(row.get("total_bytes")),
                has_events,
                has_derivatives,
                _to_int(row.get("n_event_files")),
                _to_int(row.get("n_derivative_files")),
                _to_int(row.get("n_primary_files")),
                _to_int(row.get("n_metadata_files")),
                json.dumps(row.get("raw_metadata") or {}, sort_keys=True, default=str),
                json.dumps(row.get("raw_description") or {}, sort_keys=True, default=str),
                _to_text(row.get("updated_at")),
            ],
        )

        self._replace_values("dataset_modalities", "modality", dataset_id, modalities)
        self._replace_values("dataset_tasks", "task", dataset_id, tasks)
        self._replace_values("dataset_authors", "author", dataset_id, authors)
        derived_keywords = sorted(set(keywords) | _derive_keywords(row, modalities, tasks))
        self._replace_values("dataset_keywords", "keyword", dataset_id, derived_keywords)
        self._replace_file_summaries(dataset_id, row.get("file_summaries") or [])
        _commit(self._con)

    def upsert_many(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.upsert(row)

    def _replace_values(
        self,
        table: str,
        column: str,
        dataset_id: str,
        values: list[str],
    ) -> None:
        self._con.execute(f"DELETE FROM {table} WHERE dataset_id = ?", [dataset_id])
        for value in sorted({v.strip() for v in values if v and v.strip()}):
            self._con.execute(
                f"INSERT OR REPLACE INTO {table} (dataset_id, {column}) VALUES (?, ?)",
                [dataset_id, value],
            )

    def _replace_file_summaries(self, dataset_id: str, summaries: list[dict[str, Any]]) -> None:
        self._con.execute("DELETE FROM dataset_file_summaries WHERE dataset_id = ?", [dataset_id])
        for summary in summaries:
            category = _to_text(summary.get("category"))
            value = _to_text(summary.get("value"))
            if not category or not value:
                continue
            self._con.execute(
                """
                INSERT OR REPLACE INTO dataset_file_summaries
                    (dataset_id, category, value, n_files, bytes)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    dataset_id,
                    category,
                    value,
                    _to_int(summary.get("n_files")) or 0,
                    _to_int(summary.get("bytes")),
                ],
            )

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str | None = None,
        modality: str | None = None,
        min_subjects: int | None = None,
        limit: int = 50,
        *,
        task: str | None = None,
        author: str | None = None,
        license: str | None = None,
        max_size_gb: float | None = None,
        has_events: bool | None = None,
        has_derivatives: bool | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search datasets using structured filters and weighted text ranking."""
        rows = self._candidate_rows(
            modality=modality,
            task=task,
            author=author,
            license=license,
            min_subjects=min_subjects,
            max_size_gb=max_size_gb,
            has_events=has_events,
            has_derivatives=has_derivatives,
        )
        tokens = _tokens(query)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            profile = self._hydrate_row(row)
            score = _score(profile, tokens)
            if tokens and score <= 0:
                continue
            profile["score"] = round(score, 4)
            ranked.append((score, profile))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].get("n_subjects") or -1,
                item[1].get("n_files") or -1,
                item[1].get("dataset_id") or "",
            ),
            reverse=True,
        )
        selected = ranked[offset : offset + limit]
        return [row for _, row in selected]

    def _candidate_rows(
        self,
        *,
        modality: str | None,
        task: str | None,
        author: str | None,
        license: str | None,
        min_subjects: int | None,
        max_size_gb: float | None,
        has_events: bool | None,
        has_derivatives: bool | None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if modality:
            clauses.append(
                "EXISTS (SELECT 1 FROM dataset_modalities m WHERE m.dataset_id = datasets.dataset_id AND LOWER(m.modality) = LOWER(?))"
            )
            params.append(modality)
        if task:
            clauses.append(
                "EXISTS (SELECT 1 FROM dataset_tasks t WHERE t.dataset_id = datasets.dataset_id AND LOWER(t.task) = LOWER(?))"
            )
            params.append(task)
        if author:
            clauses.append(
                "EXISTS (SELECT 1 FROM dataset_authors a WHERE a.dataset_id = datasets.dataset_id AND LOWER(a.author) LIKE LOWER(?))"
            )
            params.append(f"%{author}%")
        if license:
            clauses.append("LOWER(license) = LOWER(?)")
            params.append(license)
        if min_subjects is not None:
            clauses.append("n_subjects >= ?")
            params.append(min_subjects)
        if max_size_gb is not None:
            clauses.append("total_bytes <= ?")
            params.append(int(max_size_gb * 1e9))
        if has_events is not None:
            clauses.append("has_events = ?")
            params.append(1 if has_events else 0)
        if has_derivatives is not None:
            clauses.append("has_derivatives = ?")
            params.append(1 if has_derivatives else 0)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cursor = self._con.execute(f"SELECT * FROM datasets {where}", params)
        cols = _columns(cursor, self._con)
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get(self, dataset_id: str) -> dict[str, Any] | None:
        cursor = self._con.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?", [dataset_id]
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        cols = _columns(cursor, self._con)
        return self._hydrate_row(dict(zip(cols, rows[0])))

    def profile(self, dataset_id: str) -> dict[str, Any] | None:
        """Return a fully hydrated catalog profile for one dataset."""
        return self.get(dataset_id)

    def facets(self, *, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Return common discovery facets for UI and CLI consumers."""
        return {
            "modalities": self._facet("dataset_modalities", "modality", limit=limit),
            "tasks": self._facet("dataset_tasks", "task", limit=limit),
            "licenses": self._license_facet(limit=limit),
            "keywords": self._facet("dataset_keywords", "keyword", limit=limit),
        }

    def count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]

    def close(self) -> None:
        self._con.close()

    def _hydrate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["authors"] = _json_list(row.get("authors"))
        row["modalities"] = _json_list(row.get("modalities"))
        row["tasks"] = _json_list(row.get("tasks"))
        row["keywords"] = _json_list(row.get("keywords"))
        row["raw_metadata"] = _json_obj(row.get("raw_metadata"))
        row["raw_description"] = _json_obj(row.get("raw_description"))
        row["has_events"] = _to_bool(row.get("has_events"))
        row["has_derivatives"] = _to_bool(row.get("has_derivatives"))
        row["file_summaries"] = self._file_summaries(row["dataset_id"])
        return row

    def _file_summaries(self, dataset_id: str) -> list[dict[str, Any]]:
        cursor = self._con.execute(
            """
            SELECT category, value, n_files, bytes
            FROM dataset_file_summaries
            WHERE dataset_id = ?
            ORDER BY category, n_files DESC, value
            """,
            [dataset_id],
        )
        return [
            {"category": category, "value": value, "n_files": n_files, "bytes": bytes_}
            for category, value, n_files, bytes_ in cursor.fetchall()
        ]

    def _facet(self, table: str, column: str, *, limit: int) -> list[dict[str, Any]]:
        cursor = self._con.execute(
            f"""
            SELECT {column}, COUNT(*) AS n
            FROM {table}
            GROUP BY {column}
            ORDER BY n DESC, {column}
            LIMIT ?
            """,
            [limit],
        )
        return [{"value": value, "n": n} for value, n in cursor.fetchall()]

    def _license_facet(self, *, limit: int) -> list[dict[str, Any]]:
        cursor = self._con.execute(
            """
            SELECT license, COUNT(*) AS n
            FROM datasets
            WHERE license IS NOT NULL AND license != ''
            GROUP BY license
            ORDER BY n DESC, license
            LIMIT ?
            """,
            [limit],
        )
        return [{"value": value, "n": n} for value, n in cursor.fetchall()]


def _connect(path: Path) -> Any:
    try:
        import duckdb
        return duckdb.connect(str(path))
    except ImportError:
        return sqlite3.connect(str(path))


def _commit(connection: Any) -> None:
    commit = getattr(connection, "commit", None)
    if commit is not None:
        commit()


def _is_duckdb(connection: Any) -> bool:
    return type(connection).__module__.startswith("duckdb")


def _ensure_column(connection: Any, table: str, column: str, sql_type: str) -> None:
    """Add *column* to *table* if it does not already exist.

    Uses INFORMATION_SCHEMA for DuckDB (ANSI-standard) and PRAGMA for SQLite.
    """
    try:
        if _is_duckdb(connection):
            rows = connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? AND column_name = ?",
                [table, column],
            ).fetchall()
            if rows:
                return
        else:
            rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
            if any(row[1] == column for row in rows):
                return
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    except Exception:
        pass


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
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


def _to_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, list):
                return [str(item) for item in decoded if str(item).strip()]
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return _as_list(value)
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    return []


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _columns(cursor: Any, connection: Any) -> list[str]:
    description = getattr(cursor, "description", None)
    if description:
        return [item[0] for item in description]
    # DuckDB returns column names directly via .columns on the relation result
    columns = getattr(cursor, "columns", None)
    if columns:
        return list(columns)
    description = getattr(connection, "description", None)
    return [item[0] for item in description or []]


def _tokens(query: str | None) -> list[str]:
    if not query:
        return []
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(query)]


def _score(row: dict[str, Any], tokens: list[str]) -> float:
    if not tokens:
        return float(row.get("n_subjects") or 0)
    fields = {
        "dataset_id": (row.get("dataset_id") or "", 8.0),
        "name": (row.get("name") or "", 6.0),
        "description": (row.get("description") or "", 3.0),
        "doi": (row.get("doi") or "", 2.0),
        "authors": (" ".join(row.get("authors") or []), 2.5),
        "modalities": (" ".join(row.get("modalities") or []), 3.0),
        "tasks": (" ".join(row.get("tasks") or []), 4.0),
        "keywords": (" ".join(row.get("keywords") or []), 2.0),
        "license": (row.get("license") or "", 1.0),
    }
    score = 0.0
    for token in tokens:
        for value, weight in fields.values():
            text = value.lower()
            if text == token:
                score += weight * 2.0
            elif token in text:
                score += weight
    if row.get("n_subjects"):
        score += min(float(row["n_subjects"]) / 1000.0, 0.5)
    return score


def _derive_keywords(
    row: dict[str, Any],
    modalities: list[str],
    tasks: list[str],
) -> set[str]:
    values: set[str] = set()
    for value in [row.get("dataset_id"), row.get("license"), row.get("doi")]:
        values.update(_tokens(_to_text(value)))
    for value in modalities + tasks:
        values.update(_tokens(value))
    for field in ["name", "description"]:
        values.update(token for token in _tokens(_to_text(row.get(field))) if len(token) >= 4)
    return values


def summarize_manifest_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Digest OpenNeuro file metadata into compact catalog features."""
    extensions: Counter[str] = Counter()
    datatypes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    n_events = 0
    n_derivatives = 0
    n_metadata = 0
    n_primary = 0
    bytes_by_extension: Counter[str] = Counter()
    bytes_by_datatype: Counter[str] = Counter()

    for raw in files:
        path = str(raw.get("filename") or raw.get("path") or "")
        if not path or bool(raw.get("directory")):
            continue
        size = _to_int(raw.get("size")) or 0
        extension = _extension(path)
        datatype = _datatype(path)
        suffix = _suffix(path, extension)
        if extension:
            extensions[extension] += 1
            bytes_by_extension[extension] += size
        if datatype:
            datatypes[datatype] += 1
            bytes_by_datatype[datatype] += size
        if suffix:
            suffixes[suffix] += 1
        if suffix == "events":
            n_events += 1
        if path.startswith("derivatives/") or "/derivatives/" in path:
            n_derivatives += 1
        if extension in {".json", ".tsv", ".csv", ".bvec", ".bval"} or path.rsplit("/", 1)[-1] in {
            "README",
            "CHANGES",
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
        }:
            n_metadata += 1
        if suffix and suffix not in {"events", "channels", "electrodes", "coordsystem", "scans"} and extension not in {".json", ".tsv", ".csv", ".bvec", ".bval"}:
            n_primary += 1

    summaries: list[dict[str, Any]] = []
    summaries.extend(_counter_summaries("extension", extensions, bytes_by_extension))
    summaries.extend(_counter_summaries("datatype", datatypes, bytes_by_datatype))
    summaries.extend(_counter_summaries("suffix", suffixes, Counter()))
    return {
        "has_events": n_events > 0,
        "has_derivatives": n_derivatives > 0,
        "n_event_files": n_events,
        "n_derivative_files": n_derivatives,
        "n_primary_files": n_primary,
        "n_metadata_files": n_metadata,
        "file_summaries": summaries,
    }


def _counter_summaries(
    category: str,
    counts: Counter[str],
    bytes_by_value: Counter[str],
) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "value": value,
            "n_files": count,
            "bytes": bytes_by_value.get(value, 0),
        }
        for value, count in counts.most_common()
    ]


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    for compound in (".nii.gz", ".tar.gz"):
        if name.endswith(compound):
            return compound
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def _datatype(path: str) -> str:
    parts = path.split("/")
    for part in parts:
        if part in {"anat", "func", "dwi", "fmap", "eeg", "meg", "ieeg", "nirs", "pet", "beh"}:
            return part
    return ""


def _suffix(path: str, extension: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if extension and name.endswith(extension):
        name = name[: -len(extension)]
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[-1]
