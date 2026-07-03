"""SQLite FTS5 lexical retriever — real Okapi BM25 ranking (IDF-weighted),
fielded with per-column boosts.

This replaces ``CatalogIndex._score()``'s ``token in text`` substring overlap,
which has no notion of term rarity (a dataset matching on "brain" scores the
same per-hit as one matching on a rare discriminative term) and cannot express
partial relevance, only inclusion. FTS5's ``bm25()`` implements the standard
BM25 formula (k1=1.2, b=0.75) natively in C, in-process — no server, no extra
runtime dependency beyond the stdlib ``sqlite3`` module (FTS5 is compiled into
CPython's bundled SQLite on every platform Qortex targets).

Design choice: this is a *separate* small SQLite file next to the DuckDB
catalog, not an attempt to run FTS inside DuckDB. DuckDB remains the system of
record for structured facts (the right tool for columnar filtering/facets);
SQLite FTS5 is purpose-built, extremely fast, inverted-index text search (the
right tool for BM25) — using each engine for what it's actually good at beats
forcing one engine to do both.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# dataset_id is UNINDEXED (no bm25 weight slot); the remaining six columns are
# indexed in this order, and _WEIGHTS maps 1:1 onto them.
_TEXT_COLUMNS = ["name", "description", "authors", "tasks", "modalities", "keywords"]
_WEIGHTS = (6.0, 3.0, 2.5, 5.0, 3.0, 2.0)


class LexicalIndex:
    """Disk-backed SQLite FTS5 BM25 index mirroring the DuckDB catalog's text
    fields. Call ``sync()`` after any catalog refresh; querying is read-only
    and safe to call concurrently from multiple threads (separate connections
    recommended for heavy concurrent use — this class holds one connection)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
                dataset_id UNINDEXED,
                name, description, authors, tasks, modalities, keywords,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
        self._con.commit()

    def sync(self, rows: list[dict[str, Any]]) -> int:
        """Full delete-then-insert upsert for the given catalog rows. Returns
        the number of documents written. Cheap enough to call on every
        ``refresh_indexes()`` for corpora up to the tens-of-thousands scale —
        FTS5 insert throughput is not the bottleneck at that size; the
        DuckDB→dict hydration that produces ``rows`` is."""
        cur = self._con.cursor()
        n = 0
        for row in rows:
            dataset_id = row.get("dataset_id")
            if not dataset_id:
                continue
            cur.execute("DELETE FROM docs WHERE dataset_id = ?", (dataset_id,))
            cur.execute(
                "INSERT INTO docs (dataset_id, name, description, authors, tasks, modalities, keywords) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    row.get("name") or "",
                    row.get("description") or "",
                    " ".join(row.get("authors") or []),
                    " ".join(row.get("tasks") or []),
                    " ".join(row.get("modalities") or []),
                    " ".join(row.get("keywords") or []),
                ),
            )
            n += 1
        self._con.commit()
        return n

    def search(self, terms: list[str], *, limit: int = 200) -> list[tuple[str, float]]:
        """Return [(dataset_id, bm25_score)] sorted best-first, score higher
        = more relevant. Empty/all-stopword ``terms`` -> []."""
        if not terms:
            return []
        clauses = [f'"{_escape(t)}"*' for t in terms if t and t.strip()]
        if not clauses:
            return []
        match_query = " OR ".join(clauses)
        weight_args = ", ".join(str(w) for w in _WEIGHTS)
        try:
            cur = self._con.execute(
                f"""
                SELECT dataset_id, bm25(docs, {weight_args}) AS rank
                FROM docs WHERE docs MATCH ?
                ORDER BY rank LIMIT ?
                """,
                (match_query, limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # malformed MATCH syntax from an adversarial/edge-case query term
            # (e.g. a lone operator character) — degrade to no lexical hits
            # rather than raising, so a bad query term never breaks the page.
            return []
        # FTS5's bm25(): lower (more negative) = more relevant; flip sign so
        # "higher = better" holds across every retriever in the fusion stage.
        return [(row[0], -row[1]) for row in rows]

    def close(self) -> None:
        self._con.close()


def _escape(term: str) -> str:
    return term.replace('"', '""')
