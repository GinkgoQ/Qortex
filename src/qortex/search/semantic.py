"""Semantic retriever — structure-derived "cards" (not raw descriptions;
OpenNeuro free text is inconsistent, see qortex-atlas.md §3) embedded via a
Latent Semantic Analysis pipeline (TF-IDF + truncated SVD) that runs fully
offline with no model download, and transparently upgrades to a locally cached
sentence-transformer encoder if one is available — best-effort, never
required, never blocking on network I/O.

Engineering note on scale (the "fastest possible stack" decision): for the
corpus sizes Qortex indexes today (hundreds to low tens-of-thousands of
datasets), an exact brute-force cosine search via one dense matrix multiply is
*faster end-to-end* than building and querying an approximate index — index
construction overhead dominates at this scale, and a single BLAS-backed
matmul is already vectorized. The engine only switches to FAISS's exact
``IndexFlatIP`` past ``_FAISS_THRESHOLD`` documents, where the constant-factor
win starts to matter; it deliberately does *not* reach for an approximate
(HNSW/IVF) index at all — recall loss from approximation isn't worth it until
the corpus is orders of magnitude larger than anything in scope here, and
every extra moving part is a maintenance/debugging cost.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FAISS_THRESHOLD = 5_000
_SVD_COMPONENTS = 128


def synthesize_card(row: dict[str, Any]) -> str:
    """Deterministically render a catalog row's *structure* as prose, so the
    embedding captures confirmed facts, not just whatever free text an author
    happened to write. See qortex-atlas-search-engine.md §4.3 — this is the
    exact mechanism that lets a query like "cross-subject movement decoding"
    match a dataset whose description says nothing about decoding but whose
    labels/protocol clearly fit.
    """
    parts: list[str] = []
    modalities = row.get("modalities") or []
    tasks = row.get("tasks") or []
    if modalities:
        parts.append(f"{'/'.join(modalities)} dataset")
    if tasks:
        parts.append("task: " + ", ".join(tasks[:5]))
    n_subjects = row.get("n_subjects")
    if n_subjects:
        parts.append(f"{n_subjects} subjects")
    if row.get("has_events"):
        parts.append("has event-related trial data")
    if row.get("has_derivatives"):
        parts.append("includes derivative/preprocessed data")
    license_ = row.get("license")
    if license_:
        parts.append(f"license {license_}")
    keywords = row.get("keywords") or []
    if keywords:
        parts.append("keywords: " + ", ".join(keywords[:10]))
    name = row.get("name") or ""
    description = (row.get("description") or "")[:500]
    card = f"{name}. " + " ".join(parts)
    if description and description.lower() not in card.lower():
        card += f". {description}"
    return card.strip()


@dataclass
class _FitState:
    dataset_ids: list[str]
    vectors: np.ndarray  # (n_docs, dim), L2-normalized rows
    content_hash: str
    backend_name: str


class SemanticIndex:
    """Fit once per catalog snapshot, query many times. Not thread-safe during
    ``fit()``; ``search()`` is read-only and safe to call concurrently once
    fitted."""

    def __init__(self, cache_path: Path | None = None, *, prefer_transformer: bool = False) -> None:
        self._cache_path = cache_path
        self._prefer_transformer = prefer_transformer
        self._state: _FitState | None = None
        self._vectorizer: Any = None
        self._svd: Any = None
        self._transformer: Any = None
        self._faiss_index: Any = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    @property
    def backend_name(self) -> str | None:
        return self._state.backend_name if self._state else None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self._state = _FitState(dataset_ids=[], vectors=np.zeros((0, 1), dtype=np.float32), content_hash="", backend_name="empty")
            return

        cards = [synthesize_card(r) for r in rows]
        ids = [r["dataset_id"] for r in rows]
        content_hash = hashlib.sha256("\x1f".join(cards).encode("utf-8")).hexdigest()

        if self._cache_path and self._try_load_cache(content_hash):
            return

        backend_name, vectors = self._embed_fit(cards)
        vectors = _l2_normalize(vectors)
        self._state = _FitState(dataset_ids=ids, vectors=vectors, content_hash=content_hash, backend_name=backend_name)
        self._maybe_build_faiss()
        if self._cache_path:
            self._save_cache()

    def search(self, query_text: str, *, limit: int = 200) -> list[tuple[str, float]]:
        if not self._state or not self._state.dataset_ids or not query_text.strip():
            return []
        query_vec = self._embed_query(query_text)
        query_vec = _l2_normalize(query_vec.reshape(1, -1))[0]

        if self._faiss_index is not None:
            k = min(limit, len(self._state.dataset_ids))
            scores, idx = self._faiss_index.search(query_vec.reshape(1, -1).astype("float32"), k)
            return [
                (self._state.dataset_ids[i], float(s))
                for s, i in zip(scores[0], idx[0])
                if i >= 0
            ]

        sims = self._state.vectors @ query_vec
        top = np.argsort(-sims)[:limit]
        return [(self._state.dataset_ids[i], float(sims[i])) for i in top]

    # ── embedding backends ──────────────────────────────────────────────

    def _embed_fit(self, cards: list[str]) -> tuple[str, np.ndarray]:
        if self._prefer_transformer:
            vectors = self._try_transformer_embed(cards)
            if vectors is not None:
                return "sentence-transformer", vectors
        return "lsa-tfidf-svd", self._lsa_embed_fit(cards)

    def _embed_query(self, text: str) -> np.ndarray:
        if self._state and self._state.backend_name == "sentence-transformer" and self._transformer is not None:
            return np.asarray(self._transformer.encode([text])[0], dtype=np.float32)
        return self._lsa_embed_transform([text])[0]

    def _try_transformer_embed(self, cards: list[str]) -> np.ndarray | None:
        """Best-effort only: requires a locally cached model (``local_files_only``
        — this engine never triggers a network download as a side effect of a
        search). Falls back silently to the always-available LSA backend."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.info("sentence-transformer backend unavailable (%s); using local LSA embedding", exc)
            return None
        self._transformer = model
        return np.asarray(model.encode(cards, show_progress_bar=False), dtype=np.float32)

    def _lsa_embed_fit(self, cards: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+-]{1,}\b",
        )
        tfidf = self._vectorizer.fit_transform(cards)
        n_components = min(_SVD_COMPONENTS, max(2, min(tfidf.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        return self._svd.fit_transform(tfidf).astype(np.float32)

    def _lsa_embed_transform(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("SemanticIndex.fit() must be called before search()")
        tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(tfidf).astype(np.float32)

    # ── optional FAISS acceleration ─────────────────────────────────────

    def _maybe_build_faiss(self) -> None:
        assert self._state is not None
        if len(self._state.dataset_ids) < _FAISS_THRESHOLD:
            self._faiss_index = None
            return
        try:
            import faiss
        except ImportError:
            self._faiss_index = None
            return
        dim = self._state.vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # exact inner product on L2-normalized rows == cosine
        index.add(self._state.vectors.astype("float32"))
        self._faiss_index = index

    # ── disk cache (skip re-embedding an unchanged corpus) ──────────────

    def _try_load_cache(self, content_hash: str) -> bool:
        if not self._cache_path or not self._cache_path.exists():
            return False
        try:
            data = np.load(self._cache_path, allow_pickle=True)
            if str(data["content_hash"]) != content_hash:
                return False
            self._state = _FitState(
                dataset_ids=list(data["dataset_ids"]),
                vectors=data["vectors"],
                content_hash=content_hash,
                backend_name=str(data["backend_name"]),
            )
            self._vectorizer = pickle.loads(data["vectorizer"].tobytes())
            self._svd = pickle.loads(data["svd"].tobytes())
            self._maybe_build_faiss()
            return True
        except Exception as exc:  # pragma: no cover - corrupt/foreign cache file
            logger.info("semantic index cache load failed (%s); refitting", exc)
            return False

    def _save_cache(self) -> None:
        assert self._state is not None and self._cache_path is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._cache_path,
            content_hash=self._state.content_hash,
            dataset_ids=np.array(self._state.dataset_ids, dtype=object),
            vectors=self._state.vectors,
            backend_name=self._state.backend_name,
            vectorizer=np.frombuffer(pickle.dumps(self._vectorizer), dtype=np.uint8),
            svd=np.frombuffer(pickle.dumps(self._svd), dtype=np.uint8),
        )


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
