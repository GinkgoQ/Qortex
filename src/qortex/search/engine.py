"""SearchEngine — orchestrates the full retrieval pipeline:

    query compiler
      -> {structured, lexical (BM25), semantic (LSA/embedding)} retrievers
      -> Reciprocal Rank Fusion
      -> structural fitness re-rank (reuses qortex.inspect.selector.DatasetSelector
         — the existing transparent, hard-fail-aware, evidence-decomposed
         scorer; not rebuilt, only promoted to run over the fused shortlist)
      -> evidence-partitioned filtering (qortex.checks.EvidenceState)
      -> negative-space summary

Every stage is local-first: structured filtering and BM25 lexical search never
touch the network; the semantic index embeds locally (LSA by default, no
model download); only the optional ``deep=True`` structural re-rank stage may
call the live OpenNeuro API (via the existing ``DatasetSelector``), and even
then only for the fused shortlist, never the whole corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.catalog.index import CatalogIndex
from qortex.checks import EvidenceState
from qortex.core.config import get_config
from qortex.search.compiler import QueryPlan, compile_query
from qortex.search.evidence import EvidencePartition, partition_has_events
from qortex.search.fusion import FusedResult, reciprocal_rank_fusion
from qortex.search.lexical import LexicalIndex
from qortex.search.negative_space import NegativeSpaceReport, build_negative_space_report
from qortex.search.ontology import Ontology
from qortex.search.semantic import SemanticIndex


@dataclass
class SearchResult:
    dataset_id: str
    fused_score: float
    row: dict[str, Any]
    matched_by: list[str]           # which retrievers returned this id
    explanation: list[str]          # human-readable "why" lines, per retriever
    evidence_flags: dict[str, str]  # field -> EvidenceState value
    fitness: Any = None             # DatasetFitness, only populated when deep=True


@dataclass
class SearchResponse:
    results: list[SearchResult]
    plan: QueryPlan
    negative_space: NegativeSpaceReport | None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Query: {self.plan.describe()}", ""]
        for r in self.results:
            row = r.row
            mods = ", ".join(row.get("modalities") or [])
            evidence = ", ".join(f"{k}={v}" for k, v in r.evidence_flags.items())
            lines.append(
                f"[{r.dataset_id}] {row.get('name') or '(no name)'}  "
                f"score={r.fused_score:.4f}  subjects={row.get('n_subjects')}  "
                f"modalities={mods}  via={'+'.join(r.matched_by)}  {evidence}"
            )
        if self.negative_space:
            lines += ["", self.negative_space.render()]
        return "\n".join(lines)


class SearchEngine:
    """One instance per catalog; holds the lexical + semantic indexes in
    memory across calls (this is the object a long-lived process, e.g.
    Atlas's FastAPI app, should keep as a singleton — mirroring the
    ``atlas_cache.TTLCache`` pattern already used for other expensive Qortex
    objects, rather than rebuilding indexes per request)."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        cfg = get_config()
        self._catalog_path = catalog_path or (cfg.cache_dir / "catalog" / "catalog.duckdb")
        self._lexical_path = self._catalog_path.with_name("catalog_fts.sqlite")
        self._semantic_path = self._catalog_path.with_name("catalog_semantic.npz")
        self._ontology = Ontology.default()
        self._lexical: LexicalIndex | None = None
        self._semantic: SemanticIndex | None = None

    # ── index lifecycle ───────────────────────────────────────────────

    def refresh_indexes(self) -> dict[str, Any]:
        """(Re)build the lexical + semantic indexes from the current catalog
        state. Cheap to call on every catalog refresh: both indexes hash
        their input and skip re-embedding/re-indexing work for documents that
        haven't changed (``SemanticIndex`` at the whole-corpus level via
        content hash; ``LexicalIndex`` is a fast delete+insert per row either
        way, since FTS5 insert throughput isn't the bottleneck at this
        corpus scale)."""
        t0 = time.perf_counter()
        catalog = CatalogIndex(self._catalog_path)
        try:
            rows = catalog.all_hydrated_rows()
        finally:
            catalog.close()

        self._ontology.mine_from_rows(rows)

        self._lexical = self._lexical or LexicalIndex(self._lexical_path)
        n_lexical = self._lexical.sync(rows)

        self._semantic = self._semantic or SemanticIndex(self._semantic_path)
        self._semantic.fit(rows)

        return {
            "n_datasets": len(rows),
            "n_lexical_docs": n_lexical,
            "semantic_backend": self._semantic.backend_name,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    def _ensure_indexes(self) -> None:
        if self._lexical is None or self._semantic is None:
            self.refresh_indexes()

    # ── search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str | None = None,
        *,
        modality: str | None = None,
        min_subjects: int | None = None,
        max_size_gb: float | None = None,
        license_open: bool | None = None,
        has_events: bool | None = None,
        include_unknown_evidence: bool = True,
        deep: bool = False,
        limit: int = 20,
    ) -> SearchResponse:
        """Run the full pipeline. ``deep=True`` additionally promotes the
        fused shortlist through ``DatasetSelector`` for a transparent,
        dimension-decomposed fitness score (may call the live OpenNeuro API
        for engagement/BIDS-version data — opt-in, never the default)."""
        self._ensure_indexes()
        assert self._lexical is not None and self._semantic is not None
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        plan = compile_query(
            query,
            ontology=self._ontology,
            modality=modality,
            min_subjects=min_subjects,
            max_size_gb=max_size_gb,
            license_open=license_open,
            has_events=has_events,
        )
        timings["compile_ms"] = (time.perf_counter() - t0) * 1000

        catalog = CatalogIndex(self._catalog_path)
        try:
            t0 = time.perf_counter()
            scope_candidates, structural_candidates = self._structural_retrieve(catalog, plan)
            timings["structured_ms"] = (time.perf_counter() - t0) * 1000

            by_id = {r["dataset_id"]: r for r in structural_candidates}
            admissible_ids = set(by_id)

            t0 = time.perf_counter()
            lexical_hits = self._lexical.search(plan.lexical_terms, limit=max(200, limit * 10))
            lexical_hits = [(d, s) for d, s in lexical_hits if d in admissible_ids]
            timings["lexical_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            semantic_hits = self._semantic.search(plan.semantic_text, limit=max(200, limit * 10))
            semantic_hits = [(d, s) for d, s in semantic_hits if d in admissible_ids]
            timings["semantic_ms"] = (time.perf_counter() - t0) * 1000

            ranked_lists: dict[str, list[tuple[str, float]]] = {}
            weights: dict[str, float] = {}
            if lexical_hits:
                ranked_lists["lexical"] = lexical_hits
                weights["lexical"] = 1.0
            if semantic_hits:
                ranked_lists["semantic"] = semantic_hits
                # a crisp free-text query with real lexical hits leans on BM25;
                # a vague/conceptual query (few/no lexical hits) leans on
                # semantics — the query-adaptive weighting the design doc calls
                # for, driven by whether the compiler found any soft terms at
                # all and whether lexical actually returned anything.
                weights["semantic"] = 1.0 if (plan.soft_terms and not lexical_hits) else 0.5
            if not ranked_lists:
                # pure structural browse (no free text at all): rank by
                # subject count as the best available quality proxy.
                ranked_lists["structured"] = sorted(
                    ((r["dataset_id"], float(r.get("n_subjects") or 0)) for r in structural_candidates),
                    key=lambda t: t[1],
                    reverse=True,
                )
                weights["structured"] = 1.0

            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
            timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

            events_partition = partition_has_events(structural_candidates)

            t0 = time.perf_counter()
            fitness_by_id: dict[str, Any] = {}
            if deep:
                fitness_by_id = self._structural_rerank(fused[: max(limit * 3, 30)], plan)
                fused = self._apply_fitness_order(fused, fitness_by_id)
            timings["structural_rerank_ms"] = (time.perf_counter() - t0) * 1000

            results = self._build_results(fused[:limit], by_id, events_partition, fitness_by_id)

            admitted_ids = {r.dataset_id for r in results}
            # Negative-space accounting must see the *topical* scope (before
            # numeric hard constraints were applied), not just the already-
            # admissible set — otherwise every near-miss was already discarded
            # by the structural retriever's SQL filter and there is nothing
            # left to diagnose. See _structural_retrieve's scope/admissible
            # split below.
            negative_space = build_negative_space_report(
                all_candidates=scope_candidates,
                admitted_ids=admitted_ids,
                hard_constraints={k: v for k, v in plan.hard.items() if k not in {"modality", "license_open"}},
            )
            if include_unknown_evidence:
                negative_space.n_unknown_resolvable = len(events_partition.ids(EvidenceState.unknown))

            return SearchResponse(results=results, plan=plan, negative_space=negative_space, timings_ms=timings)
        finally:
            catalog.close()

    # ── stage implementations ────────────────────────────────────────

    def _structural_retrieve(
        self, catalog: CatalogIndex, plan: QueryPlan
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Two-tier structural retrieval, returning ``(scope, admissible)``.

        ``scope`` applies only *topical* hard constraints (modality, license)
        at the SQL level. ``admissible`` additionally applies the *numeric*
        hard constraints (min_subjects, max_size_gb) in Python, in-process,
        over ``scope`` — kept separate from the SQL filter specifically so
        near-misses on numeric constraints are still visible to negative-space
        reporting (§8 of qortex-atlas-search-engine.md). Folding min_subjects
        straight into the SQL WHERE clause (as an earlier version of this
        method did) is cheaper but silently discards exactly the "58 rejected,
        28 too few subjects" diagnosis this engine exists to produce.
        """
        modality_constraint = plan.hard.get("modality")
        license_c = plan.hard.get("license_open")
        scope = catalog.search_candidates(
            modality=sorted(modality_constraint.value) if modality_constraint else None,
            license_open=bool(license_c.value) if license_c else None,
        )

        min_subjects_c = plan.hard.get("min_subjects")
        max_size_c = plan.hard.get("max_size_gb")
        if min_subjects_c is None and max_size_c is None:
            return scope, scope

        admissible = [
            row
            for row in scope
            if (min_subjects_c is None or (row.get("n_subjects") or 0) >= min_subjects_c.value)
            and (max_size_c is None or (row.get("total_bytes") or 0) <= max_size_c.value * 1e9)
        ]
        return scope, admissible

    def _structural_rerank(self, fused: list[FusedResult], plan: QueryPlan) -> dict[str, Any]:
        """Promote the existing DatasetFitness engine to the universal Stage-3
        re-ranker (qortex-atlas-search-engine.md §5.2) — reused, not rebuilt.
        ``tier2_api=False`` by default at this call site: Tier 2 makes one
        live HTTP call per candidate, which is fine for a handful of
        goal-ranked candidates but not something a search call should trigger
        implicitly for up to ``limit*3`` datasets."""
        from qortex.inspect.selector import DatasetSelector, ResearchGoal

        modality_constraint = plan.hard.get("modality")
        goal = ResearchGoal(
            modality=sorted(modality_constraint.value)[0] if modality_constraint else None,
            task_keywords=plan.soft_terms,
            min_subjects=int(plan.hard["min_subjects"].value) if "min_subjects" in plan.hard else None,
            max_size_gb=float(plan.hard["max_size_gb"].value) if "max_size_gb" in plan.hard else None,
            license_must_be_open="license_open" in plan.hard,
        )
        selector = DatasetSelector()
        ids = [fr.dataset_id for fr in fused]
        if not ids:
            return {}
        try:
            fitness_list = selector.rank(ids, goal, tier2_api=False)
        except Exception:
            return {}
        return {f.dataset_id: f for f in fitness_list}

    @staticmethod
    def _apply_fitness_order(fused: list[FusedResult], fitness_by_id: dict[str, Any]) -> list[FusedResult]:
        if not fitness_by_id:
            return fused

        def sort_key(fr: FusedResult) -> tuple[float, float]:
            fitness = fitness_by_id.get(fr.dataset_id)
            if fitness is None:
                return (-1.0, fr.fused_score)
            if fitness.hard_fail:
                return (-2.0, fr.fused_score)  # hard-failed candidates sink to the bottom
            return (fitness.total_score, fr.fused_score)

        return sorted(fused, key=sort_key, reverse=True)

    @staticmethod
    def _build_results(
        fused: list[FusedResult],
        by_id: dict[str, Any],
        events_partition: EvidencePartition,
        fitness_by_id: dict[str, Any],
    ) -> list[SearchResult]:
        results = []
        for fr in fused:
            explanation = [
                f"{retriever}: rank {rank} (score {score:.3f})"
                for retriever, (rank, score) in sorted(fr.contributions.items())
            ]
            results.append(
                SearchResult(
                    dataset_id=fr.dataset_id,
                    fused_score=fr.fused_score,
                    row=by_id.get(fr.dataset_id, {}),
                    matched_by=sorted(fr.contributions.keys()),
                    explanation=explanation,
                    evidence_flags={"has_events": events_partition.state_of(fr.dataset_id).value},
                    fitness=fitness_by_id.get(fr.dataset_id),
                )
            )
        return results
