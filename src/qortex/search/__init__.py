"""Qortex's multi-method dataset search engine.

Replaces ``CatalogIndex.search()``'s single substring-overlap scorer with a
staged pipeline: a deterministic query compiler, three parallel retrievers
(structured/faceted, BM25 lexical, semantic/LSA), Reciprocal Rank Fusion,
optional structural fitness re-ranking (reusing
``qortex.inspect.selector.DatasetSelector``), and evidence-partitioned
filtering (reusing ``qortex.checks.EvidenceState``). See
``SearchEngine.search`` for the primary entry point.
"""

from __future__ import annotations

from qortex.search.compiler import Constraint, QueryPlan, compile_query
from qortex.search.engine import SearchEngine, SearchResponse, SearchResult
from qortex.search.evidence import EvidencePartition, partition_has_derivatives, partition_has_events
from qortex.search.fusion import FusedResult, reciprocal_rank_fusion
from qortex.search.negative_space import NegativeSpaceReport, build_negative_space_report
from qortex.search.ontology import Ontology, canonical_modalities, mine_synonym_clusters

__all__ = [
    "SearchEngine",
    "SearchResponse",
    "SearchResult",
    "QueryPlan",
    "Constraint",
    "compile_query",
    "Ontology",
    "canonical_modalities",
    "mine_synonym_clusters",
    "FusedResult",
    "reciprocal_rank_fusion",
    "EvidencePartition",
    "partition_has_events",
    "partition_has_derivatives",
    "NegativeSpaceReport",
    "build_negative_space_report",
]
