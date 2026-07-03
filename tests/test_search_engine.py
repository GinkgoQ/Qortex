"""Tests for qortex.search — the multi-method dataset search engine.

Unit tests cover the deterministic pieces (query compiler, RRF math,
synonym clustering, evidence partitioning) with no I/O. The end-to-end test
exercises SearchEngine against a small synthetic catalog built in a temp
directory, so it needs no network access and no pre-existing local catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qortex.catalog.index import CatalogIndex
from qortex.checks import EvidenceState
from qortex.search.compiler import compile_query
from qortex.search.evidence import partition_has_events
from qortex.search.fusion import reciprocal_rank_fusion
from qortex.search.negative_space import build_negative_space_report
from qortex.search.ontology import Ontology, canonical_modalities, mine_synonym_clusters


# ── Query compiler ──────────────────────────────────────────────────────────

class TestQueryCompiler:
    def test_extracts_min_subjects_from_free_text(self):
        plan = compile_query("EEG motor imagery with at least 40 subjects")
        assert "min_subjects" in plan.hard
        assert plan.hard["min_subjects"].value == 40
        assert plan.hard["min_subjects"].op == "ge"
        assert plan.provenance["min_subjects"] == "grammar"

    def test_extracts_max_size(self):
        plan = compile_query("mri datasets under 5 gb")
        assert plan.hard["max_size_gb"].value == 5.0
        assert plan.hard["max_size_gb"].op == "le"

    def test_open_license_phrase(self):
        plan = compile_query("open license eeg dataset")
        assert plan.hard["license_open"].value is True

    def test_modality_detected_via_ontology(self):
        plan = compile_query("resting state fmri dataset")
        assert plan.hard["modality"].value == {"mri"}
        assert plan.provenance["modality"] == "ontology"

    def test_explicit_kwargs_take_precedence_over_grammar(self):
        plan = compile_query("at least 5 subjects", min_subjects=999)
        assert plan.hard["min_subjects"].value == 999
        assert plan.provenance["min_subjects"] == "passthrough"

    def test_leftover_text_becomes_soft_terms(self):
        plan = compile_query("EEG motor imagery at least 40 subjects")
        assert "motor" in plan.soft_terms
        assert "imagery" in plan.soft_terms
        # the quantitative + modality spans should have been consumed, not soft
        assert "40" not in plan.soft_terms
        assert "subjects" not in plan.soft_terms

    def test_empty_query_produces_empty_plan(self):
        plan = compile_query(None)
        assert plan.hard == {}
        assert plan.soft_terms == []

    def test_short_abbreviation_does_not_fuzzy_match_modality(self):
        # "MI" must not fuzzy-match the "mri" modality token (regression test
        # for a real bug found while validating against the live catalog).
        plan = compile_query("MI")
        assert "modality" not in plan.hard
        assert plan.soft_terms == ["mi"]


# ── Ontology ─────────────────────────────────────────────────────────────

class TestOntology:
    def test_mri_hierarchy_collapses_to_single_catalog_token(self):
        # Regression: an earlier version expanded "mri" into {anat,func,dwi,
        # fmap}, which don't exist as separate tokens in the catalog's
        # dataset-level `modalities` field (only in deep per-file summaries).
        assert canonical_modalities("fmri") == {"mri"}
        assert canonical_modalities("resting-state") == {"mri"}
        assert canonical_modalities("dwi") == {"mri"}

    def test_eeg_synonyms(self):
        assert canonical_modalities("electroencephalography") == {"eeg"}

    def test_unknown_term_returns_empty(self):
        assert canonical_modalities("xyzzy_not_a_modality") == set()

    def test_synonym_clustering_does_not_chain(self):
        # The exact vocabulary shape that caused single-linkage/union-find
        # chaining in an earlier version: short numbered variants bridging to
        # an unrelated cluster.
        vocab = ["mot1", "mot2", "mot3", "mot4", "motor", "n-back", "nback", "resting state", "rest"]
        clusters = mine_synonym_clusters(vocab, threshold=90.0)
        cluster_sizes: dict[int, int] = {}
        for cid in clusters.values():
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
        # no single cluster should have swallowed the whole vocabulary
        assert max(cluster_sizes.values()) < len(vocab)
        # "n-back"/"nback" are close enough to cluster together...
        assert clusters["n-back"] == clusters["nback"]
        # ...but must not be pulled into the "mot*" cluster
        assert clusters["n-back"] != clusters["mot1"]

    def test_mine_from_rows_uses_tasks_not_keywords(self):
        ontology = Ontology.default()
        rows = [
            {"tasks": ["motorimagery"], "keywords": ["some", "totally", "unrelated", "noise", "tokens"]},
            {"tasks": ["motor-imagery"], "keywords": []},
        ]
        ontology.mine_from_rows(rows)
        assert "motorimagery" in ontology.task_synonyms("motor-imagery") or \
               "motor-imagery" in ontology.task_synonyms("motorimagery")


# ── Reciprocal rank fusion ──────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_agreement_across_retrievers_outranks_single_retriever_hit(self):
        ranked = {
            "lexical": [("a", 10.0), ("b", 5.0)],
            "semantic": [("b", 0.9), ("a", 0.1)],
        }
        fused = reciprocal_rank_fusion(ranked)
        # "a" is rank 1 lexical + rank 2 semantic; "b" is rank 2 lexical + rank
        # 1 semantic -> symmetric, should tie.
        assert fused[0].fused_score == pytest.approx(fused[1].fused_score)

    def test_document_only_one_retriever_still_included(self):
        ranked = {"lexical": [("only_here", 1.0)]}
        fused = reciprocal_rank_fusion(ranked)
        assert fused[0].dataset_id == "only_here"

    def test_weights_shift_ranking(self):
        ranked = {
            "lexical": [("x", 1.0), ("y", 1.0)],
            "semantic": [("y", 1.0), ("x", 1.0)],
        }
        fused = reciprocal_rank_fusion(ranked, weights={"lexical": 5.0, "semantic": 0.1})
        assert fused[0].dataset_id == "x"  # ranked 1st by the heavily-weighted retriever

    def test_zero_weight_excludes_retriever(self):
        ranked = {"lexical": [("a", 1.0)], "semantic": [("b", 1.0)]}
        fused = reciprocal_rank_fusion(ranked, weights={"semantic": 0.0})
        ids = {r.dataset_id for r in fused}
        assert ids == {"a"}


# ── Evidence partitioning ──────────────────────────────────────────────────

class TestEvidencePartition:
    def test_null_is_unknown_not_false(self):
        rows = [
            {"dataset_id": "confirmed_present", "has_events": True},
            {"dataset_id": "confirmed_absent", "has_events": False},
            {"dataset_id": "never_scanned", "has_events": None},
        ]
        partition = partition_has_events(rows)
        assert partition.state_of("confirmed_present") == EvidenceState.inferred
        assert partition.state_of("confirmed_absent") == EvidenceState.missing
        assert partition.state_of("never_scanned") == EvidenceState.unknown

    def test_admissible_default_keeps_unknown(self):
        rows = [{"dataset_id": "a", "has_events": None}, {"dataset_id": "b", "has_events": False}]
        partition = partition_has_events(rows)
        admissible = partition.admissible()
        assert "a" in admissible  # unknown kept by default
        assert "b" not in admissible  # confirmed-missing excluded

    def test_strict_mode_drops_unknown(self):
        rows = [{"dataset_id": "a", "has_events": None}]
        partition = partition_has_events(rows)
        assert partition.admissible(include_unknown=False) == set()


# ── Negative-space reporting ─────────────────────────────────────────────

class TestNegativeSpace:
    def test_reports_rejection_reason(self):
        from qortex.search.compiler import Constraint

        candidates = [
            {"dataset_id": "a", "n_subjects": 5},
            {"dataset_id": "b", "n_subjects": 50},
        ]
        report = build_negative_space_report(
            all_candidates=candidates,
            admitted_ids={"b"},
            hard_constraints={"min_subjects": Constraint("min_subjects", "ge", 20)},
        )
        assert report.n_in_scope == 2
        assert report.n_admitted == 1
        assert report.n_rejected == 1
        assert any("min_subjects" in reason for reason in report.rejection_reasons)


# ── End-to-end SearchEngine (synthetic catalog, no network) ────────────────

@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> Path:
    db_path = tmp_path / "catalog.duckdb"
    index = CatalogIndex(db_path)
    index.upsert_many([
        {
            "dataset_id": "ds000001",
            "name": "Motor Imagery EEG Study",
            "description": "A cued left/right hand motor imagery paradigm for BCI research.",
            "authors": ["A. Researcher"],
            "modalities": ["eeg"],
            "tasks": ["motorimagery"],
            "keywords": ["bci", "motor"],
            "n_subjects": 52,
            "license": "CC0",
            "total_bytes": int(2e9),
            "has_events": True,
        },
        {
            "dataset_id": "ds000002",
            "name": "Resting State fMRI Cohort",
            "description": "Eyes-closed resting-state functional MRI in healthy adults.",
            "authors": ["B. Scientist"],
            "modalities": ["mri"],
            "tasks": ["rest"],
            "keywords": ["resting-state"],
            "n_subjects": 8,
            "license": "CC-BY-4.0",
            "total_bytes": int(40e9),
            "has_events": False,
        },
        {
            "dataset_id": "ds000003",
            "name": "Sleep EEG Archive",
            "description": "Overnight polysomnography with EEG for sleep-stage classification.",
            "authors": ["C. Author"],
            "modalities": ["eeg"],
            "tasks": ["sleep"],
            "keywords": ["polysomnography"],
            "n_subjects": 30,
            "license": "proprietary",
            "total_bytes": int(10e9),
            "has_events": None,
        },
    ])
    index.close()
    return db_path


class TestSearchEngineEndToEnd:
    def test_lexical_hit_ranks_exact_title_match_first(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search("motor imagery", limit=3)
        assert resp.results
        assert resp.results[0].dataset_id == "ds000001"

    def test_min_subjects_hard_filter_excludes_small_dataset(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search("at least 20 subjects", limit=10)
        ids = {r.dataset_id for r in resp.results}
        assert "ds000002" not in ids  # only 8 subjects
        assert "ds000001" in ids
        assert "ds000003" in ids

    def test_modality_filter_via_ontology(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search("resting state fmri", limit=10)
        ids = {r.dataset_id for r in resp.results}
        assert ids <= {"ds000002"}

    def test_license_filter(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search("open license", limit=10)
        ids = {r.dataset_id for r in resp.results}
        assert "ds000003" not in ids  # proprietary license

    def test_evidence_flags_distinguish_unknown_from_confirmed_absent(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search(None, limit=10)
        by_id = {r.dataset_id: r for r in resp.results}
        assert by_id["ds000001"].evidence_flags["has_events"] == "inferred"
        assert by_id["ds000002"].evidence_flags["has_events"] == "missing"
        assert by_id["ds000003"].evidence_flags["has_events"] == "unknown"

    def test_negative_space_report_present(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        engine.refresh_indexes()
        resp = engine.search("at least 40 subjects", limit=10)
        assert resp.negative_space is not None
        assert resp.negative_space.n_in_scope == 3

    def test_index_refresh_is_idempotent_on_unchanged_catalog(self, synthetic_catalog):
        from qortex.search.engine import SearchEngine

        engine = SearchEngine(synthetic_catalog)
        first = engine.refresh_indexes()
        second = engine.refresh_indexes()
        assert first["n_datasets"] == second["n_datasets"] == 3
