"""Neuro-domain ontology: canonical modality hierarchy/synonyms + corpus-mined
task-label synonym clusters.

Two distinct sources of truth, deliberately kept apart:

1. Modality hierarchy/synonyms (``MODALITY_HIERARCHY`` / ``MODALITY_SYNONYMS``)
   is curated from the BIDS specification — it is small, stable, and the same
   for every OpenNeuro dataset, so hand-curation is the right call.
2. Task-label synonym clusters are NOT hand-curated. They are computed from
   this catalog's *own* observed ``tasks`` vocabulary via connected-components
   clustering over a fuzzy-string-similarity graph (``mine_synonym_clusters``).
   OpenNeuro task labels are free text chosen per-lab ("motorimagery",
   "motor-imagery-task", "MotorImageryLR") — a generic thesaurus would miss
   most of these; grounding the map in the corpus's real strings is the only
   approach that scales without manual upkeep as new datasets are indexed.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, process

# ── Canonical modality hierarchy ────────────────────────────────────────────
# Deliberately NOT a BIDS-datatype hierarchy (anat/func/dwi/fmap as separate
# leaves) — verified against the actual local catalog: OpenNeuro's GraphQL
# `metadata.modalities` field (what CatalogIndex ingests at Level 0/1, before
# any per-file manifest scan) reports one flat "mri" token for T1w/BOLD/DWI/
# fieldmap datasets alike; fine-grained BIDS datatypes only appear later, in
# per-file `dataset_file_summaries` rows from a *deep* refresh, which most
# indexed datasets never receive. An ontology that expanded "mri" into
# {anat, func, dwi, fmap} would silently return zero structural candidates for
# every "mri"/"fmri"/"resting state" query against a Level-0/1 catalog — this
# was caught by running real queries against the real local catalog (293
# datasets), not assumed from the BIDS spec. "meeg" is kept as a genuine
# grouping since eeg/meg/ieeg *are* indexed as distinct tokens.
MODALITY_HIERARCHY: dict[str, frozenset[str]] = {
    "meeg": frozenset({"eeg", "meg", "ieeg"}),
}

# canonical (as actually stored in dataset_modalities) -> observed surface forms.
MODALITY_SYNONYMS: dict[str, frozenset[str]] = {
    "mri": frozenset({
        "mri", "anat", "t1", "t1w", "t2", "t2w", "anatomical", "structural", "mprage",
        "func", "bold", "fmri", "functional", "functional-mri", "rest", "resting-state",
        "rs-fmri", "task-fmri", "dwi", "dti", "diffusion", "diffusion-weighted",
        "tractography", "fmap", "fieldmap", "field-map", "perf", "asl", "perfusion",
    }),
    "eeg": frozenset({"eeg", "electroencephalography", "electroencephalogram"}),
    "meg": frozenset({"meg", "magnetoencephalography"}),
    "ieeg": frozenset({"ieeg", "ecog", "seeg", "intracranial", "electrocorticography", "stereo-eeg"}),
    "nirs": frozenset({"nirs", "fnirs", "near-infrared", "near-infrared-spectroscopy"}),
    "pet": frozenset({"pet", "positron-emission-tomography"}),
    "mrs": frozenset({"mrs", "spectroscopy", "magnetic-resonance-spectroscopy"}),
    "beh": frozenset({"beh", "behavioral", "behavioural"}),
    "motion": frozenset({"motion", "mocap", "motion-capture"}),
    "micr": frozenset({"micr", "microscopy"}),
}

_SURFACE_TO_CANONICAL: dict[str, str] = {
    surface: canon for canon, surfaces in MODALITY_SYNONYMS.items() for surface in surfaces
}

_FUZZY_SCORE_CUTOFF = 82.0
_MAX_TASK_VOCAB_FOR_MINING = 4000  # O(n^2) cdist guard — see mine_synonym_clusters


def _normalize(term: str) -> str:
    return " ".join(term.strip().lower().replace("_", " ").split()).replace(" ", "-")


def canonical_modalities(term: str) -> set[str]:
    """Expand a free-text modality mention to the set of canonical BIDS
    datatype tokens it should match, including hierarchy expansion
    (e.g. "mri" -> {anat, func, dwi, fmap, perf})."""
    t = _normalize(term)
    if not t:
        return set()
    if t in MODALITY_HIERARCHY:
        out: set[str] = set()
        for child in MODALITY_HIERARCHY[t]:
            out |= canonical_modalities(child)
        return out
    if t in _SURFACE_TO_CANONICAL:
        return {_SURFACE_TO_CANONICAL[t]}
    if t in MODALITY_SYNONYMS:
        return {t}
    # Fuzzy fallback only for tokens long enough that similarity is meaningful.
    # Short tokens (<=3 chars, e.g. "MI" for "motor imagery") produce false
    # positives against equally-short surface forms like "mri" under WRatio —
    # caught by testing "MI" against the real catalog, which fuzzy-matched it
    # to the MRI modality instead of leaving it as a free-text soft term.
    if len(t) <= 3:
        return set()
    match = process.extractOne(
        t, _SURFACE_TO_CANONICAL.keys(), scorer=fuzz.WRatio, score_cutoff=_FUZZY_SCORE_CUTOFF
    )
    if match:
        return {_SURFACE_TO_CANONICAL[match[0]]}
    return set()


def mine_synonym_clusters(vocabulary: list[str], *, threshold: float = 90.0) -> dict[str, int]:
    """Cluster a corpus vocabulary into synonym groups via **complete-linkage**
    hierarchical clustering over a fuzzy-similarity distance matrix.

    Complete linkage (not single-linkage/union-find) is a deliberate choice:
    an earlier version of this function used union-find over a thresholded
    pairwise-similarity graph, and running it against the real local catalog
    surfaced a textbook *chaining* failure — short alphanumeric task tokens
    ("mot1", "mot2", "mot3", "mot4", "motor", ...) formed a similarity chain
    where each adjacent pair was similar enough to merge, but the resulting
    cluster ended up containing ~300 largely unrelated task labels
    transitively, poisoning query expansion with noise. Complete linkage fixes
    this structurally: a cluster only forms when *every pair* of its members
    is within ``threshold``, i.e. it merges by worst-case (maximum) pairwise
    distance, not by a chain of nearest neighbors — the standard remedy for
    single-linkage chaining in agglomerative clustering.

    ``rapidfuzz.process.cdist`` is a vectorized C implementation, so building
    the distance matrix is fast for realistic per-corpus task-label
    vocabularies (hundreds to low thousands of unique strings); a hard cap
    guards against O(n^2) blowup if called with an unfiltered, much larger
    vocabulary.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    terms = sorted({v.strip().lower() for v in vocabulary if v and v.strip()})
    n = len(terms)
    if n == 0:
        return {}
    if n > _MAX_TASK_VOCAB_FOR_MINING:
        terms = terms[:_MAX_TASK_VOCAB_FOR_MINING]
        n = len(terms)
    if n == 1:
        return {terms[0]: 0}

    sim = process.cdist(terms, terms, scorer=fuzz.WRatio, dtype=np.float64)
    dist = 100.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, dist.T)  # cdist(x, x) is exactly symmetric modulo fp noise; guard fcluster's assumption
    condensed = squareform(dist, checks=False)

    linkage_matrix = linkage(condensed, method="complete")
    labels = fcluster(linkage_matrix, t=100.0 - threshold, criterion="distance")
    return {terms[i]: int(labels[i]) for i in range(n)}


class Ontology:
    """Bundles the fixed modality ontology with a corpus-mined task-synonym
    map. One instance per catalog snapshot; call ``mine_from_rows`` whenever
    the catalog is refreshed (cheap — see ``mine_synonym_clusters``)."""

    def __init__(self) -> None:
        self._task_cluster_of: dict[str, int] = {}
        self._cluster_members: dict[int, set[str]] = {}

    @classmethod
    def default(cls) -> "Ontology":
        return cls()

    def canonical_modalities(self, term: str) -> set[str]:
        return canonical_modalities(term)

    def mine_from_rows(self, rows: list[dict[str, Any]]) -> None:
        """Rebuild task-synonym clusters from this catalog's actual ``tasks``
        vocabulary (not ``keywords`` — those are auto-derived, noisy free
        tokens, not curated labels; see ``catalog/index.py:_derive_keywords``)."""
        vocab: set[str] = set()
        for row in rows:
            vocab.update(t for t in (row.get("tasks") or []) if t)
        clusters = mine_synonym_clusters(sorted(vocab))
        self._task_cluster_of = clusters
        members: dict[int, set[str]] = {}
        for term, cluster_id in clusters.items():
            members.setdefault(cluster_id, set()).add(term)
        self._cluster_members = members

    def task_synonyms(self, term: str) -> set[str]:
        """Other corpus task-label strings judged to be the same paradigm as
        ``term`` (empty set if unmined or no cluster-mates)."""
        t = term.strip().lower()
        cluster_id = self._task_cluster_of.get(t)
        if cluster_id is None:
            return set()
        return self._cluster_members.get(cluster_id, set()) - {t}
