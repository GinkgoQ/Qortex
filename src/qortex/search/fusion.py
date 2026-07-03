"""Reciprocal Rank Fusion (RRF) — combines multiple ranked retriever outputs
into one ranking without ever needing BM25 scores and cosine similarities to
be on a comparable scale (they are not, and coercing them to be — e.g. min-max
normalizing each independently — is a common and fragile mistake: it makes the
fused score sensitive to the score *distribution* of whichever retriever
happened to return the fewest/noisiest candidates that query).

    score(d) = sum over retrievers r that returned d of  w_r / (k + rank_r(d))

``k=60`` is the constant from the original RRF paper (Cormack, Clarke &
Buettcher, SIGIR 2009) — high enough that the fused ranking isn't dominated by
whichever retriever happens to rank one document 1st vs. 2nd, low enough that
being in the top few results of any single retriever still matters a lot. It
is a scale-free, rank-only fusion — the only per-retriever signal that
crosses into the fused score is ordinal position, which is exactly the one
thing every retriever type (lexical, semantic, structural, graph) can produce
on a common footing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_K = 60.0


@dataclass
class FusedResult:
    dataset_id: str
    fused_score: float = 0.0
    # retriever name -> (1-based rank in that retriever's list, raw score)
    contributions: dict[str, tuple[int, float]] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]],
    *,
    weights: dict[str, float] | None = None,
    k: float = _K,
) -> list[FusedResult]:
    """``ranked_lists``: {retriever_name: [(dataset_id, raw_score), ...]},
    each list best-first. ``weights``: per-retriever multiplier, default 1.0.
    Returns all documents seen by >=1 retriever, sorted best-first."""
    weights = weights or {}
    fused: dict[str, FusedResult] = {}
    for retriever, results in ranked_lists.items():
        w = weights.get(retriever, 1.0)
        if w <= 0:
            continue
        for rank, (dataset_id, raw_score) in enumerate(results, start=1):
            entry = fused.setdefault(dataset_id, FusedResult(dataset_id=dataset_id))
            entry.fused_score += w / (k + rank)
            entry.contributions[retriever] = (rank, raw_score)
    return sorted(fused.values(), key=lambda r: r.fused_score, reverse=True)
