"""Evidence-partitioned filtering — every uncertain constraint resolves to one
of Qortex's existing ``EvidenceState`` values (``qortex.checks.EvidenceState``)
instead of being silently coerced to a boolean pass/fail.

This directly fixes the defect in the original ``CatalogIndex.search()``:
``has_events = ?`` in SQL treats "never checked" (NULL) and "confirmed absent"
(0) identically — both fail a ``has_events=True`` filter, so a dataset simply
missing a manifest scan looks indistinguishable from one that's genuinely
event-less. A three-valued partition keeps those separate and lets the caller
decide how to treat "unknown" (default: keep and flag, never silently drop).

Reusing ``EvidenceState`` (rather than inventing a parallel confirmed/inferred/
unknown vocabulary, which the original search-engine design doc proposed
before this codebase's own ``qortex.checks`` module was found to already have
a 7-state evidence model) keeps this consistent with the vocabulary
``qortex.checks`` and Atlas's ``console/atlas_evidence.py`` already use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qortex.checks import EvidenceState


@dataclass
class EvidencePartition:
    field_name: str
    by_state: dict[EvidenceState, list[str]] = field(default_factory=dict)

    def ids(self, *states: EvidenceState) -> list[str]:
        out: list[str] = []
        for s in states:
            out.extend(self.by_state.get(s, []))
        return out

    def state_of(self, dataset_id: str) -> EvidenceState:
        for state, ids in self.by_state.items():
            if dataset_id in ids:
                return state
        return EvidenceState.unknown

    def admissible(
        self,
        *,
        include_unknown: bool = True,
        include_inferred: bool = True,
    ) -> set[str]:
        states = [EvidenceState.confirmed]
        if include_inferred:
            states.append(EvidenceState.inferred)
        if include_unknown:
            states.append(EvidenceState.unknown)
        return set(self.ids(*states))


def partition_has_events(rows: list[dict[str, Any]]) -> EvidencePartition:
    """``has_events`` in the catalog is populated by a Level-1 manifest scan —
    it records file *presence* (an ``events.tsv`` companion exists somewhere in
    the tree), not confirmed trial-type content. So:

    - ``True``  -> ``inferred`` (a file exists; whether it has usable labels
      needs a Level-2+ content read — see ``LabelLandscapeAnalyzer``)
    - ``False`` -> ``missing`` (no event file anywhere in the manifest — a
      real, confirmed structural fact, not a guess)
    - ``NULL``  -> ``unknown`` (the manifest itself was never fetched for this
      dataset — the catalog literally does not know)

    ``confirmed`` is reserved for a future Level-2 wiring where trial-type
    columns have actually been read (``LabelLandscape.label_column`` present)
    — intentionally left empty here rather than faked.
    """
    partition = EvidencePartition(field_name="has_events")
    for row in rows:
        value = row.get("has_events")
        dataset_id = row["dataset_id"]
        if value is None:
            state = EvidenceState.unknown
        elif value:
            state = EvidenceState.inferred
        else:
            state = EvidenceState.missing
        partition.by_state.setdefault(state, []).append(dataset_id)
    return partition


def partition_has_derivatives(rows: list[dict[str, Any]]) -> EvidencePartition:
    """Same three-valued logic as ``partition_has_events``, for derivatives."""
    partition = EvidencePartition(field_name="has_derivatives")
    for row in rows:
        value = row.get("has_derivatives")
        dataset_id = row["dataset_id"]
        if value is None:
            state = EvidenceState.unknown
        elif value:
            state = EvidenceState.inferred
        else:
            state = EvidenceState.missing
        partition.by_state.setdefault(state, []).append(dataset_id)
    return partition
