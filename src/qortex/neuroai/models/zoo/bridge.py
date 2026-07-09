"""Bridge from the zoo registry into the legacy curated contract registry.

qortex.neuroai.models._contracts.ModelContractEntry predates the zoo
registry and is what suggest-models actually reads. This module is a
one-way, additive sync: it never modifies _contracts.py, only calls its
public register()/lookup() functions, so suggest-models sees zoo entries
without either registry needing to know about the other's internals.

Entries without both an input_contract and output_contract (external
engines, generative models with no classification/segmentation output) are
skipped — ModelContractEntry requires both, and suggest-models's ranking
logic assumes output_contract.output_type is always present.
"""

from __future__ import annotations

from qortex.neuroai.models import _contracts
from qortex.neuroai.models.zoo.registry import list_entries


def sync_into_legacy_registry() -> int:
    """Register every fully-contracted ZooEntry into the legacy registry.

    Idempotent: entries already present in the legacy registry (by id) are
    skipped, so this is safe to call on every suggest-models invocation.

    Returns
    -------
    int
        Number of entries newly registered by this call.
    """
    synced = 0
    for entry in list_entries():
        if entry.input_contract is None or entry.output_contract is None:
            continue
        if _contracts.lookup(entry.id) is not None:
            continue
        _contracts.register(_contracts.ModelContractEntry(
            model_id=entry.id,
            provider=entry.provider,
            input_contract=entry.input_contract,
            output_contract=entry.output_contract,
            estimated_memory_mb=None,
            notes=entry.display_name,
        ))
        synced += 1
    return synced


__all__ = ["sync_into_legacy_registry"]
