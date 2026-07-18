"""Bridge from the zoo registry into the legacy curated contract registry.

qortex.neuroai.models._contracts.ModelContractEntry predates the zoo
registry and is what suggest-models actually reads. This module is a
one-way, additive sync: it never modifies _contracts.py, only calls its
public register()/lookup() functions, so suggest-models sees zoo entries
without either registry needing to know about the other's internals.

Entries without both an input_contract and output_contract object present
(external engines, generative models with no classification/segmentation
output) are skipped — ModelContractEntry requires both, and
suggest-models's ranking logic assumes output_contract.output_type is
always present.

Note: "has both contracts" means the InputContract/OutputContract objects
exist, not that every field inside them is populated. Many zoo entries
carry contracts with evidence_status=unknown and no confirmed shape data —
those are synced too, deliberately: the existing CompatibilityEngine
already degrades unknown fields to a low-ranked "uncertain" status with
visible unknowns, so suggest-models never presents an unverified entry as
equivalent to a confirmed match. Excluding them here would just hide
legitimate (if uncertain) candidates from suggest-models entirely.
"""

from __future__ import annotations

from qortex.neuroai.models import _contracts
from qortex.neuroai.models.zoo.registry import list_entries


def _legacy_id_candidates(zoo_id: str) -> list[str]:
    """Ids to check for an existing legacy entry describing the same model.

    Zoo entries are namespaced "monai.<bundle_name>"; several bundles were
    already curated in the legacy registry under the bare bundle name
    (e.g. "wholeBody_ct_segmentation") before the zoo package existed.
    Checking both forms avoids registering a duplicate legacy entry for a
    model that's already there under its un-namespaced id.
    """
    candidates = [zoo_id]
    if zoo_id.startswith("monai."):
        candidates.append(zoo_id.split(".", 1)[1])
    return candidates


def sync_into_legacy_registry() -> int:
    """Register every ZooEntry with both contract objects into the legacy registry.

    Idempotent: entries already present in the legacy registry (by id, or
    by the un-namespaced bundle name for "monai.*" ids — see
    _legacy_id_candidates) are skipped, so this is safe to call on every
    suggest-models invocation.

    Returns
    -------
    int
        Number of entries newly registered by this call.
    """
    synced = 0
    for entry in list_entries():
        if entry.input_contract is None or entry.output_contract is None:
            continue
        modalities = tuple(entry.modality or ())
        if not modalities:
            modalities = (str(getattr(entry.input_contract, "modality", "") or "unknown"),)
        for modality in modalities:
            input_contract = _copy_input_contract_with_modality(entry.input_contract, modality)
            if _legacy_entry_exists(entry.id, modality):
                continue
            if modality == str(getattr(entry.input_contract, "modality", "") or ""):
                if any(
                    _contracts.lookup(cid) is not None
                    and _legacy_entry_exists(cid, modality)
                    for cid in _legacy_id_candidates(entry.id)
                ):
                    continue
            _contracts.register(_contracts.ModelContractEntry(
                model_id=entry.id,
                provider=entry.provider,
                input_contract=input_contract,
                output_contract=entry.output_contract,
                estimated_memory_mb=None,
                notes=entry.display_name,
            ))
            synced += 1
    return synced


def _legacy_entry_exists(model_id: str, modality: str) -> bool:
    model_key = model_id.lower()
    modality_key = modality.lower()
    return any(
        existing.model_id.lower() == model_key
        and str(getattr(existing.input_contract, "modality", "") or "").lower() == modality_key
        for existing in _contracts.list_entries()
    )


def _copy_input_contract_with_modality(input_contract, modality: str):
    if hasattr(input_contract, "model_copy"):
        return input_contract.model_copy(update={"modality": modality})
    data = input_contract.model_dump() if hasattr(input_contract, "model_dump") else dict(input_contract.__dict__)
    data["modality"] = modality
    return type(input_contract)(**data)


__all__ = ["sync_into_legacy_registry"]
