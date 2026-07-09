"""P0 Braindecode EEG entries (design spec section 12.3), excluding
braindecode.EEGNet which was seeded in Phase 1 (zoo/seed_examples.py).

All 11 entries here are registered architecture-only
(qortex_status="architecture_available") — Braindecode's own docs state
several of these have HF Hub pretrained checkpoints, but the exact HF repo
IDs are not confirmable offline in this environment. Registering a
"pretrained" variant with a guessed repo id would violate the "no
fabricated contracts" invariant, so pretrained entries are deferred until
a real, confirmed checkpoint id is available — not silently dropped.

No entry here carries a fabricated n_channels/sampling_rate_hz/n_classes:
the design spec's own text gives no numeric facts for any of these 11
models (unlike Phase 2's MONAI entries, where a few had spec-confirmed
counts). Only LaBraM and REVE get a pretraining-scale fact, and it's
recorded in notes, never coerced into a tensor-contract field.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType

_DOCS_BASE = "https://braindecode.org/stable/generated/braindecode.models."


def _doc_url(class_name: str) -> str:
    return f"{_DOCS_BASE}{class_name}.html"


def _unknown_eeg_input() -> InputContract:
    return InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        evidence_status=EvidenceStatus.unknown,
    )


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _bci_entry(class_name: str, display_name: str, extra_notes: list[str] | None = None, paper_url: str | None = None) -> ZooEntry:
    return ZooEntry(
        id=f"braindecode.{class_name}",
        display_name=display_name,
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url=_doc_url(class_name),
        paper_url=paper_url,
        modality=["eeg"],
        task=["classification", "eeg_decoding"],
        input_contract=_unknown_eeg_input(),
        output_contract=None,
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
        notes=extra_notes or [],
    )


def _sleep_entry(class_name: str, display_name: str) -> ZooEntry:
    return ZooEntry(
        id=f"braindecode.{class_name}",
        display_name=display_name,
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url=_doc_url(class_name),
        modality=["eeg"],
        task=["classification", "sleep_staging"],
        input_contract=_unknown_eeg_input(),
        output_contract=None,
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def register_all() -> None:
    register(_bci_entry("Deep4Net", "Deep4Net"))
    register(_bci_entry("ShallowFBCSPNet", "ShallowFBCSPNet"))
    register(_bci_entry("EEGConformer", "EEGConformer"))
    register(_bci_entry("BENDR", "BENDR"))
    register(_bci_entry("BIOT", "BIOT"))
    register(_bci_entry(
        "Labram", "LaBraM",
        paper_url="https://arxiv.org/abs/2405.18765",
        extra_notes=[
            "LaBraM (arXiv:2405.18765) reports pretraining on approximately "
            "2,500 hours of EEG from around 20 datasets. Pretraining scale "
            "only -- not a tensor contract fact.",
        ],
    ))
    register(_bci_entry(
        "REVE", "REVE",
        paper_url="https://arxiv.org/abs/2510.21585",
        extra_notes=[
            "REVE (arXiv:2510.21585) reports pretraining on over 60,000 "
            "hours of EEG from 92 datasets and 25,000 subjects. Pretraining "
            "scale only -- not a tensor contract fact.",
        ],
    ))
    register(_sleep_entry("USleep", "USleep"))
    register(_sleep_entry("AttnSleep", "AttnSleep"))
    register(_sleep_entry("DeepSleepNet", "DeepSleepNet"))
    register(_bci_entry("SignalJEPA", "SignalJEPA"))


__all__ = ["register_all"]
