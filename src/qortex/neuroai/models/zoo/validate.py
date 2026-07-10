"""Offline self-checks for the zoo registry — no network, no weights.

Run via ``qortex neuroai zoo validate`` or directly in CI to catch a
registry entry that fabricates or omits required contract data before it
ships. See docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 19.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import list_entries
from qortex.neuroai.models.zoo.schema import ZooEntry, ZooEntryType

# Providers that are not dispatched through make_model_adapter — they run
# through neuroai/external.py's own command-builder dispatch instead.
_EXTERNAL_ONLY_PROVIDERS = {"external_cli"}


@dataclass
class ValidationIssue:
    entry_id: str
    severity: str  # "error" | "warning"
    message: str


def _is_well_formed_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return bool(parsed.scheme) and bool(parsed.netloc)


def _check_urls(entry: ZooEntry) -> list[ValidationIssue]:
    issues = []
    for field_name in ("source_url", "paper_url", "model_url", "docs_url"):
        value = getattr(entry, field_name, None)
        if value and not _is_well_formed_url(value):
            issues.append(ValidationIssue(
                entry.id, "error", f"{field_name} is not a well-formed URL: {value!r}",
            ))
    return issues


def _check_license_and_evidence(entry: ZooEntry) -> list[ValidationIssue]:
    issues = []
    if entry.license is None:
        issues.append(ValidationIssue(entry.id, "error", "missing license info"))
    if entry.evidence_status is None:
        issues.append(ValidationIssue(entry.id, "error", "missing evidence_status"))
    return issues


def _check_interaction_contract(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.entry_type not in (ZooEntryType.promptable_model, ZooEntryType.foundation_model):
        return []
    if entry.interaction_contract is None:
        return [ValidationIssue(
            entry.id, "error",
            "promptable/foundation entry missing interaction_contract",
        )]
    if not entry.interaction_contract.supported_prompt_types:
        return [ValidationIssue(
            entry.id, "error",
            "interaction_contract.supported_prompt_types is empty",
        )]
    return []


def _check_external_engine_contract(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.entry_type != ZooEntryType.external_engine:
        return []
    if entry.external_engine_contract is None:
        return [ValidationIssue(
            entry.id, "error",
            "external_engine entry missing external_engine_contract",
        )]
    return []


def _check_provider_dispatch(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.provider in _EXTERNAL_ONLY_PROVIDERS:
        return []
    # Uses the gate-free dispatch resolver deliberately, not
    # make_model_adapter(): this check verifies registry structural
    # validity (does the provider string resolve to a real adapter class),
    # which must not be entangled with license/remote-code gates -- those
    # are a separate, execution-time concern, and an unrelated
    # unknown-license entry must never mask a genuine unknown-provider
    # ValueError here.
    from qortex.neuroai.models._registry import resolve_provider_dispatch
    from qortex.neuroai.spec import ModelSpec

    try:
        resolve_provider_dispatch(ModelSpec(provider=entry.provider, id=entry.id))
    except ImportError:
        return []  # optional dependency missing — not a registry defect
    except ValueError:
        return [ValidationIssue(
            entry.id, "error", f"provider {entry.provider!r} has no adapter dispatch",
        )]
    except Exception:
        # Constructing the adapter touched something else offline (e.g. a
        # missing local path) — that's a runtime concern, not a registry
        # validity concern.
        return []
    return []


def _check_eeg_contract_consistency(entry: ZooEntry) -> list[ValidationIssue]:
    ic = entry.input_contract
    if ic is None or str(ic.modality).lower() != "eeg":
        return []
    if entry.evidence_status != EvidenceStatus.confirmed:
        return []
    has_confirmed_shape = ic.n_channels is not None or ic.sampling_rate_hz is not None
    if has_confirmed_shape:
        return []
    return [ValidationIssue(
        entry.id, "warning",
        "entry.evidence_status=confirmed but input_contract has no "
        "confirmed n_channels or sampling_rate_hz -- confirm this is "
        "intentional (metadata confirmed, tensor shape architecture-only)",
    )]


def validate_registry() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for entry in list_entries():
        issues.extend(_check_urls(entry))
        issues.extend(_check_license_and_evidence(entry))
        issues.extend(_check_interaction_contract(entry))
        issues.extend(_check_external_engine_contract(entry))
        issues.extend(_check_provider_dispatch(entry))
        issues.extend(_check_eeg_contract_consistency(entry))
    return issues


__all__ = ["ValidationIssue", "validate_registry"]
