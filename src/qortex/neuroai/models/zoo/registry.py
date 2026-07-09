"""In-memory registry for ZooEntry records.

This is the Phase-1 zoo registry, deliberately separate from the legacy
qortex.neuroai.models._contracts registry (13 curated entries). Phase 2
adds the compatibility bridge that lets suggest-models read from both.
"""

from __future__ import annotations

from qortex.neuroai.models.zoo.schema import ZooEntry, ZooEntryType

_REGISTRY: dict[str, ZooEntry] = {}


def register(entry: ZooEntry) -> None:
    if entry.id in _REGISTRY:
        raise ValueError(f"ZooEntry id already registered: {entry.id!r}")
    _REGISTRY[entry.id] = entry


def replace(entry: ZooEntry) -> None:
    """Overwrite an existing entry. Raises ValueError if entry.id is not
    already registered -- use register() to add a genuinely new entry."""
    if entry.id not in _REGISTRY:
        raise ValueError(f"Cannot replace unregistered ZooEntry id: {entry.id!r}")
    _REGISTRY[entry.id] = entry


def lookup(entry_id: str) -> ZooEntry | None:
    return _REGISTRY.get(entry_id)


def list_entries(
    *,
    entry_type: ZooEntryType | str | None = None,
    provider: str | None = None,
    modality: str | None = None,
    task: str | None = None,
    priority: str | None = None,
) -> list[ZooEntry]:
    results = list(_REGISTRY.values())
    if entry_type is not None:
        want = entry_type.value if isinstance(entry_type, ZooEntryType) else str(entry_type)
        results = [e for e in results if e.entry_type.value == want]
    if provider is not None:
        results = [e for e in results if e.provider == provider]
    if modality is not None:
        results = [e for e in results if modality in e.modality]
    if task is not None:
        results = [e for e in results if task in e.task]
    if priority is not None:
        results = [e for e in results if e.priority == priority]
    return sorted(results, key=lambda e: e.id)


def clear_registry() -> None:
    """Test-only: reset registry state between test modules."""
    _REGISTRY.clear()


__all__ = ["register", "replace", "lookup", "list_entries", "clear_registry"]
