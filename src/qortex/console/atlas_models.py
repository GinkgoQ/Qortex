"""Model contracts and public model-zoo data exposed to Qortex Atlas.

Qortex's ``qortex.neuroai`` subsystem ships a *compatibility engine*
(``CompatibilityEngine.check``) and a full contract type system
(``SourceProfile`` / ``ModelProfile`` / ``InputContract``), but — correctly —
no opinion about which specific published models exist. That catalog is a
product decision, not a library concern, so it lives here in the console
layer: real ``ModelProfile`` objects, each a faithful contract for a
well-known EEG/MEG architecture, evaluated by the real, unmodified
``CompatibilityEngine``.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from qortex.neuroai.contracts import ModelProfile
from qortex.neuroai.models.cache import ModelCache


def compatibility_catalog() -> dict[str, ModelProfile]:
    """Build compatibility profiles from the same public zoo contracts.

    Compatibility is available whenever a registry entry declares both input
    and output contracts. This avoids maintaining a second, divergent list of
    hand-authored model identifiers and requirements in the console layer.
    """
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401
    from qortex.neuroai.models.zoo.registry import list_entries

    profiles: dict[str, ModelProfile] = {}
    for entry in list_entries():
        if entry.input_contract is None or entry.output_contract is None:
            continue
        profiles[entry.id] = ModelProfile(
            model_id=entry.id,
            provider=entry.provider,
            task=entry.task[0] if entry.task else None,
            license=entry.license.name,
            trusted=not entry.security.trust_remote_code_required,
            input_contract=entry.input_contract,
            output_contract=entry.output_contract,
        )
    return profiles


MODEL_CATALOG: dict[str, ModelProfile] = compatibility_catalog()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def list_models(*, cache: ModelCache | None = None) -> list[dict[str, Any]]:
    """Return the registered public zoo with live local-runtime evidence.

    Importing the zoo package performs its explicit seed registration. Cache
    metadata comes only from Qortex's persisted model-cache manifest; a model
    is never reported as cached merely because its provider is installed.
    """
    from qortex.neuroai.models import (
        zoo as _zoo,  # noqa: F401 - triggers registry population
    )
    from qortex.neuroai.models.zoo.registry import list_entries
    from qortex.neuroai.models.zoo.status import is_runtime_executable, runtime_status

    model_cache = cache or ModelCache()
    cached = {entry.model_id: entry for entry in model_cache.list_cached()}
    rows: list[dict[str, Any]] = []
    for entry in list_entries():
        cache_entry = cached.get(entry.id)
        external = entry.external_engine_contract
        executable = external.executable if external is not None else None
        rows.append({
            "id": entry.id,
            "display_name": entry.display_name,
            "entry_type": entry.entry_type.value,
            "provider": entry.provider,
            "execution_mode": entry.execution_mode.value,
            "modality": list(entry.modality),
            "task": list(entry.task),
            "runtime_status": runtime_status(entry).value,
            "runtime_executable_claim": is_runtime_executable(entry),
            "executable": executable,
            "executable_available": shutil.which(executable) is not None if executable else None,
            "source_url": entry.source_url,
            "paper_url": entry.paper_url,
            "model_url": entry.model_url,
            "docs_url": entry.docs_url,
            "license": _json_value(entry.license),
            "evidence_status": entry.evidence_status.value,
            "priority": entry.priority,
            "input_contract": _json_value(entry.input_contract),
            "output_contract": _json_value(entry.output_contract),
            "interaction_contract": _json_value(entry.interaction_contract),
            "cached": cache_entry is not None,
            "cache": asdict(cache_entry) if cache_entry is not None else None,
            "compatibility_available": entry.id in MODEL_CATALOG,
        })
    return rows


def runtime_summary(*, cache: ModelCache | None = None) -> dict[str, Any]:
    """Return measured backend and cache state for the Atlas model workspace."""
    from qortex.neuroai.models.zoo._backend import backend_diagnostics

    model_cache = cache or ModelCache()
    cached = model_cache.list_cached()
    return {
        "backends": backend_diagnostics(),
        "cache": {
            "path": str(Path(model_cache.cache_dir).expanduser().resolve()),
            "manifest_path": str(Path(model_cache.manifest_path).expanduser().resolve()),
            "exists": model_cache.manifest_path.exists(),
            "entries": len(cached),
            "size_bytes": model_cache.disk_usage(),
        },
        "offline_available_models": [entry.model_id for entry in cached],
    }
