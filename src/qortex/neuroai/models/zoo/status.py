"""Runtime status helpers for model zoo entries."""

from __future__ import annotations

from enum import Enum

from qortex.neuroai.models.zoo.schema import ZooEntry


class RuntimeStatus(str, Enum):
    architecture_available = "architecture_available"
    checkpoint_unresolved = "checkpoint_unresolved"
    runnable_after_contract_validation = "runnable_after_contract_validation"
    runnable_if_executable_available = "runnable_if_executable_available"
    weights_verified = "weights_verified"
    weights_cached = "weights_cached"
    integration_tested = "integration_tested"
    runtime_verified = "runtime_verified"
    blocked = "blocked"
    unknown = "unknown"


_KNOWN = {item.value: item for item in RuntimeStatus}


def runtime_status(entry: ZooEntry) -> RuntimeStatus:
    """Return the normalized runtime status for a zoo entry."""

    return _KNOWN.get(str(entry.qortex_status), RuntimeStatus.unknown)


def is_runtime_executable(entry: ZooEntry) -> bool:
    """Return whether the entry claims executable runtime support now."""

    return runtime_status(entry) in {
        RuntimeStatus.runnable_after_contract_validation,
        RuntimeStatus.runnable_if_executable_available,
        RuntimeStatus.weights_verified,
        RuntimeStatus.weights_cached,
        RuntimeStatus.integration_tested,
        RuntimeStatus.runtime_verified,
    }


__all__ = ["RuntimeStatus", "runtime_status", "is_runtime_executable"]
