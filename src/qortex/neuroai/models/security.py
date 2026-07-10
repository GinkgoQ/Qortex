"""Remote-code and executable allowlist gates for NeuroAI zoo entries."""

from __future__ import annotations

from pathlib import Path

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.zoo.schema import ZooEntry


def check_remote_code_gate(entry: ZooEntry, *, allow_remote_code: bool = False) -> None:
    """Block entries that require remote code unless explicitly allowed."""

    if not entry.security.trust_remote_code_required:
        return
    if allow_remote_code or entry.security.allow_remote_code:
        return
    raise ModelAdapterError(
        f"{entry.id} requires remote Python code execution. "
        "Pass --allow-remote-code only in a trusted environment."
    )


def check_executable_allowlist(entry: ZooEntry, resolved_executable_path: str) -> None:
    """Verify a resolved executable path against the entry's declared allowlist."""

    allowed = entry.security.executable_names
    if not allowed:
        return
    basename = Path(resolved_executable_path).name
    if basename not in allowed:
        raise ModelAdapterError(
            f"{entry.id}'s security policy allows executables {allowed!r}, "
            f"but resolved executable is {basename!r}."
        )


__all__ = ["check_remote_code_gate", "check_executable_allowlist"]
