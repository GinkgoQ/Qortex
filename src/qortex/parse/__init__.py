"""Parser and loader public API.

Concrete loader modules depend on modality-specific optional packages. Keep
the package import lightweight and load concrete classes on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from qortex.parse._base import BaseLoader
from qortex.parse._registry import LoaderRegistry

__all__ = [
    "BaseLoader",
    "LoaderRegistry",
    "EEGLoader",
    "MEGLoader",
    "IEEGLoader",
    "FNIRSLoader",
    "MRILoader",
    "FMRILoader",
    "DWILoader",
    "PETLoader",
    "BehaviorLoader",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BehaviorLoader": ("qortex.parse.behavior", "BehaviorLoader"),
    "DWILoader": ("qortex.parse.dwi", "DWILoader"),
    "EEGLoader": ("qortex.parse.eeg", "EEGLoader"),
    "FMRILoader": ("qortex.parse.fmri", "FMRILoader"),
    "FNIRSLoader": ("qortex.parse.fnirs", "FNIRSLoader"),
    "IEEGLoader": ("qortex.parse.ieeg", "IEEGLoader"),
    "MEGLoader": ("qortex.parse.meg", "MEGLoader"),
    "MRILoader": ("qortex.parse.mri", "MRILoader"),
    "PETLoader": ("qortex.parse.pet", "PETLoader"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
