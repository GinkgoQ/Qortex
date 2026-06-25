"""Loader plugin registry — thread-safe, priority-ordered, instance-based.

Built-in loaders are registered at import time; third-party loaders are
discovered via ``entry_points["qortex.loaders"]``.

Usage
-----
    registry = LoaderRegistry()
    registry.discover()          # idempotent; safe to call many times
    loader = registry.resolve(file_record)
    if loader is not None:
        record = loader.load(file_record, local_path)
        for sample in loader.to_sample_records(record):
            ...
"""

from __future__ import annotations

import threading
import warnings
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from qortex.core.entities import FileRecord
from qortex.core.exceptions import LoaderNotFoundError

if TYPE_CHECKING:
    from qortex.parse._base import BaseLoader


# Priority order for built-in loaders (higher index = checked first in resolve)
_BUILTIN_MODALITY_PRIORITY: list[str] = [
    "behavior",
    "pet",
    "dwi",
    "mri",
    "fmri",
    "fnirs",
    "ieeg",
    "meg",
    "eeg",
]

_BUILTIN_LOADERS: tuple[tuple[str, str], ...] = (
    ("qortex.parse.behavior", "BehaviorLoader"),
    ("qortex.parse.pet", "PETLoader"),
    ("qortex.parse.dwi", "DWILoader"),
    ("qortex.parse.mri", "MRILoader"),
    ("qortex.parse.fmri", "FMRILoader"),
    ("qortex.parse.fnirs", "FNIRSLoader"),
    ("qortex.parse.ieeg", "IEEGLoader"),
    ("qortex.parse.meg", "MEGLoader"),
    ("qortex.parse.eeg", "EEGLoader"),
)


class LoaderRegistry:
    """Per-instance registry of modality loaders.

    Use ``LoaderRegistry()`` then ``discover()`` to auto-register everything.
    The class-level singleton ``_global`` is used by ConversionPipeline when
    no explicit registry is passed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaders: dict[str, BaseLoader] = {}
        self._discovered = False

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, loader: "BaseLoader") -> None:
        with self._lock:
            self._loaders[loader.modality] = loader

    def register_many(self, loaders: list["BaseLoader"]) -> None:
        with self._lock:
            for loader in loaders:
                self._loaders[loader.modality] = loader

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self, force: bool = False) -> None:
        """Register built-in loaders and all ``qortex.loaders`` entry-points.

        Idempotent unless *force=True*.  Thread-safe.
        """
        with self._lock:
            if self._discovered and not force:
                return
            self._register_builtins()
            self._load_entry_points()
            self._discovered = True

    def _register_builtins(self) -> None:
        for module_name, class_name in _BUILTIN_LOADERS:
            try:
                module = import_module(module_name)
                cls = getattr(module, class_name)
                instance = cls()
            except Exception as exc:
                warnings.warn(
                    f"Skipped built-in qortex loader {class_name}: {exc}",
                    RuntimeWarning,
                    stacklevel=4,
                )
                continue
            self._loaders[instance.modality] = instance

    def _load_entry_points(self) -> None:
        eps = entry_points(group="qortex.loaders")
        for ep in eps:
            try:
                loader_cls = ep.load()
                instance = loader_cls()
                self._loaders[instance.modality] = instance
            except Exception as exc:
                warnings.warn(
                    f"Failed to load qortex loader plugin {ep.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=4,
                )

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, modality: str) -> "BaseLoader":
        self.discover()
        with self._lock:
            loader = self._loaders.get(modality)
        if loader is None:
            raise LoaderNotFoundError(modality)
        return loader

    def resolve(self, file: FileRecord) -> "BaseLoader | None":
        """Return the best-matched loader for *file*, respecting priority order.

        Loaders are checked in *_BUILTIN_MODALITY_PRIORITY* order (highest
        specificity first), then any remaining registered loaders.
        """
        self.discover()
        with self._lock:
            loaders = dict(self._loaders)

        # Check in priority order first
        for modality in reversed(_BUILTIN_MODALITY_PRIORITY):
            loader = loaders.get(modality)
            if loader is not None and loader.can_load(file):
                return loader

        # Then any third-party loaders not in the priority list
        for modality, loader in loaders.items():
            if modality not in _BUILTIN_MODALITY_PRIORITY and loader.can_load(file):
                return loader

        return None

    def available(self) -> list[str]:
        self.discover()
        with self._lock:
            return list(self._loaders)
