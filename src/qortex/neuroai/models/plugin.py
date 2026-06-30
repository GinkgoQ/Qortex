"""Custom plugin model adapter.

Loads a user-supplied Python module as a Qortex model plugin.

Security model:
  - Plugin must be a local Python file (no arbitrary URL execution)
  - ``spec.trust_remote_code=True`` must be explicitly set
  - The plugin module must expose a ``QortexPlugin`` class
  - All plugin calls are wrapped in try/except with structured error reporting
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from qortex.neuroai.contracts import (
    InputContract,
    ModelProfile,
    OutputContract,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)

_REQUIRED_METHODS = ("inspect", "required_input", "output_schema", "load", "predict")


class CustomPluginAdapter(ModelAdapter):
    """Adapter that loads a user-supplied model plugin from a local Python file.

    The plugin file must define a class ``QortexPlugin`` that implements the
    same interface as ``ModelAdapter``.  All calls are wrapped in error
    handling so a broken plugin cannot crash the runtime.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="plugin"`` and ``id=<path to .py file>``.
        ``spec.trust_remote_code`` **must** be ``True`` or loading is refused.

    Raises
    ------
    ModelAdapterError
        When ``trust_remote_code`` is not explicitly enabled.
    FileNotFoundError
        When the plugin file does not exist.
    TypeError
        When the plugin file does not expose a valid ``QortexPlugin`` class.
    """

    def __init__(self, spec: ModelSpec) -> None:
        if not spec.trust_remote_code:
            from qortex.core.exceptions import ModelAdapterError
            raise ModelAdapterError(
                "CustomPluginAdapter requires spec.trust_remote_code=True. "
                "Set trust_remote_code=True in your ModelSpec to enable custom plugin loading. "
                "WARNING: Only load plugins from sources you fully trust.",
                model_id=spec.id,
                provider=spec.provider,
            )
        self._spec = spec
        self._plugin_path = Path(spec.id).expanduser().resolve()
        if not self._plugin_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {self._plugin_path}")
        self._plugin_instance = None
        self._module = None

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        self._ensure_plugin_loaded()
        try:
            result = self._plugin_instance.inspect()
            if not isinstance(result, ModelProfile):
                raise TypeError(
                    f"QortexPlugin.inspect() must return ModelProfile, got {type(result)}"
                )
            log.warning(
                "Custom plugin loaded from %s — verify this plugin is trusted.",
                self._plugin_path,
            )
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Plugin {self._plugin_path.name} raised error in inspect(): {exc}"
            ) from exc

    def required_input(self) -> InputContract:
        self._ensure_plugin_loaded()
        try:
            return self._plugin_instance.required_input()
        except Exception as exc:
            raise RuntimeError(
                f"Plugin {self._plugin_path.name} raised error in required_input(): {exc}"
            ) from exc

    def output_schema(self) -> OutputContract:
        self._ensure_plugin_loaded()
        try:
            return self._plugin_instance.output_schema()
        except Exception as exc:
            raise RuntimeError(
                f"Plugin {self._plugin_path.name} raised error in output_schema(): {exc}"
            ) from exc

    def load(self, runtime: RuntimeSpec) -> None:
        self._ensure_plugin_loaded()
        try:
            self._plugin_instance.load(runtime)
            self._loaded = True
            log.info("Plugin %s loaded successfully.", self._plugin_path.name)
        except Exception as exc:
            raise RuntimeError(
                f"Plugin {self._plugin_path.name} raised error in load(): {exc}"
            ) from exc

    def predict(self, batch: Any) -> ModelOutput:
        if self._plugin_instance is None:
            raise RuntimeError("Plugin not loaded — call load() first")
        try:
            result = self._plugin_instance.predict(batch)
            if not isinstance(result, ModelOutput):
                raise TypeError(
                    f"QortexPlugin.predict() must return ModelOutput, got {type(result)}"
                )
            return result
        except Exception as exc:
            raise RuntimeError(
                f"Plugin {self._plugin_path.name} raised error in predict(): {exc}"
            ) from exc

    def unload(self) -> None:
        if self._plugin_instance is not None and hasattr(self._plugin_instance, "unload"):
            try:
                self._plugin_instance.unload()
            except Exception as exc:
                log.warning("Plugin unload() error: %s", exc)
        self._plugin_instance = None
        self._loaded = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_plugin_loaded(self) -> None:
        if self._module is not None and self._plugin_instance is not None:
            return
        if self._module is not None and self._plugin_instance is None:
            self._plugin_instance = self._module.QortexPlugin()
            return

        log.warning(
            "Loading custom plugin: %s — ensure this file is from a trusted source.",
            self._plugin_path,
        )

        spec = importlib.util.spec_from_file_location(
            f"qortex_plugin_{self._plugin_path.stem}",
            str(self._plugin_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {self._plugin_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ImportError(
                f"Plugin {self._plugin_path.name} failed to import: {exc}"
            ) from exc

        if not hasattr(module, "QortexPlugin"):
            raise TypeError(
                f"Plugin {self._plugin_path.name} must define a 'QortexPlugin' class."
            )

        plugin_cls = module.QortexPlugin
        for method in _REQUIRED_METHODS:
            if not hasattr(plugin_cls, method):
                raise TypeError(
                    f"QortexPlugin is missing required method: {method!r}"
                )

        self._module = module
        self._plugin_instance = plugin_cls()
