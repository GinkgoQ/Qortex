"""Real backend-availability checks for the model registry.

No fabricated status: each entry is a genuine ``importlib`` probe of the
optional dependency a provider needs, run at call time.
"""

from __future__ import annotations

import importlib

_BACKEND_MODULES: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "keras": "tensorflow",
    "huggingface": "transformers",
    "monai": "monai",
    "onnxruntime": "onnxruntime",
    "braindecode": "braindecode",
    "ultralytics": "ultralytics",
}


def backend_diagnostics() -> dict[str, dict[str, str | bool | None]]:
    """Probe optional runtimes and retain import-failure diagnostics.

    Optional scientific runtimes can be installed yet unusable because a
    compiled dependency or version constraint is broken. Those failures are
    runtime unavailability, not API failures, so every ordinary exception is
    captured and made inspectable by callers.
    """
    status: dict[str, dict[str, str | bool | None]] = {}
    for label, module_name in _BACKEND_MODULES.items():
        try:
            importlib.import_module(module_name)
            status[label] = {"available": True, "module": module_name, "error": None}
        except Exception as exc:  # noqa: BLE001 - a broken optional runtime is unavailable
            status[label] = {
                "available": False,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return status


def backend_availability() -> dict[str, bool]:
    """Return ``{backend_name: is_importable}`` for every supported provider."""
    return {name: bool(result["available"]) for name, result in backend_diagnostics().items()}


__all__ = ["backend_availability", "backend_diagnostics"]
