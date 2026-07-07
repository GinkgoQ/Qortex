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


def backend_availability() -> dict[str, bool]:
    """Return ``{backend_name: is_importable}`` for every supported provider."""
    status: dict[str, bool] = {}
    for label, module_name in _BACKEND_MODULES.items():
        try:
            importlib.import_module(module_name)
            status[label] = True
        except ImportError:
            status[label] = False
    return status


__all__ = ["backend_availability"]
