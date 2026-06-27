"""Auto-select the correct ModelAdapter from a ModelSpec."""

from __future__ import annotations

from qortex.neuroai.models._base import ModelAdapter
from qortex.neuroai.spec import ModelSpec


def make_model_adapter(spec: ModelSpec) -> ModelAdapter:
    """Factory: return the right ModelAdapter for the given spec.

    Raises
    ------
    ValueError
        When the provider is unknown or unsupported.
    ImportError
        When the required optional dependency is missing.
    """
    provider = (spec.provider or "").lower().strip()

    if provider in ("huggingface", "hf", "transformers"):
        from qortex.neuroai.models.huggingface import HuggingFaceAdapter
        return HuggingFaceAdapter(spec)

    if provider in ("onnx", "onnxruntime"):
        from qortex.neuroai.models.onnx import ONNXModelAdapter
        return ONNXModelAdapter(spec)

    if provider in ("torch", "pytorch", "torchscript"):
        from qortex.neuroai.models.torch import TorchModelAdapter
        return TorchModelAdapter(spec)

    if provider in ("monai", "monai_bundle"):
        from qortex.neuroai.models.monai import MONAIBundleAdapter
        return MONAIBundleAdapter(spec)

    if provider in ("braindecode", "bd"):
        from qortex.neuroai.models.braindecode import BrainDecodeAdapter
        return BrainDecodeAdapter(spec)

    if provider in ("ultralytics", "yolo"):
        from qortex.neuroai.models.ultralytics import UltralyticsAdapter
        return UltralyticsAdapter(spec)

    raise ValueError(
        f"Unknown model provider: {provider!r}. "
        f"Supported: 'huggingface', 'onnx', 'torch', 'monai', 'braindecode', 'ultralytics'."
    )
