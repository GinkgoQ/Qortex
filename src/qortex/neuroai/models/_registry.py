"""Auto-select the correct ModelAdapter from a ModelSpec."""

from __future__ import annotations

from qortex.neuroai.models._base import ModelAdapter
from qortex.neuroai.spec import ModelSpec


def _enforce_zoo_gates(spec: ModelSpec) -> None:
    """Enforce the zoo entry's license/remote-code policy, if one exists.

    This is the one choke point every adapter construction path already
    goes through -- CLI commands and any direct Python API caller alike --
    so a caller cannot bypass a declared zoo runtime policy simply by
    skipping the CLI. A no-op when spec.id has no corresponding zoo entry
    (e.g. a raw local bundle/checkpoint path never registered in the zoo).
    """
    try:
        from qortex.neuroai.models.zoo.registry import lookup
    except Exception:
        return
    entry = lookup(spec.id)
    if entry is None:
        return
    from qortex.neuroai.models.license import check_license_gate
    from qortex.neuroai.models.security import check_remote_code_gate

    check_license_gate(
        entry,
        accept_unknown_license_risk=bool(spec.extra.get("accept_unknown_license_risk", False)),
    )
    check_remote_code_gate(
        entry,
        allow_remote_code=bool(spec.extra.get("allow_remote_code", False)),
    )


def resolve_provider_dispatch(spec: ModelSpec) -> ModelAdapter:
    """Construct the adapter for *spec*'s provider, with NO gate enforcement.

    This is the raw dispatch table, deliberately separate from
    make_model_adapter()'s gate-enforcing wrapper below. zoo/validate.py's
    offline registry self-check calls this directly: registry validation
    must verify "does this provider string resolve to a real adapter"
    without also tripping a license/remote-code gate that belongs to
    execution time, not registry structural validity -- otherwise a
    ModelAdapterError from an unrelated unknown-license entry would mask
    the actual unknown-provider ValueError this check exists to catch.

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

    if provider in ("torch", "pytorch"):
        from qortex.neuroai.models.torch import TorchModelAdapter
        return TorchModelAdapter(spec)

    if provider in ("torchscript", "ts"):
        from qortex.neuroai.models.torch import TorchModelAdapter
        # provider hint is used inside the adapter to decide jit.load vs torch.load
        return TorchModelAdapter(spec)

    if provider in ("monai", "monai_bundle"):
        from qortex.neuroai.models.monai import MONAIBundleAdapter
        return MONAIBundleAdapter(spec)

    if provider in ("vista3d",):
        from qortex.neuroai.models.monai import VISTA3DAdapter
        return VISTA3DAdapter(spec)

    if provider in ("braindecode", "bd"):
        from qortex.neuroai.models.braindecode import BrainDecodeAdapter
        return BrainDecodeAdapter(spec)

    if provider in ("ultralytics", "yolo"):
        from qortex.neuroai.models.ultralytics import UltralyticsAdapter
        return UltralyticsAdapter(spec)

    if provider in ("plugin", "custom"):
        from qortex.neuroai.models.plugin import CustomPluginAdapter
        return CustomPluginAdapter(spec)

    if provider in ("torchvision", "tv"):
        from qortex.neuroai.models.torchvision_adapter import TorchvisionAdapter
        return TorchvisionAdapter(spec)

    if provider in ("keras", "tensorflow", "tf"):
        from qortex.neuroai.models.keras_adapter import KerasAdapter
        return KerasAdapter(spec)

    if provider in ("medsam",):
        from qortex.neuroai.models.sam_adapters import MedSAMAdapter
        return MedSAMAdapter(spec)

    if provider in ("sam_med3d",):
        from qortex.neuroai.models.sam_adapters import SAMMed3DAdapter
        return SAMMed3DAdapter(spec)

    raise ValueError(
        f"Unknown model provider: {provider!r}. "
        f"Supported: 'huggingface', 'onnx', 'torch', 'torchscript', "
        f"'monai', 'vista3d', 'braindecode', 'ultralytics', 'torchvision', 'keras', "
        f"'medsam', 'sam_med3d', 'plugin'."
    )


def make_model_adapter(spec: ModelSpec) -> ModelAdapter:
    """Factory: return the right ModelAdapter for the given spec.

    Enforces the zoo entry's license and remote-code gates (see
    models/license.py, models/security.py) before constructing any adapter,
    when spec.id matches a registered zoo entry -- this is the canonical
    boundary, not just the CLI commands that happen to look up an entry
    first. Use resolve_provider_dispatch() directly only for gate-free
    structural checks (e.g. the offline zoo registry validator).

    Raises
    ------
    ValueError
        When the provider is unknown or unsupported.
    ImportError
        When the required optional dependency is missing.
    ModelAdapterError
        When the resolved zoo entry's license is unknown/blocked or its
        remote-code requirement is not explicitly accepted.
    """
    _enforce_zoo_gates(spec)
    return resolve_provider_dispatch(spec)
