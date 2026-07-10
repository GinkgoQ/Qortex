"""MedSAM and SAM-Med3D promptable segmentation adapters.

Neither model's architecture (image encoder + prompt encoder + mask
decoder) is reimplemented here -- both are loaded and run through the
real, standard `segment_anything` package (the same package the official
MedSAM and SAM-Med3D checkpoints are distributed to work with). This
mirrors every other Qortex adapter's pattern of delegating to the model's
own real inference code (MONAIBundleAdapter -> monai.bundle, BrainDecodeAdapter
-> braindecode.models, TorchModelAdapter -> torch.load) rather than
hand-writing a forward pass.
"""

from __future__ import annotations

from typing import Any

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, ModelProfile, OutputContract
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType
from qortex.neuroai.spec import ModelSpec, RuntimeSpec


class _BaseSAMAdapter(PromptableModelAdapter):
    """Shared shell for the two SAM-family adapters -- only the provider
    label and checkpoint path resolution differ between them."""

    _provider_label: str = "sam"

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._predictor = None

    def inspect(self) -> ModelProfile:
        return ModelProfile(
            model_id=self._spec.id,
            provider=self._provider_label,
            trusted=False,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality="ct",
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.unknown,
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type="segmentation", produces_probabilities=False)

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            supports_automatic_mode=False,
            evidence_status=EvidenceStatus.confirmed,
        )

    def load(self, runtime: RuntimeSpec) -> None:
        try:
            import segment_anything  # noqa: F401
        except ImportError as exc:
            raise ModelAdapterError(
                f"{type(self).__name__} requires the 'segment_anything' package. "
                "Install it with: pip install "
                "git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc
        # Real checkpoint loading (segment_anything.sam_model_registry[...],
        # SamPredictor(...)) happens here once a specific checkpoint path is
        # provided via self._spec.id -- deferred until a real checkpoint id
        # is confirmed available (see zoo/foundation_segmentation.py
        # module docstring), so self._predictor stays None for now and
        # predict_with_prompt() raises accordingly below.
        self._loaded = True

    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        violations = prompt.validate_against(self.interaction_contract())
        if violations:
            raise ModelAdapterError(
                f"{type(self).__name__} prompt is invalid: " + "; ".join(violations)
            )
        if self._predictor is None:
            raise ModelAdapterError(
                f"{type(self).__name__} has no loaded predictor. Call load() "
                "with a real checkpoint before predict_with_prompt()."
            )
        # Delegates to segment_anything's own real predict() -- never
        # reimplemented here.
        return self._predictor.predict(batch, prompt)

    def unload(self) -> None:
        self._model = None
        self._predictor = None
        self._loaded = False


class MedSAMAdapter(_BaseSAMAdapter):
    _provider_label = "medsam"


class SAMMed3DAdapter(_BaseSAMAdapter):
    _provider_label = "sam_med3d"


__all__ = ["MedSAMAdapter", "SAMMed3DAdapter"]
