"""MedSAM and SAM-Med3D promptable segmentation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, ModelProfile, OutputContract
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.models.artifacts import download_medsam_checkpoint, resolve_medsam_checkpoint
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType
from qortex.neuroai.spec import ModelSpec, RuntimeSpec


class _BaseSAMAdapter(PromptableModelAdapter):
    """Shared shell for SAM-family adapters."""

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
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ModelAdapterError(
                f"{type(self).__name__} requires the 'segment_anything' package. "
                "Install it with: pip install "
                "git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        checkpoint = self._checkpoint_path()
        device = _resolve_torch_device(runtime.device)
        model_type = str(self._spec.extra.get("sam_model_type") or "vit_b")
        try:
            model = sam_model_registry[model_type](checkpoint=str(checkpoint))
            model.to(device)
            model.eval()
        except KeyError as exc:
            raise ModelAdapterError(f"Unsupported SAM model type: {model_type!r}") from exc
        except (RuntimeError, ValueError, OSError) as exc:
            raise ModelAdapterError(
                f"Failed to construct {type(self).__name__} from checkpoint {checkpoint}: {exc}"
            ) from exc

        self._model = model
        self._predictor = SamPredictor(model)
        self._loaded = True

    def _checkpoint_path(self) -> Path:
        raise ModelAdapterError(
            f"{type(self).__name__} has no standardized checkpoint resolver configured."
        )

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

        image, prompt_2d, slice_index = _prepare_sam_input(batch, prompt)
        self._predictor.set_image(image)
        point_coords = None
        point_labels = None
        if prompt_2d.points is not None:
            point_coords = np.asarray(prompt_2d.points, dtype=np.float32)
            point_labels = np.asarray(prompt_2d.point_labels or [1] * len(point_coords), dtype=np.int32)
        box = None
        if prompt_2d.boxes:
            box = np.asarray(prompt_2d.boxes[0], dtype=np.float32)
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        score = float(scores[0]) if len(scores) else None
        return ModelOutput(
            output_type="segmentation",
            raw=logits,
            mask=masks[0].astype(np.uint8),
            regression_value=score,
            metadata={
                "provider": self._provider_label,
                "sam_score": score,
                "slice_index": slice_index,
            },
        )

    def unload(self) -> None:
        self._model = None
        self._predictor = None
        self._loaded = False


class MedSAMAdapter(_BaseSAMAdapter):
    _provider_label = "medsam"

    def _checkpoint_path(self) -> Path:
        explicit = self._spec.extra.get("checkpoint") or self._spec.extra.get("checkpoint_path")
        if self._spec.extra.get("download_artifacts") or self._spec.extra.get("download_checkpoint"):
            return download_medsam_checkpoint(target=explicit)
        return resolve_medsam_checkpoint(explicit)


class SAMMed3DAdapter(_BaseSAMAdapter):
    _provider_label = "sam_med3d"


def _resolve_torch_device(device: str) -> str:
    import torch
    if device in ("auto", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _prepare_sam_input(batch: Any, prompt: Prompt) -> tuple[np.ndarray, Prompt, int | None]:
    array = _as_numpy(batch)
    if array.ndim == 2:
        return _to_rgb_uint8(array), _prompt_to_2d(prompt), None
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        return _to_rgb_uint8(array), _prompt_to_2d(prompt), None
    if array.ndim == 3:
        slice_index = _slice_index_from_prompt(prompt, array.shape[0])
        return _to_rgb_uint8(array[slice_index]), _prompt_to_2d(prompt), slice_index
    raise ModelAdapterError(
        f"MedSAM expects a 2D image, RGB image, or 3D volume; received array shape={array.shape!r}."
    )


def _as_numpy(batch: Any) -> np.ndarray:
    if isinstance(batch, np.ndarray):
        return batch
    if hasattr(batch, "data"):
        return np.asarray(batch.data)
    if hasattr(batch, "detach"):
        return batch.detach().cpu().numpy()
    return np.asarray(batch)


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        lo = float(np.nanpercentile(image, 0.5))
        hi = float(np.nanpercentile(image, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        image_u8 = (scaled * 255.0).astype(np.uint8)
        return np.stack([image_u8, image_u8, image_u8], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 3:
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32)
        lo = float(np.nanmin(image))
        hi = float(np.nanmax(image))
        if hi <= lo:
            hi = lo + 1.0
        return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    raise ModelAdapterError(f"Cannot convert array shape={image.shape!r} to an RGB image.")


def _slice_index_from_prompt(prompt: Prompt, depth: int) -> int:
    if prompt.points:
        first = prompt.points[0]
        if len(first) >= 3:
            return int(np.clip(round(float(first[2])), 0, depth - 1))
    if prompt.boxes:
        first = prompt.boxes[0]
        if len(first) >= 6:
            return int(np.clip(round((float(first[2]) + float(first[5])) / 2.0), 0, depth - 1))
    return depth // 2


def _prompt_to_2d(prompt: Prompt) -> Prompt:
    points = [tuple(point[:2]) for point in prompt.points] if prompt.points is not None else None
    boxes = None
    if prompt.boxes is not None:
        boxes = []
        for box in prompt.boxes:
            if len(box) >= 6:
                boxes.append((box[0], box[1], box[3], box[4]))
            else:
                boxes.append(tuple(box[:4]))
    return Prompt(points=points, point_labels=prompt.point_labels, boxes=boxes, text=prompt.text)


__all__ = ["MedSAMAdapter", "SAMMed3DAdapter"]
