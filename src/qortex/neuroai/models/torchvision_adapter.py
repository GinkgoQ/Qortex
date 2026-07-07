"""torchvision model-zoo adapter (classification / detection / segmentation).

spec.id is a constructor name in ``torchvision.models`` (e.g. ``"resnet18"``,
``"fasterrcnn_resnet50_fpn"``). Pretrained weights are downloaded only when
``spec.extra["pretrained"]`` is explicitly ``True`` — never implicitly.
"""

from __future__ import annotations

import logging
from typing import Any

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, ModelProfile, OutputContract
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)


class TorchvisionAdapter(ModelAdapter):
    """Adapter for ``torchvision.models`` classification/detection/segmentation zoos."""

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model_name = spec.id
        self._pretrained = bool(spec.extra.get("pretrained", False))
        self._model = None
        self._device = "cpu"

    def inspect(self) -> ModelProfile:
        tv = _require_torchvision()
        ctor = _resolve_constructor(tv, self._model_name)
        model = ctor(weights=None)
        n_params = sum(p.numel() for p in model.parameters())
        return ModelProfile(
            model_id=f"torchvision/{self._model_name}",
            provider="torchvision",
            task=self._spec.task or _infer_task(self._model_name),
            model_hash=None,
            estimated_params=n_params,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        task = self._spec.task or _infer_task(self._model_name)
        spatial = (3, 224, 224) if task == "classification" else None
        return InputContract(
            modality="image",
            spatial_shape=spatial,
            dtype="float32",
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.inferred,
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type=self._spec.task or _infer_task(self._model_name))

    def load(self, runtime: RuntimeSpec) -> None:
        tv = _require_torchvision()
        self._device = _resolve_device(runtime.device)
        ctor = _resolve_constructor(tv, self._model_name)
        weights = "DEFAULT" if self._pretrained else None
        self._model = ctor(weights=weights).to(self._device)
        self._model.eval()
        self._loaded = True
        log.info("Loaded torchvision model %s (pretrained=%s) on %s", self._model_name, self._pretrained, self._device)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        torch = _require_torchvision_torch()
        import numpy as np

        x = batch
        if isinstance(batch, np.ndarray):
            x = torch.from_numpy(batch.astype(np.float32))
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self._device)

        task = self._spec.task or _infer_task(self._model_name)
        with torch.no_grad():
            out = self._model(x)

        if task == "detection":
            det = out[0] if isinstance(out, list) else out
            boxes = det["boxes"].cpu().numpy().tolist()
            scores = det["scores"].cpu().numpy().tolist()
            labels = det["labels"].cpu().numpy().tolist()
            return ModelOutput(
                output_type="detection", raw=det,
                metadata={"boxes": boxes, "scores": scores, "labels": labels},
            )

        logits = out.cpu().numpy()[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return ModelOutput(
            output_type="classification", raw=out, class_index=idx, class_name=f"class_{idx}",
            probabilities={f"class_{i}": float(p) for i, p in enumerate(probs)},
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False


def _require_torchvision():
    try:
        import torchvision
        return torchvision
    except ImportError:
        raise ImportError(
            "torchvision model adapter requires torchvision: pip install torchvision"
        )


def _require_torchvision_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("torchvision model adapter requires torch: pip install torch")


def _resolve_constructor(tv, name: str):
    ctor = getattr(tv.models, name, None)
    if ctor is None:
        raise ValueError(f"Unknown torchvision model: {name!r}")
    return ctor


def _infer_task(name: str) -> str:
    if any(k in name for k in ("fasterrcnn", "retinanet", "ssd", "fcos", "maskrcnn", "keypointrcnn")):
        return "detection"
    if any(k in name for k in ("deeplab", "fcn_", "lraspp")):
        return "segmentation"
    return "classification"


def _resolve_device(device: str) -> str:
    try:
        import torch
        if device in ("auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"


__all__ = ["TorchvisionAdapter"]
