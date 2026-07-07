"""Keras Applications model adapter (image classification zoo).

spec.id is a class name in ``tensorflow.keras.applications`` (e.g.
``"EfficientNetB0"``, ``"MobileNetV2"``). Pretrained ImageNet weights are
downloaded only when ``spec.extra["pretrained"]`` is explicitly ``True``.
"""

from __future__ import annotations

import logging
from typing import Any

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, ModelProfile, OutputContract
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)


class KerasAdapter(ModelAdapter):
    """Adapter for ``tf.keras.applications`` classification models."""

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model_name = spec.id
        self._pretrained = bool(spec.extra.get("pretrained", False))
        self._model = None

    def inspect(self) -> ModelProfile:
        ctor = _resolve_constructor(self._model_name)
        model = ctor(weights=None)
        n_params = int(model.count_params())
        input_shape = tuple(d for d in model.input_shape if d is not None)
        return ModelProfile(
            model_id=f"keras/{self._model_name}",
            provider="keras",
            task=self._spec.task or "classification",
            model_hash=None,
            estimated_params=n_params,
            input_contract=self.required_input(input_shape),
            output_contract=self.output_schema(),
        )

    def required_input(self, spatial_shape: tuple | None = None) -> InputContract:
        return InputContract(
            modality="image",
            spatial_shape=spatial_shape,
            dtype="float32",
            axis_convention=AxisConvention.channels_last,
            evidence_status=EvidenceStatus.inferred,
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type=self._spec.task or "classification")

    def load(self, runtime: RuntimeSpec) -> None:
        ctor = _resolve_constructor(self._model_name)
        weights = "imagenet" if self._pretrained else None
        self._model = ctor(weights=weights)
        self._loaded = True
        log.info("Loaded keras model %s (pretrained=%s)", self._model_name, self._pretrained)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        import numpy as np

        x = np.asarray(batch, dtype="float32")
        if x.ndim == 3:
            x = x[None, ...]
        out = self._model.predict(x, verbose=0)
        logits = np.asarray(out)[0]
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


def _resolve_constructor(name: str):
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError(
            "Keras model adapter requires tensorflow: pip install tensorflow"
        )
    ctor = getattr(keras.applications, name, None)
    if ctor is None:
        raise ValueError(f"Unknown keras.applications model: {name!r}")
    return ctor


__all__ = ["KerasAdapter"]
