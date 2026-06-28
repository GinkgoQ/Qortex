"""Ultralytics (YOLO) model adapter.

Supports YOLOv8/YOLO11 and other Ultralytics-family models for detection,
segmentation, classification, and pose estimation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    ModelProfile,
    OutputContract,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)


class UltralyticsAdapter(ModelAdapter):
    """Adapter for Ultralytics YOLO models.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="ultralytics"`` or ``"yolo"`` and
        ``id=<model name or path>`` e.g. ``"yolov8n.pt"``,
        ``"ultralytics/assets/yolov8n.pt"``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._task: str = "detect"
        self._device = "cpu"

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        YOLO = _require_yolo()
        model = YOLO(self._spec.id)
        self._task = str(getattr(model, "task", "detect") or "detect")

        try:
            info = model.info(verbose=False)
        except Exception:
            info = {}

        return ModelProfile(
            model_id=self._spec.id,
            provider="ultralytics",
            task=self._task,
            revision=None,
            model_hash=None,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality="image",
            n_channels=3,
            sampling_rate_hz=None,
            spatial_shape=(640, 640),
            dtype="float32",
            axis_convention=AxisConvention.batch_channels_xyz,
            evidence_status=EvidenceStatus.confirmed,
        )

    def output_schema(self) -> OutputContract:
        task = self._task.lower()
        output_type = (
            "detection" if task == "detect"
            else "segmentation" if task == "segment"
            else "classification" if task == "classify"
            else "pose" if task == "pose"
            else "detection"
        )
        return OutputContract(
            output_type=output_type,
            n_classes=None,
        )

    def load(self, runtime: RuntimeSpec) -> None:
        YOLO = _require_yolo()
        self._device = _resolve_device(runtime.device)
        self._model = YOLO(self._spec.id)
        self._task = str(getattr(self._model, "task", "detect") or "detect")
        self._model.to(self._device)
        if runtime.fp16 and "cuda" in self._device:
            self._model.half()
        self._loaded = True
        log.info("Loaded Ultralytics YOLO: %s (task=%s) on %s",
                 self._spec.id, self._task, self._device)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        # Convert to format expected by ultralytics
        if isinstance(batch, np.ndarray):
            # Expect [H, W, C] or [1, C, H, W] from batch
            if batch.ndim == 4:
                batch = batch[0]  # [C, H, W]
            if batch.ndim == 3 and batch.shape[0] in (1, 3):
                batch = batch.transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
            inp = (batch * 255).astype(np.uint8) if batch.max() <= 1.0 else batch.astype(np.uint8)
        elif hasattr(batch, "data"):
            arr = np.array(batch.data, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            inp = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        else:
            inp = batch

        results = self._model(inp, verbose=False)
        return self._parse_results(results)

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_results(self, results) -> ModelOutput:
        if not results:
            return ModelOutput(output_type=self._task, raw=results)

        r = results[0]
        task = self._task.lower()

        if task == "classify":
            probs = r.probs
            if probs is not None:
                p = probs.data.cpu().numpy() if hasattr(probs.data, "cpu") else np.array(probs.data)
                idx = int(np.argmax(p))
                names = r.names or {}
                return ModelOutput(
                    output_type="classification",
                    raw=p,
                    class_index=idx,
                    class_name=names.get(idx, f"class_{idx}"),
                    probabilities={names.get(i, f"class_{i}"): float(v) for i, v in enumerate(p)},
                )

        if task == "segment" and r.masks is not None:
            masks = r.masks.data.cpu().numpy() if hasattr(r.masks.data, "cpu") else np.array(r.masks.data)
            boxes = _parse_boxes(r, r.names or {})
            return ModelOutput(
                output_type="segmentation",
                raw=masks,
                mask=masks[0] if len(masks) == 1 else masks,
                bbox=boxes[0] if boxes else None,
                metadata={"all_boxes": boxes, "all_masks": masks.tolist() if masks.ndim <= 3 else None},
            )

        # Detection (default)
        boxes = _parse_boxes(r, r.names or {})
        return ModelOutput(
            output_type="detection",
            raw=r,
            bbox=boxes[0] if boxes else None,
            probabilities={b["class_name"]: b["confidence"] for b in boxes} if boxes else {},
            metadata={"detections": boxes},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_boxes(result, names: dict) -> list[dict]:
    boxes = []
    if result.boxes is None:
        return boxes
    try:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            boxes.append({
                "x1": float(xyxy[i, 0]),
                "y1": float(xyxy[i, 1]),
                "x2": float(xyxy[i, 2]),
                "y2": float(xyxy[i, 3]),
                "confidence": float(confs[i]),
                "class_index": int(cls_ids[i]),
                "class_name": names.get(int(cls_ids[i]), f"class_{cls_ids[i]}"),
            })
    except Exception as exc:
        log.debug("Error parsing YOLO boxes: %s", exc)
    return boxes


def _require_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics YOLO adapter requires ultralytics. "
            "Install with: pip install 'qortex[ultralytics]' or pip install ultralytics"
        )


def _resolve_device(device: str) -> str:
    try:
        import torch
        if device in ("auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"
