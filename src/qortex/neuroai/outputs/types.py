"""Canonical output type classes for Qortex NeuroAI runtime.

These structured types represent model outputs for specific tasks and
carry enough metadata to trace results back to the source data, model,
and preprocessing chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """Single detection bounding box in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_index: int
    confidence: float
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_coco(self) -> list[float]:
        """Return [x, y, width, height] in COCO format."""
        return [self.x1, self.y1, self.width, self.height]

    def to_yolo(self, img_w: float, img_h: float) -> list[float]:
        """Return [cx, cy, w, h] normalized 0-1 in YOLO format."""
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = self.width / img_w
        h = self.height / img_h
        return [cx, cy, w, h]


@dataclass
class ClassificationOutput:
    """Structured output for classification tasks."""
    class_name: str
    class_index: int
    probabilities: dict[str, float]
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_probs(
        cls,
        probs: dict[str, float],
        top_k: int = 5,
    ) -> "ClassificationOutput":
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best_name, best_prob = sorted_probs[0]
        best_idx = list(probs.keys()).index(best_name)
        return cls(
            class_name=best_name,
            class_index=best_idx,
            probabilities=probs,
            confidence=best_prob,
            top_k=sorted_probs[:top_k],
        )


@dataclass
class DetectionOutput:
    """Structured output for object detection tasks."""
    boxes: list[BoundingBox]
    n_detections: int
    image_shape: tuple[int, int]  # (H, W)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, image_shape: tuple[int, int] = (0, 0)) -> "DetectionOutput":
        return cls(boxes=[], n_detections=0, image_shape=image_shape)


@dataclass
class SegmentationOutput:
    """Structured output for segmentation tasks."""
    mask: Any                          # numpy [H, W] or [Z, Y, X] or [N, H, W]
    n_classes: int
    class_labels: dict[int, str]       # {index: name}
    affine: list | None = None         # 4×4 matrix (nested list) for 3D
    voxel_sizes: tuple | None = None   # (dz, dy, dx) in mm
    geometry_validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionOutput:
    """Structured output for regression tasks."""
    value: float
    units: str | None = None
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingOutput:
    """Structured output for embedding / feature extraction tasks."""
    vector: Any           # numpy [D]
    dimensionality: int
    model_layer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesPredictionOutput:
    """Structured output for frame-by-frame time-series prediction."""
    predictions: Any                       # numpy [T] or [T, n_classes]
    timestamps: list[float] | None = None  # Unix epoch per sample
    sampling_rate_hz: float | None = None
    label_map: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventMarkerOutput:
    """Structured output for closed-loop trigger events."""
    label: str
    value: float | str
    timestamp: float            # Unix epoch
    duration: float | None = None
    confidence: float | None = None
    source_window: dict | None = None   # which data window triggered this
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VolumePredictionOutput:
    """Structured output for volumetric (3D) prediction tasks."""
    mask: Any                                  # numpy [Z, Y, X]
    affine: list                               # 4×4 as nested list
    voxel_sizes: tuple[float, float, float]    # (dz, dy, dx) in mm
    n_classes: int
    class_labels: dict[int, str]
    geometry_source: str = "from_source"       # "from_source" | "resampled"
    geometry_validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportOutput:
    """Structured output for clinical-style text reports."""
    title: str
    findings: list[str]
    measurements: dict[str, float]
    confidence: str              # "high" | "medium" | "low"
    warnings: list[str]
    source_id: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BoundingBox",
    "ClassificationOutput",
    "DetectionOutput",
    "SegmentationOutput",
    "RegressionOutput",
    "EmbeddingOutput",
    "TimeSeriesPredictionOutput",
    "EventMarkerOutput",
    "VolumePredictionOutput",
    "ReportOutput",
]
