from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.outputs._registry import make_output_adapter
from qortex.neuroai.outputs.types import (
    BoundingBox,
    ClassificationOutput,
    DetectionOutput,
    EmbeddingOutput,
    EventMarkerOutput,
    ReportOutput,
    RegressionOutput,
    SegmentationOutput,
    TimeSeriesPredictionOutput,
    VolumePredictionOutput,
)

__all__ = [
    "OutputAdapter",
    "make_output_adapter",
    # Canonical output types
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
