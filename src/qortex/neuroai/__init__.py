"""Qortex NeuroAI Runtime — contract-driven source → model → output pipelines.

The NeuroAI runtime provides:

  * Declarative YAML pipeline specs
  * Source adapters: local EDF/BDF/FIF, BIDS, DICOM/DICOMweb, NWB, XDF, LSL,
    BrainFlow, image/video
  * Model adapters: HuggingFace, ONNX, PyTorch/TorchScript, MONAI bundles,
    Braindecode, Ultralytics YOLO, custom plugins
  * Compatibility engine: checks source↔model feasibility before weight loading
  * Preprocessing planner: builds the minimal, documented transform chain
  * Output adapters: JSONL, Parquet, CSV, LSL markers, WebSocket, HTTP,
    NIfTI, DICOM-SEG, DICOM-SR, BIDS derivatives, COCO JSON, YOLO txt, overlay
  * Closed-loop trigger system: class-conditional event markers to all outputs
  * Latency profiler: per-stage p50/p95/p99 benchmarking
  * Provenance: every artifact carries a full ArtifactContract (9-file directory)

Quickstart::

    from qortex.neuroai import Pipeline

    pipe = Pipeline.from_yaml("pipeline.yaml")
    report = pipe.check()
    print(report.summary())

    if report.is_runnable:
        run = pipe.run()
        print(run.latency_report.summary())

CLI::

    qortex neuroai check pipeline.yaml
    qortex neuroai run pipeline.yaml
    qortex neuroai benchmark pipeline.yaml
    qortex neuroai replay pipeline.yaml --source recording.xdf
    qortex neuroai inspect-source data.edf
    qortex neuroai inspect-model hf://org/model
"""

from qortex.neuroai.pipeline import Pipeline
from qortex.neuroai.spec import (
    ModelSpec,
    OutputSpec,
    PipelineSpec,
    PreprocessSpec,
    RuntimeSpec,
    SourceSpec,
    TriggerSpec,
    WindowSpec,
)
from qortex.neuroai.contracts import (
    ArtifactContract,
    AxisConvention,
    CompatibilityReport,
    CompatibilityStatus,
    EvidenceStatus,
    InputContract,
    LatencyReport,
    ModelProfile,
    OutputContract,
    PipelineRunReport,
    PreprocessPlan,
    QortexTimeSeries,
    QortexVolume,
    QortexImageSeries,
    QortexVideo,
    QortexEmbeddingTable,
    QortexClinicalContext,
    QortexStream,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
    WarningItem,
)
from qortex.neuroai.compatibility import CompatibilityEngine
from qortex.neuroai.preprocess import PreprocessPlanner, TransformExecutor
from qortex.neuroai.benchmark import PipelineProfiler
from qortex.neuroai.sources import SourceAdapter, make_source_adapter
from qortex.neuroai.models import ModelAdapter, ModelOutput, make_model_adapter
from qortex.neuroai.outputs import (
    OutputAdapter,
    make_output_adapter,
    ClassificationOutput,
    DetectionOutput,
    SegmentationOutput,
    RegressionOutput,
    EmbeddingOutput,
    TimeSeriesPredictionOutput,
    EventMarkerOutput,
    VolumePredictionOutput,
    ReportOutput,
    BoundingBox,
)
from qortex.neuroai.artifact import ArtifactWriter


def check(
    source: str | None = None,
    model: str | None = None,
    pipeline: str | None = None,
    *,
    task: str | None = None,
    provider: str = "huggingface",
) -> CompatibilityReport:
    """Convenience function: check source-model compatibility.

    Parameters
    ----------
    source:
        Path to a local file, BIDS directory, or source specifier.
    model:
        HuggingFace model ID, ONNX path, or model specifier.
    pipeline:
        Path to a pipeline YAML (alternative to source + model args).
    task:
        Model task hint, e.g. ``"eeg_classification"``.
    provider:
        Model provider: ``"huggingface"`` | ``"onnx"`` | ``"torch"``.

    Returns
    -------
    CompatibilityReport

    Examples
    --------
    >>> report = qortex.neuroai.check(
    ...     source="data.edf",
    ...     model="braindecode/eegnet",
    ...     task="eeg_classification",
    ... )
    >>> print(report.summary())
    """
    if pipeline:
        pipe = Pipeline.from_yaml(pipeline)
        return pipe.check()

    if source is None or model is None:
        raise ValueError("Provide either pipeline= or both source= and model=")

    spec = PipelineSpec.from_dict({
        "name": "ad_hoc_check",
        "source": {"type": "local_file", "path": source},
        "model": {"provider": provider, "id": model, "task": task},
        "outputs": [{"type": "jsonl", "path": "/tmp/qortex_adhoc.jsonl"}],
    })
    return Pipeline(spec).check()


__all__ = [
    # Top-level
    "Pipeline",
    "check",
    # Spec
    "PipelineSpec",
    "SourceSpec",
    "ModelSpec",
    "WindowSpec",
    "PreprocessSpec",
    "RuntimeSpec",
    "OutputSpec",
    "TriggerSpec",
    # Contracts
    "SourceProfile",
    "ModelProfile",
    "InputContract",
    "OutputContract",
    "CompatibilityReport",
    "CompatibilityStatus",
    "PreprocessPlan",
    "PipelineRunReport",
    "ArtifactContract",
    "LatencyReport",
    "WarningItem",
    "EvidenceStatus",
    "AxisConvention",
    "TransformKind",
    "TransformDescriptor",
    # Abstractions
    "QortexTimeSeries",
    "QortexVolume",
    "QortexImageSeries",
    "QortexVideo",
    "QortexEmbeddingTable",
    "QortexClinicalContext",
    "QortexStream",
    # Canonical output types
    "ClassificationOutput",
    "DetectionOutput",
    "SegmentationOutput",
    "RegressionOutput",
    "EmbeddingOutput",
    "TimeSeriesPredictionOutput",
    "EventMarkerOutput",
    "VolumePredictionOutput",
    "ReportOutput",
    "BoundingBox",
    # Artifact system
    "ArtifactWriter",
    # Engines
    "CompatibilityEngine",
    "PreprocessPlanner",
    "TransformExecutor",
    "PipelineProfiler",
    # Adapters
    "SourceAdapter",
    "make_source_adapter",
    "ModelAdapter",
    "ModelOutput",
    "make_model_adapter",
    "OutputAdapter",
    "make_output_adapter",
]
