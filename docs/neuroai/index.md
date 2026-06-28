# NeuroAI Runtime

The NeuroAI runtime connects neuroimaging data sources to AI models and structured output sinks through a single declarative pipeline. It separates compatibility checking from execution: the model is never loaded until the runtime has confirmed the source and model can actually work together.

## The core loop

```
source.probe()          ← header-only scan, no data loaded
model.inspect()         ← config.json only, no weights
CompatibilityEngine     ← reports required transforms, blockers, unknowns
PreprocessPlanner       ← builds the documented transform chain
model.load()            ← weights loaded only when check passes
source.stream()         ← windowed data
TransformExecutor       ← applies the plan
model.predict()         ← inference
output.write()          ← one or many sinks
ArtifactWriter          ← 9-file provenance directory
```

Nothing in this chain is implicit. Each step produces a typed contract: `SourceProfile`, `ModelProfile`, `CompatibilityReport`, `PreprocessPlan`, `PipelineRunReport`, `ArtifactContract`. These contracts flow into the artifact directory written after each run.

## When to use it

Use the Dataset workflow (`qortex.Dataset`) when your goal is to work with OpenNeuro BIDS datasets — catalog search, download, conversion, readiness checks.

Use the NeuroAI runtime when:
- You have local data (EDF, NWB, XDF, DICOM, video) and want to run a model on it
- You have a live data stream (LSL, BrainFlow) and need real-time inference
- You want to evaluate model compatibility before committing to a full run
- You need a traceable artifact with provenance for every inference run

## Contents

- [Pipeline](pipeline.md) — YAML format, Python API, `check()`, `run()`, `benchmark()`, `replay()`
- [Sources](sources.md) — BIDS, DICOM, NWB, XDF, LSL, BrainFlow, image, video
- [Models](models.md) — HuggingFace, ONNX, Torch, MONAI, Ultralytics, custom plugins
- [Outputs](outputs.md) — JSONL, Parquet, CSV, LSL markers, NIfTI, DICOM-SEG, BIDS, COCO, YOLO, overlay, HTTP
- [Contracts](contracts.md) — SourceProfile, ModelProfile, CompatibilityReport, PreprocessPlan, ArtifactContract
- [Ring buffer](ring-buffer.md) — lock-free windowing for real-time sources; Python fallback + Rust extension

## Quick example

```python
from qortex.neuroai import Pipeline

pipe = Pipeline.from_yaml("pipeline.yaml")

report = pipe.check()
print(report.summary())
# status: runnable
# transforms: [resample_250→128Hz, z_score_per_channel]
# blockers: []
# unknowns: [channel_labels]

if report.is_runnable:
    run = pipe.run(artifact_dir="artifacts/run_001")
    print(run.latency_report.summary())
```

```bash
qortex neuroai check pipeline.yaml
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001
qortex neuroai suggest-models data.edf
```
