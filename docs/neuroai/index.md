# NeuroAI Runtime

!!! warning "Research runtime"
    The NeuroAI runtime is a contract-driven inference surface, but it is not a
    clinical device and does not make medical decisions. Validate every source,
    transform, model, and output contract for your own scientific or regulated
    workflow before relying on results.

The NeuroAI runtime connects neuroimaging data sources to AI models through a single
declarative pipeline. It separates compatibility checking from execution: the model is
never loaded until the runtime has confirmed the source and model can work together.

## The core loop

```
source.probe()          ← header-only scan, no data loaded
model.inspect()         ← config.json only, no weights
CompatibilityEngine     ← reports required transforms, blockers, unknowns
PreprocessPlanner       ← builds the documented transform chain
model.load()            ← weights loaded only when check passes
source.stream()         ← windowed data
TransformExecutor       ← applies the plan; raises TransformError on failure
model.predict()         ← inference
output.write()          ← one or many sinks
ArtifactWriter          ← 9-file provenance directory
```

Nothing in this chain is implicit. Each step produces a typed contract:
`SourceProfile`, `ModelProfile`, `CompatibilityReport`, `PreprocessPlan`,
`PipelineRunReport`, `ArtifactContract`. These contracts flow into the artifact
directory written after each run.

Pipeline configuration is parsed as a contract, not as loose YAML. `subject` and
`session` are accepted as scalar aliases for `subjects` and `sessions`; window
durations accept numeric seconds or strings such as `"2s"` / `"500ms"`; boolean
strings are parsed explicitly. Invalid or contradictory specs raise
`ContractValidationError` through `Pipeline.from_yaml()` / `Pipeline.from_dict()`.

Preprocessing is also contract-driven. The compatibility engine plans only
transforms required by the model input contract, and it respects
`preprocessing.allow`, `preprocessing.deny`, and boolean gates such as
`normalize`, `resample`, and `channel_select`. A denied required transform becomes
an incompatibility blocker rather than a silent automatic operation.

## When to use it

Use `qortex.Dataset` when your goal is OpenNeuro BIDS datasets — catalog search,
download, conversion, readiness checks.

Use the NeuroAI runtime when:

- You have local data (EDF, NWB, XDF, DICOM) and want to run a model on it
- You want to evaluate source-model compatibility **before** committing to a run
- You need a traceable artifact with provenance for every inference run

## Contents

- [Pipeline](pipeline.md) — YAML format, Python API, `check()`, `run()`, `benchmark()`, `replay()`
- [Sources](sources.md) — BIDS, DICOM, NWB, XDF, LSL, BrainFlow, image, video
- [Models](models.md) — HuggingFace (native tasks), ONNX, Torch, MONAI, Braindecode, Ultralytics, plugins
- [Outputs](outputs.md) — JSONL, Parquet, CSV, NIfTI, DICOM-SEG, BIDS, COCO, YOLO, overlay, HTTP
- Contracts — SourceProfile, ModelProfile, CompatibilityReport, PreprocessPlan, ArtifactContract
- Ring buffer — lock-free windowing for real-time sources; Python fallback + Rust extension

## Quick example

```python
from qortex.neuroai import Pipeline

pipe = Pipeline.from_yaml("pipeline.yaml")

report = pipe.check()
print(report.summary())
# CompatibilityReport: COMPATIBLE_WITH_TRANSFORMS
#   source=local_file:sub-01.edf  model=braindecode/EEGNet
#   Required transforms (1):
#     • resample(from_hz=250.0, to_hz=128.0)  [irreversible]
#   Warnings (1):
#     ⚠ channel labels inferred from index, not names

if report.is_runnable:
    run = pipe.run(artifact_dir="artifacts/run_001")
    print(run.latency_report.summary())
```

```bash
qortex neuroai check pipeline.yaml
qortex neuroai plan pipeline.yaml --json
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001
qortex neuroai suggest-models data.edf --task classification
```

## Known limitations

### HuggingFace adapter

`transformers.pipeline()` accepts a fixed set of native task strings
(`image-classification`, `audio-classification`, etc.).
Domain-specific tasks such as `eeg_classification` are **not** native pipeline tasks.
The adapter fails closed for those tasks unless the user supplies an explicit
native task, ONNX/Torch/Braindecode/MONAI model, or trusted plugin adapter.

For EEG or medical-imaging models, prefer the **Braindecode**, **ONNX**, or **Torch**
adapter.  The HuggingFace adapter's input contract inference reads `num_channels` /
`in_channels` from the model config when available; when those fields are absent the
channel count is `unknown` (not guessed).  Window duration is **never** estimated from
config fields — doing so has no scientific basis.

### Compatibility engine

When a model does not declare `n_channels` or `sampling_rate_hz` in its config,
the engine marks those dimensions as `uncertain`.  This is intentional: a false
`compatible` verdict is more dangerous than an honest `uncertain`.  Build a curated
`InputContract` for models that lack machine-readable specs.

### Preprocessing — contract-driven only

The `PreprocessPlanner` inserts only transforms that the `CompatibilityEngine`
determined are required by the model's `InputContract`.  No normalization or
per-modality intensity rescaling is added automatically.  Wrong normalization
destroys the distribution a model expects; only the model contract knows what's right.
Unknown normalization methods raise `TransformError` instead of silently
passing data through. Supported methods include `channel_zscore`,
`global_zscore`, `robust_zscore`, `per_volume_zscore`, `minmax`,
`percentile_clip`, `hu_window`, and `exponential_moving_standardize`.

### TransformExecutor — critical transforms raise, not skip

Critical transforms — `resample`, `reorient`, `normalize`, `rescale_intensity`,
`cast_dtype`, `bandpass`, `channel_select`, `pad_or_crop`, `transpose_axes` — raise `TransformError`
on failure.  The engine catches these at the window level, records the window as
dropped, and continues.  Non-critical structural transforms (`add_batch_dim`,
`to_tensor`) log a warning and pass data through.

Runtime failure behavior is explicit. `runtime.source_failure_policy` controls
source iterator errors (`strict`, `skip_window`, `continue_recording`), and
`runtime.preprocess_failure_policy` controls batch preprocessing errors
(`strict`, `drop_failed`).
`runtime.max_windows`, `runtime.max_duration_s`, `runtime.idle_timeout_s`, and
`runtime.fail_on_no_windows` bound smoke tests, offline replay, and long-running
streams without changing model or preprocessing semantics.

### Reorientation

Volumetric reorientation uses `nibabel.orientations` when nibabel is installed
(any 3-character orientation code pair).  Without nibabel, only LPS↔RAS is
supported via a direct axis flip, and only when the array axes map directly to the
orientation codes.  Install `qortex[mri]` for correct reorientation.

### Latency profiler

Source-read time is measured around the actual blocking `next()` call on the
source iterator.  Benchmark numbers are a lower bound — they do not include Python
GIL contention, data-loader initialisation, or GPU↔CPU transfer outside the timed
region.

### Source and output adapter coverage

| Adapter | Status |
|---|---|
| Local EDF/BDF/FIF, NIfTI, DICOM | Tested |
| BIDS directory | Tested |
| LSL, BrainFlow, XDF | Prototype |
| DICOMweb | Prototype |
| JSONL, Parquet, CSV output | Tested |
| DICOM-SEG, DICOM-SR | Partial |
| COCO JSON, YOLO txt, WebSocket | Partial |

### Scope

The runtime covers many source, model, and output types. The strongest path is
the contract-checked `check → plan → run → artifact → validate` workflow; source
or model adapters marked prototype/partial should be promoted with project
fixtures before they are used for claims.
