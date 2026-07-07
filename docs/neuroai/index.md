# NeuroAI Runtime

!!! warning "Research runtime"
    NeuroAI runs research inference pipelines. It does not diagnose, treat, triage, or recommend care. Validate every source, transform, model, and output contract before using results in a scientific or regulated workflow.

NeuroAI connects a data source, a model, and one or more outputs through a typed pipeline contract. Its main job is to prevent an unsafe run from starting: the runtime probes the source, inspects the model contract, plans required transforms, and reports blockers before weights are loaded.

## Use It When

| You need | Why NeuroAI fits |
|---|---|
| Source/model compatibility before inference | `check()` reports required transforms, blockers, unknowns, and whether the run is allowed. |
| Reproducible inference artifacts | Runs write predictions plus contracts, provenance, warnings, latency, and validation records. |
| Multiple data sources | Local files, BIDS directories, DICOM, NWB, XDF, LSL, BrainFlow, images, and video use the same source profile contract. |
| Explicit preprocessing | Required transforms come from the model input contract; denied transforms become blockers. |
| Replay and benchmark loops | `benchmark()` and `replay()` reuse the same pipeline semantics as `run()`. |

Use `qortex.Dataset` instead when your job is OpenNeuro search, readiness, selective download, conversion, or artifact creation.

## Runtime Loop

```text
source.probe()       header-only source profile
model.inspect()      model contract, no weights yet
check()              compatibility report
plan()               preprocessing steps
model.load()         only after compatibility passes
source.stream()      windows or records
transform()          strict critical transforms
predict()            model output
write()              JSONL, CSV, Parquet, NIfTI, DICOM, BIDS, overlay, HTTP, ...
artifact()           provenance, contract, warnings, latency, validation
```

Each step produces a typed object: `SourceProfile`, `ModelProfile`, `CompatibilityReport`, `PreprocessPlan`, `PipelineRunReport`, and `ArtifactContract`.

## Minimal Pipeline

```yaml
source:
  type: local_file
  path: data/sub-01.edf
  modality: eeg

model:
  type: braindecode
  name: eegnet

outputs:
  - type: jsonl
    path: predictions/predictions.jsonl

runtime:
  device: cpu
  max_windows: 10
```

```bash
qortex neuroai check pipeline.yaml
qortex neuroai plan pipeline.yaml --json
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001
```

## What Compatibility Means

Compatibility is not a slogan. It is a structured comparison between the source profile and the model input contract.

| Check | Examples |
|---|---|
| Modality | EEG model cannot silently accept a DICOM volume. |
| Shape and axes | Channel/time arrays, image volumes, row/column tables, batch dimensions. |
| Sampling and spacing | EEG sampling rate, voxel spacing, temporal windows. |
| Channels | Required channel names, channel order, channel-selection plans. |
| Dtype and range | Casts, intensity rescaling, normalization requirements. |
| Required transforms | Resample, normalize, reorient, pad/crop, transpose, channel select. |
| Policy gates | `preprocessing.allow`, `preprocessing.deny`, and per-transform booleans. |

A denied required transform is an incompatibility blocker. Unknown model dimensions stay `uncertain`; Qortex does not guess them.

## Artifact Contents

Every run can write a self-describing artifact directory:

| File | Purpose |
|---|---|
| `artifact_contract.json` | What the run promised to produce. |
| `artifact_manifest.json` | Files written by the run and their hashes. |
| `provenance.json` | Source, model, config, runtime, and environment metadata. |
| `compatibility_report.json` | Source/model compatibility evidence. |
| `preprocess_plan.json` | Ordered transforms and whether they are reversible. |
| `runtime_report.json` | Windows processed, dropped records, outputs written. |
| `latency_report.json` | Timing summary for source, preprocessing, model, and output stages. |
| `warnings.json` | Non-fatal problems kept with the run. |
| `outputs/` | Predictions and sink-specific files. |

## Strongest Paths Today

| Area | Status |
|---|---|
| Local EDF/BDF/FIF, NIfTI, DICOM | Tested source paths |
| BIDS directory source | Tested |
| JSONL, CSV, Parquet outputs | Tested |
| ONNX, Torch, Braindecode, MONAI, Ultralytics, HuggingFace adapters | Available with contract checks |
| DICOM-SEG, DICOM-SR, COCO, YOLO, overlay, HTTP outputs | Implemented paths; use workflow fixtures before making claims |
| LSL, BrainFlow, XDF, DICOMweb | Useful for integration work; validate with your hardware or service |

## Pages

- [Pipeline](pipeline.md): YAML schema, Python API, `check()`, `plan()`, `run()`, `benchmark()`, and `replay()`.
- [Sources](sources.md): local files, BIDS, DICOM, DICOMweb, NWB, XDF, LSL, BrainFlow, image, video.
- [Models](models.md): HuggingFace, ONNX, Torch, MONAI, Braindecode, Ultralytics, and trusted plugin adapters.
- [Outputs](outputs.md): JSONL, Parquet, CSV, NIfTI, DICOM-SEG, DICOM-SR, BIDS, COCO, YOLO, overlay, HTTP, WebSocket.
- [External runners](external-runners.md): subprocess boundary for file-based segmentation engines such as TotalSegmentator and nnU-Net.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/neuroai-contract-flow.png" alt="Contract flow diagram from source profile to model contract, compatibility report, preprocessing plan, and artifact validation.">
  <figcaption>Qortex checks source metadata, model input contracts, required transforms, and artifact validation before model execution.</figcaption>
</figure>

```bash
qortex neuroai run pipeline.yaml --artifact-dir docs/assets/results/neuroai/demo_artifact
qortex neuroai validate-artifact docs/assets/results/neuroai/demo_artifact
```

Result artifact: [neuroai-fixture-validation.txt](/Qortex/assets/results/neuroai-fixture-validation.txt)

<!-- qortex-evidence:end -->
