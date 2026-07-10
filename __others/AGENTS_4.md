Yes. The correct direction is **not “Qortex vs MONAI.”** The direction is:

> **Qortex must absorb MONAI’s useful capabilities as engines/backends, then add the missing neuroscience-specific, BIDS/OpenNeuro-specific, contract-driven, audit-driven, real-time, and model-selection layers that MONAI does not solve.**

MONAI should become one major subsystem inside Qortex, not the ceiling.

---

# 1. Product target

Qortex should become:

```text
The full-stack AI-neuroscience operating layer:
OpenNeuro/BIDS → validation → source profiling → model matching → preprocessing compiler
→ MONAI/Braindecode/ONNX/Torch/HF/nnU-Net/TotalSegmentator execution
→ training/inference/replay/benchmark/deployment
→ scientific artifact + audit + visualization + reproducibility
```

MONAI is excellent for **medical imaging deep learning**: transforms, networks, losses, metrics, training workflows, bundles, model zoo, annotation, and deployment. Its own README defines it as a PyTorch-based healthcare imaging framework with flexible preprocessing, compositional APIs, domain-specific networks/losses/metrics, and multi-GPU/multi-node support.

Qortex must become broader:

```text
medical imaging + EEG + MEG + iEEG + fNIRS + fMRI + DWI + PET + CT + DICOM + BIDS + OpenNeuro + live neuro streams + model contracts + scientific audit
```

---

# 2. Core principle: reuse first, rebuild only when needed

Do **not** rebuild MONAI features blindly. That wastes time and produces weaker versions. The mature strategy is:

| Decision                              | Rule                                                                                                                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Use MONAI directly**                | If MONAI already has a mature, tested implementation: transforms, networks, losses, metrics, sliding-window inference, bundles, Label, Deploy.                                    |
| **Wrap MONAI**                        | If Qortex needs source-model contracts, BIDS/OpenNeuro mapping, artifact provenance, or model selection around MONAI.                                                             |
| **Extend MONAI**                      | If the feature exists but does not support Qortex’s neurodata cases, such as BIDS semantics, EEG contracts, fMRI TR policies, source-model compatibility.                         |
| **Replace MONAI**                     | Only if MONAI’s design cannot satisfy scientific correctness, auditability, multi-modal NeuroAI, or reproducibility requirements.                                                 |
| **Build novel Qortex-native systems** | Where MONAI does not operate: OpenNeuro search/readiness, compatibility compiler, scientific artifact, multi-model matching, live stream replay, BIDS-aware model recommendation. |

---

# 3. Full MONAI parity map

## A. MONAI Core features Qortex must support

MONAI Core provides the foundation for medical imaging AI: preprocessing, compositional APIs, networks, losses, metrics, training workflows, multi-GPU/multi-node support.

Qortex needs parity through one of two paths:

| MONAI feature            | Qortex implementation                                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image transforms         | Use MONAI transforms as executable backend. Add Qortex contract wrappers.                                                                             |
| Dictionary transforms    | Support Qortex sample dicts: `image`, `label`, `mask`, `metadata`, `affine`, `subject`, `session`, `task`.                                            |
| Lazy resampling          | Expose MONAI lazy resampling in Qortex preprocessing plans. MONAI supports lazy resampling to reduce repeated spatial resamples and information loss. |
| Invertible transforms    | Store inverse transform stack in Qortex artifact.                                                                                                     |
| Networks                 | Import MONAI networks as model providers.                                                                                                             |
| Losses                   | Use MONAI losses for training mode.                                                                                                                   |
| Metrics                  | Use MONAI metrics for segmentation/classification evaluation.                                                                                         |
| Sliding-window inference | Use MONAI `sliding_window_inference` as the default volume inference backend; Qortex already does this inside its MONAI adapter.                      |
| Multi-GPU training       | Use PyTorch Distributed, MONAI workflows, Accelerate, or Lightning wrappers.                                                                          |
| Bundle format            | Treat MONAI Bundle as a first-class Qortex model package.                                                                                             |
| Docker                   | Support MONAI Docker-compatible runtime profiles.                                                                                                     |
| Model Zoo                | Index MONAI Model Zoo into Qortex model registry.                                                                                                     |

## B. MONAI Model Zoo parity

MONAI Model Zoo hosts models in MONAI Bundle format and archives versioned bundle ZIPs.

Qortex must support:

```text
qortex model import-monai-bundle
qortex model inspect-monai-bundle
qortex model validate-monai-bundle-contract
qortex model suggest --source <BIDS/NIfTI/DICOM/EDF>
qortex model run --engine monai
qortex model benchmark --engine monai
```

But Qortex should add what MONAI Model Zoo does not provide:

```text
source compatibility scoring
BIDS/OpenNeuro dataset matching
geometry compatibility checks
input contract completeness scoring
scientific risk scoring
run artifact generation
reproducibility lockfile
```

MONAI Model Zoo explicitly does not claim model suitability for diagnostic or therapeutic use. Qortex can add a **scientific suitability layer**, not a clinical approval claim.

## C. MONAI Label parity

MONAI Label is a full annotation server-client system with AI-assisted annotation, continuous learning, radiology/pathology/endoscopy support, 3D Slicer, OHIF, QuPath, DSA, CVAT, DICOMWeb, and active learning.

Qortex should not clone it. Qortex should integrate it:

```text
qortex label import-monai-label-session
qortex label export-bids-derivative
qortex label validate-label-quality
qortex label check-leakage
qortex label convert-to-training-artifact
qortex label audit-human-ai-annotation-history
```

Novel Qortex addition:

```text
BIDS-aware label governance:
- Which labels came from manual annotation?
- Which came from MONAI Label?
- Which came from model pseudo-labeling?
- Which were corrected by expert?
- Which labels leak subject/session/test information?
```

## D. MONAI Deploy parity

MONAI Deploy App SDK provides DAG-based apps, DICOM loading operators, PyTorch inference, Triton inference, MONAI transforms, app packaging, and local app runner.

Qortex should support MONAI Deploy as one deployment backend:

```text
qortex deploy export-monai-map
qortex deploy run-monai-app
qortex deploy validate-dicom-io
qortex deploy attach-qortex-artifact
qortex deploy compare-local-vs-deployed-output
```

But Qortex should add:

```text
contract-locked deployment:
- source contract
- model contract
- preprocessing contract
- output contract
- artifact contract
- version lock
- compatibility report
- failure policy
```

MONAI Deploy packages apps. Qortex should certify that the app’s **input assumptions are actually satisfied**.

---

# 4. Current Qortex strengths to preserve

Your rebuilt Qortex already has the right foundation.

## A. Contract-driven pipeline

`Pipeline.check()` probes source, inspects model, applies model contract overrides, checks compatibility, and builds preprocessing plan before loading weights.

This is the right architecture. Keep it as the central compiler.

## B. Compatibility engine

The current compatibility engine checks modality, channels, sampling rate, spatial shape, dtype, intensity range, axis convention, voxel spacing, coordinate frame, fMRI TR, memory, required metadata, and model warnings.

This is a Qortex differentiator. MONAI gives tools; Qortex decides whether a tool/model/source pairing is scientifically valid.

## C. Runtime controls

Qortex runtime now has batch size, workers, source failure policy, preprocess failure policy, max windows, max duration, idle timeout, trigger evaluation, output routing, and latency profiling.

This should become the universal inference runtime for neuroscience.

## D. Artifact system

Qortex writes a full artifact directory with manifest, contract, provenance, warnings, pipeline spec, compatibility report, preprocess plan, runtime report, and latency report.

This should become stricter and richer.

---

# 5. Critical Qortex gaps against MONAI

## Gap 1: medical imaging transform depth

Qortex has a compact transform executor. It is useful but not enough to match MONAI.

Current Qortex has transforms such as channel selection, channel map, resample, resample spatial, bandpass, pad/crop, reorient, intensity rescale, normalize, dtype cast, axis transpose, and tensor conversion.

But MONAI has a much deeper transform ecosystem with `Compose`, dictionary transforms, invertible transforms, lazy resampling, random transforms, augmentation, and metadata-aware pipelines.

### Required Qortex solution

Build a **Transform Backend Layer**:

```text
qortex.transforms.backends:
  - qortex_native
  - monai
  - torchio
  - mne
  - nilearn
  - dipy
  - itk/simpleitk
```

Each Qortex transform descriptor should compile to the strongest backend available.

Example:

```yaml
preprocessing:
  backend_priority: [monai, torchio, qortex_native]
  allow:
    - orientation
    - spacing
    - intensity
    - crop
    - pad
    - normalize
```

The Qortex plan should say:

```text
Transform: resample_spatial
Reason: model requires voxel_sizes_mm=[1.0,1.0,1.0]
Backend: MONAI Spacingd
Risk: irreversible interpolation
Affine update: required
Inverse tracking: enabled
Artifact: stored
```

## Gap 2: MONAI bundle preprocessing is not mapped

The current Qortex MONAI adapter intentionally refuses to guess MONAI preprocessing transforms. It warns when MONAI bundle config declares transforms that Qortex will not translate implicitly.

This is scientifically safe, but not enough for “best library.”

### Required Qortex solution

Build a **MONAI Bundle Contract Extractor**:

```text
MONAI metadata.json
MONAI inference.json
MONAI network_def
MONAI preprocessing transform graph
MONAI postprocessing transform graph
label map
spacing/orientation/intensity requirements
roi_size
sliding window settings
output classes
```

Then compile:

```text
MONAI Spacingd             → Qortex resample_spatial
MONAI Orientationd         → Qortex reorient
MONAI ScaleIntensityRanged → Qortex rescale_intensity
MONAI NormalizeIntensityd  → Qortex normalize
MONAI CropForegroundd      → Qortex crop_foreground
MONAI SpatialPadd          → Qortex pad
MONAI ResizeWithPadOrCropd → Qortex pad_or_crop
MONAI Invertd              → Qortex inverse_transform_stack
```

Fail if:

```text
affine missing
spacing missing
orientation unknown
label map missing
intensity range ambiguous
bundle uses custom callable transform
```

## Gap 3: affine and geometry provenance

Qortex now treats voxel spacing and coordinate frame as real compatibility checks. That is good. Voxel spacing mismatch can become `resample_spatial` only when allowed and target geometry exists; otherwise it blocks.

But current `resample_spatial` execution is array-level and uses `scipy.ndimage.zoom`; it returns the resampled array without a full visible affine/inverse metadata update.

### Required Qortex solution

Create a **Geometry Ledger**:

```text
before_affine
after_affine
before_spacing
after_spacing
before_orientation
after_orientation
interpolation_order
coordinate_frame
inverse_available
precision_loss
physical_space_valid
```

Every spatial transform must update the ledger.

Novel feature:

```text
qortex geometry verify-output
```

It checks:

```text
Does predicted mask align with input image?
Did affine change?
Was orientation preserved?
Was spacing resampled?
Can output be safely written as DICOM-SEG/NIfTI/BIDS derivative?
```

## Gap 4: BIDS semantics are shallow

The current BIDS adapter discovers subjects, modalities, target files, parses filename entities, and profiles representative recordings.

But Qortex must go far deeper than file discovery.

### Required Qortex solution

Build a **BIDS Semantic Engine**:

```text
events.tsv
channels.tsv
electrodes.tsv
coordsystem.json
participants.tsv
scans.tsv
*_bold.json
*_eeg.json
*_meg.json
*_ieeg.json
confounds.tsv
derivatives/
dataset_description.json
README
CHANGES
phenotype/
```

The engine should infer:

```text
supervised labels
event-aligned windows
class imbalance
task design
TR / sampling / timing consistency
channel montage
bad channels
subject leakage risk
session leakage risk
train/val/test split feasibility
derivatives availability
model compatibility
```

This is where Qortex can beat MONAI.

## Gap 5: training framework

MONAI has training workflows, networks, losses, metrics, and multi-GPU support.

Qortex currently looks stronger in inference/runtime/audit than training.

### Required Qortex solution

Add **Qortex Train**:

```text
qortex train plan
qortex train run
qortex train resume
qortex train evaluate
qortex train export
qortex train compare
```

Training backends:

```text
MONAI trainer for imaging
Braindecode trainer for EEG/MEG/iEEG
PyTorch Lightning / Accelerate for general models
sklearn for classical baselines
nnU-Net for segmentation baselines
```

Novel addition:

```text
Training feasibility compiler:
- Is there enough data?
- Are labels valid?
- Is subject split possible?
- Is leakage present?
- Is class balance acceptable?
- What baseline should run first?
- What model family is scientifically appropriate?
```

## Gap 6: evaluation and metrics

MONAI has strong imaging metrics. Qortex needs a broader neuroscience evaluation layer.

### Required Qortex metrics

```text
Segmentation:
- Dice
- IoU
- Hausdorff distance
- surface Dice
- lesion-wise F1
- volume error
- topology error
- small-object sensitivity

Classification:
- accuracy
- balanced accuracy
- macro F1
- AUROC
- AUPRC
- calibration error
- confusion matrix
- subject-level aggregation

EEG/MEG:
- event-level accuracy
- temporal tolerance accuracy
- onset detection error
- information transfer rate for BCI
- session generalization
- subject-transfer performance

fMRI:
- subject-level CV
- run-level CV
- temporal autocorrelation warning
- task-regressor leakage checks
- site/scanner confound checks

DWI:
- gradient table validation
- shell-specific QC
- b-vector/b-value consistency
```

Novel feature:

```text
qortex eval explain-failure
```

It tells whether failure came from:

```text
data quality
label quality
subject leakage
wrong preprocessing
model mismatch
geometry mismatch
sampling mismatch
insufficient data
domain shift
```

---

# 6. Novel features to make Qortex “the best”

These are the features that would make Qortex more than a MONAI wrapper.

## 1. NeuroAI Compatibility Compiler

Current compatibility engine is the seed. Make it a compiler.

Input:

```text
source profile + model contract + preprocessing policy + runtime policy
```

Output:

```text
runnable / blocked / uncertain
required transforms
scientific risks
runtime plan
artifact plan
evaluation plan
minimum data plan
```

This becomes:

```bash
qortex neuroai compile pipeline.yaml
```

Compiler output:

```text
Status: BLOCKED
Reason:
  - model expects RAS, source is LPS
  - model expects 1.0mm isotropic, source is 2.4×2.4×3.0mm
  - bundle requires ScaleIntensityRanged but source intensity range is unknown

Fix:
  - enable MONAI Orientationd
  - enable MONAI Spacingd
  - declare CT HU window or source intensity profile
```

## 2. Scientific Risk Score

Every run gets a risk score:

```text
green: source/model match confirmed
yellow: transforms required, low risk
orange: destructive transforms or inferred metadata
red: unknown labels, geometry mismatch, leakage risk, missing contract
black: blocked
```

Inputs:

```text
contract evidence
metadata completeness
transform destructiveness
label quality
source/model domain shift
artifact completeness
```

This is a major differentiator.

## 3. Model-Source Matchmaking

Qortex should recommend models for a dataset/source, not just run user-selected models.

Current `suggest-models` already probes source and scores curated contract entries.

Make it much deeper:

```text
qortex suggest-models ds003xxx --goal "motor imagery classification"
qortex suggest-models T1w.nii.gz --goal "brain tumor segmentation"
qortex suggest-models eeg.edf --goal "sleep staging"
```

Ranking criteria:

```text
modality match
channel match
sampling match
geometry match
label match
license
model evidence
benchmark results
artifact reproducibility
known limitations
domain similarity
```

## 4. Dataset-Model Co-evolution

Novel system:

```text
qortex build-model-card-from-source
qortex build-source-card-from-model
```

For any model:

```text
What dataset would this model need?
What preprocessing must exist?
What source metadata is required?
What labels are required?
What failure modes are expected?
```

For any dataset:

```text
Which model families are possible?
Which are impossible?
Which need additional labels?
Which need derivatives?
Which need preprocessing?
```

## 5. NeuroAI Reproducibility Lockfile

Create:

```text
qortex.lock
```

It should lock:

```text
source dataset ID/snapshot/hash
BIDS validator version
file manifest
model ID/revision/hash
MONAI version
Torch version
CUDA version
transform backends
pipeline hash
preprocessing plan hash
random seeds
split IDs
artifact schema version
```

This is stronger than standard experiment logs.

## 6. Preprocessing Proof System

For each transform, Qortex should answer:

```text
Why is this transform needed?
Who required it?
What does it change?
Is it reversible?
Does it alter scientific interpretation?
Can it be inverted?
Was it used during model training?
```

This already exists partially in `TransformDescriptor` with `required_by`, params, reversible, irreversible reason, and evidence status.

Expand it into:

```text
qortex explain-preprocessing
```

## 7. Cross-modal subject graph

Build a subject/session/run graph:

```text
subject
session
task
run
modality
recording
events
labels
derivatives
confounds
models run
outputs produced
```

This enables:

```text
cross-modal training
subject leakage detection
missing modality detection
longitudinal analysis
multi-session generalization
```

MONAI does not solve this.

## 8. Real-time NeuroAI layer

Qortex already supports LSL, XDF, BrainFlow, replay, triggers, and output markers through source registry/runtime/output adapters.

Build this into a full BCI/neurofeedback runtime:

```text
latency budget
jitter report
dropped-window report
trigger stability
event marker sync
clock drift detection
source replay
closed-loop simulation
```

MONAI is not designed for this.

## 9. Clinical/neuroscience artifact validator

Current Qortex validates artifact structure and hashes. Keep extending it.

Add validators:

```text
geometry alignment validator
DICOM-SEG validity validator
BIDS derivative validator
label-map validator
model-output semantic validator
calibration validator
source/model domain validator
PHI redaction validator
```

## 10. “Negative capability” engine

This is novel and important.

Qortex should not only say what can be done. It should say:

```text
What cannot be done with this dataset?
What cannot be trusted?
What model should not be run?
What labels are missing?
What preprocessing would be scientifically invalid?
What claims cannot be made?
```

This is a strong scientific feature.

---

# 7. Architecture for “all MONAI features + beyond MONAI”

## Layer 1: Data and source adapters

```text
qortex.sources
  bids
  openneuro
  local_nifti
  dicom
  dicomweb
  nwb
  xdf
  lsl
  brainflow
  edf/bdf/fif
  csv/parquet
  image/video
```

Current registry already supports many of these.

Add:

```text
fMRIPrep derivatives
FreeSurfer derivatives
BIDS derivatives
MNE-BIDS integration
DIPY DWI support
Nilearn fMRI support
TorchIO medical image datasets
```

## Layer 2: Contract system

```text
SourceContract
ModelContract
PreprocessContract
OutputContract
RuntimeContract
ArtifactContract
EvaluationContract
DeploymentContract
```

Current Qortex already has core contracts.

Add missing contracts:

```text
TrainingContract
GeometryContract
LabelContract
SplitContract
RiskContract
DeploymentContract
```

## Layer 3: Transform compiler

```text
Qortex transform descriptor
→ choose backend
→ compile executable transform
→ track metadata
→ prove compatibility
→ write transform ledger
```

Backends:

```text
MONAI for imaging
MNE for EEG/MEG
Nilearn for fMRI
DIPY for DWI
TorchIO/SimpleITK for geometry
Qortex-native for lightweight operations
```

## Layer 4: Model providers

```text
monai_bundle
monai_network
braindecode
onnx
torch
torchscript
huggingface
ultralytics
nnunet_external
totalsegmentator_external
custom/plugin
```

Qortex already has several of these providers and external runners.

## Layer 5: Training engine

```text
qortex.train
  monai_backend
  braindecode_backend
  torch_backend
  lightning_backend
  sklearn_baseline_backend
```

Essential commands:

```bash
qortex train plan
qortex train baseline
qortex train run
qortex train evaluate
qortex train export-model
qortex train export-bundle
qortex train compare
```

## Layer 6: Runtime engine

Current runtime engine is good. Expand into:

```text
offline inference
batch inference
stream inference
real-time inference
replay
benchmark
stress-test
drift-test
latency-test
closed-loop simulation
```

## Layer 7: Output system

Current output registry is broad. It includes JSONL, Parquet, CSV, LSL, NIfTI, DICOM-SEG, DICOM-SR, BIDS derivative, COCO, YOLO, WebSocket, HTTP, overlay.

Add:

```text
MLflow artifact
Weights & Biases artifact
DVC artifact
Hugging Face dataset/model output
MONAI Bundle export
NWB output
BIDS StatsModels output
FHIR/DICOM-SR richer report
```

## Layer 8: Artifact and audit

Make Qortex artifact the center of trust.

Current artifact layout is already strong.

Add:

```text
geometry_ledger.json
label_audit.json
split_audit.json
model_card.json
source_card.json
risk_report.json
dependency_lock.json
transform_inverse_stack.json
evaluation_report.json
deployment_report.json
```

---

# 8. “Best library” feature list

This is the target checklist.

## Data

```text
OpenNeuro search
OpenNeuro dataset profiling
BIDS validation
BIDS semantic graph
minimal download planner
metadata-only mode
derivative discovery
label discovery
events parser
channels parser
electrodes parser
coordsystem parser
confounds parser
DICOM/DICOMWeb support
NIfTI support
NWB support
XDF support
LSL live support
BrainFlow support
```

## Preprocessing

```text
MONAI transforms
MNE preprocessing
Nilearn preprocessing
DIPY preprocessing
TorchIO/SimpleITK geometry
Qortex transform compiler
affine-aware spatial transforms
inverse transform tracking
lazy resampling support
scientific risk per transform
preprocessing proof report
```

## Models

```text
MONAI Bundle
MONAI Model Zoo
Braindecode
ONNX
Torch/TorchScript
Hugging Face
Ultralytics
nnU-Net
TotalSegmentator
custom/plugin
model contract registry
model suitability scoring
model-source matchmaking
```

## Training

```text
MONAI imaging training
Braindecode EEG training
PyTorch/Lightning/Accelerate training
classical baselines
subject-wise CV
session-wise CV
site-wise CV
leakage-safe splits
hyperparameter search
ablation runner
training artifact
model export
```

## Inference

```text
batch inference
streaming inference
sliding-window inference
real-time inference
replay
latency benchmark
jitter benchmark
trigger system
closed-loop event output
DICOM-SEG output
BIDS derivative output
overlay output
HTTP/WebSocket output
```

## Evaluation

```text
segmentation metrics
classification metrics
calibration metrics
subject-level metrics
event-level metrics
BCI metrics
fMRI-specific checks
DWI-specific checks
geometry checks
output semantic validation
failure explanation
```

## Deployment

```text
FastAPI serving
Triton backend
MONAI Deploy export
Docker packaging
artifact-locked deployment
model contract endpoint
source contract endpoint
runtime health checks
latency SLA monitor
```

## Audit

```text
artifact manifest
provenance
compatibility report
preprocessing plan
geometry ledger
risk report
warnings/unknowns/blockers
dependency lock
split/leakage report
source card
model card
evaluation card
```

---

# 9. Most important novel problems to solve

These are the problems that would make Qortex genuinely novel.

## Problem 1: “Can this model scientifically run on this data?”

Most frameworks let users run wrong pipelines. Qortex should prevent that.

Output:

```text
No, because:
- source sampling rate is 512 Hz, model expects 256 Hz
- resampling is not allowed
- channel montage is missing Fz/Cz/Pz
- model card does not declare training preprocessing
```

## Problem 2: “What is the smallest valid data subset for my goal?”

Qortex already has minimum-download concepts. Expand it:

```text
For sleep staging:
- need EDF/BDF/FIF signal
- need events or annotations
- need subject-wise split
- need at least N subjects
- download these files only
```

MONAI does not solve this.

## Problem 3: “Which model should I use for this OpenNeuro dataset?”

Not model zoo browsing. Real matching.

```text
dataset → source profile → labels → task → model candidates → compatibility score → risk score → expected preprocessing → run plan
```

## Problem 4: “Did preprocessing invalidate the science?”

Every transform gets a risk explanation.

Example:

```text
Spatial resampling required.
Risk: lesion boundary may shift.
Mitigation: use nearest interpolation for masks, linear/B-spline for images.
Artifact: geometry ledger stores before/after affine.
```

## Problem 5: “Did the result preserve anatomical geometry?”

For every segmentation output:

```text
input affine == output affine?
mask shape matches?
orientation matches?
voxel spacing matches?
label map valid?
DICOM-SEG valid?
```

## Problem 6: “Can the result be reproduced exactly?”

Qortex lockfile must make rerun possible.

## Problem 7: “What can’t this dataset support?”

Negative capability report:

```text
This dataset cannot support supervised classification because no labels/events exist.
This dataset cannot support subject-generalization claims because only one subject exists.
This model cannot be used because its preprocessing contract is unknown.
```

---

# 10. Revised strategic stance

Do not market Qortex as:

```text
A MONAI alternative
```

Market it as:

```text
The AI-neuroscience operating layer above MONAI, OpenNeuro, BIDS, Braindecode, ONNX, Torch, nnU-Net, TotalSegmentator, and live neuro streams.
```

The internal engineering stance should be:

```text
Use MONAI wherever MONAI is mature.
Wrap MONAI wherever Qortex needs contracts and artifacts.
Extend MONAI wherever neuroscience/BIDS/OpenNeuro needs are missing.
Replace only unsafe/insufficient pieces.
Invent where no library has solved the problem.
```

---

# 11. Implementation priorities

## Phase 1 — MONAI parity integration

```text
1. MONAI transform backend
2. MONAI Bundle contract extractor
3. MONAI Model Zoo indexer
4. MONAI metrics wrapper
5. MONAI sliding-window inference config bridge
6. MONAI Deploy export bridge
7. MONAI Label import/export bridge
```

## Phase 2 — Qortex scientific superiority

```text
1. Geometry ledger
2. BIDS semantic engine
3. Model-source matchmaking
4. Scientific risk score
5. Preprocessing proof report
6. Artifact lockfile
7. Negative capability report
```

## Phase 3 — full AI-neuroscience platform

```text
1. Qortex Train
2. Qortex Eval
3. Qortex Deploy
4. Qortex Live
5. Qortex Model Registry
6. Qortex Dataset-Model Graph
7. Qortex Benchmark Zoo
```

---

# Final direction

To become the best, Qortex must not be a bigger MONAI. It must be the layer MONAI does not try to be:

```text
A scientifically strict, contract-driven, BIDS/OpenNeuro-native,
multi-modal NeuroAI runtime and audit system.
```

Build MONAI parity through integration. Build superiority through:

```text
compatibility compiler
BIDS semantic engine
geometry ledger
model-source matchmaking
risk scoring
negative capability reports
artifact lockfiles
real-time/replay NeuroAI runtime
cross-modal neuroscience support
```

That is the path where Qortex can be better than MONAI without wasting time copying what MONAI already does well.
