# Qortex NeuroAI Model Zoo — Production Architecture

Status: approved for implementation
Date: 2026-07-09
Owner: Qortex Core
Scope: Qortex NeuroAI, not Qortex Atlas

## 0. Implementation progress

Live checklist — check an item the moment its code lands and is verified
(test passing / CLI smoke-checked), not before. Update this section in the
same commit as the code it tracks.

### Phase 1 — Registry hardening
- [x] `ZooEntry` schema (`zoo/schema.py`)
- [x] `LicenseInfo`
- [x] `SecurityPolicy`
- [x] `InteractionContract`
- [x] `ExternalEngineContract`
- [x] `zoo/registry.py` (register/list/lookup)
- [x] `zoo/validate.py` offline self-checks
- [x] CLI: `zoo list`
- [x] CLI: `zoo show`
- [x] CLI: `zoo validate`

### Phase 2 — MONAI integration
- [x] MONAI bundle extractor (`extractors/monai_bundle.py`)
- [x] P0 MONAI imaging entries (see §12.1 — list grows below as each lands)
- [x] MONAI generative entries (§12.5)
- [x] MONAI model cache recording
- [x] Compatibility bridge into `suggest-models`

### Phase 3 — Braindecode expansion
- [x] Braindecode extractor (`extractors/braindecode_model.py`)
- [x] Expanded Braindecode entries (§12.3 — list grows below)
- [x] Architecture vs. pretrained separation
- [ ] HF pretrained registry support — Deferred; requires confirmed HF repo IDs per model, not available offline (see zoo/braindecode_eeg.py module docstring)
- [x] EEG shape/channel/sampling contract validation

### Phase 4 — External engines
- [x] SynthSeg wrapper
- [x] SynthStrip wrapper
- [x] HD-BET wrapper
- [x] FastSurfer wrapper
- [x] TractSeg wrapper
- [ ] TotalSegmentator task discovery integration — Deferred; requires a `--json` output parser to extract tasks from `totalseg_info --json`, not just a command builder
- [x] External command provenance

### Phase 5 — Promptable segmentation
- [x] `Prompt` (`prompt.py`)
- [x] `InteractionContract` wired into adapters
- [x] `PromptableModelAdapter` (`promptable.py`)
- [x] VISTA3D prompt path
- [x] MedSAM adapter
- [x] SAM-Med3D adapter
- [x] `prompt-predict` CLI

### Phase 6 — Security, license, and artifacts
- [x] License gate (`license.py`)
- [x] Remote-code gate (`security.py`)
- [x] Executable allowlist
- [x] Model zoo artifact integration
- [x] Geometry ledger requirement — Implemented as file-level provenance (existence, size, sha256) in `model_zoo_entry.json`; NIfTI header-level geometry (shape/affine/voxel spacing) deferred because it would require adding `nibabel`, not currently a dependency.
- [x] Synthetic-data notice for generative models

**Model Zoo expansion (Phases 1-6): registry/security/artifact phase complete.**

Post-review hardening completed after the MONAI comparison:

- Promptable foundation entries with unresolved checkpoints now use
  `qortex_status="checkpoint_unresolved"` and are not runtime-executable
  claims.
- DICOM SEG/SR output adapters fail closed instead of writing `.npy`/JSON
  fallbacks under DICOM output types.
- MONAI bundle loading rejects unsafe ZIP paths, malformed JSON configs,
  state-dict mismatches, and sliding-window inference failures instead of
  silently falling back.

### NeuroAI Compiler — offline execution-plan compiler

- [x] Public compiler package: `qortex.neuroai.compiler`
- [x] Typed request/result models: `CompilationRequest`,
  `CompilationResult`, `ModelCandidate`, `CapabilityState`,
  `CompatibilityProof`, `LicenseReport`, `SecurityReport`, `ResourcePlan`,
  `AcquisitionPlan`, `EvidenceGraph`, and per-candidate artifact contracts.
- [x] CLI: `qortex compile <source> --task <task> --device <device>
  --max-download-gb <gb> --max-vram-gb <gb> --output execution-plan.json`
- [x] Truthful runtime states: `checkpoint_unresolved` and
  `architecture_available` entries compile as unavailable, not runnable.
- [x] License/security/resource/acquisition blockers are reflected in
  candidate `blockers`, `repair_options`, and result-level `runnable`.
- [x] Deterministic plan hashing over canonical JSON, excluding wall-clock
  timestamps from the hash.

Compiler boundary: compile is an offline planner. It inspects local source
paths, source file hashes, registry contracts, license metadata, security
policies, executable availability, runtime status, and coarse resource limits.
It does not download OpenNeuro manifests, fetch model weights, run model
inference, or invent missing compatibility facts. Remote/catalog sources are
therefore emitted with unknown acquisition evidence until a manifest layer is
explicitly invoked.

### Registry entries implemented so far

_(append one line per entry the moment it's registered and offline-validated — id, provider, phase)_

- `monai.brats_mri_segmentation` — provider `monai`, entry_type `model` (Phase 1 seed)
- `braindecode.EEGNet` — provider `braindecode`, entry_type `model` (Phase 1 seed)
- `external.totalsegmentator` — provider `external_cli`, entry_type `external_engine` (Phase 1 seed)
- `monai.wholeBrainSeg_Large_UNEST_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.vista3d` — provider `vista3d`, entry_type `promptable_model` (Phase 2, upgraded Phase 5 with `VISTA3DAdapter` and a confirmed `InteractionContract` for point/box prompts + automatic mode; `qortex_status="checkpoint_unresolved"` until bundle-specific prompted inference and geometry restoration are wired)
- `monai.swin_unetr_btcv_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.wholeBody_ct_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.spleen_ct_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.multi_organ_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.pancreas_ct_dints_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.prostate_mri_anatomy` — provider `monai`, entry_type `model` (Phase 2)
- `monai.renalStructures_CECT_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.renalStructures_UNEST_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.ventricular_short_axis_3label` — provider `monai`, entry_type `model` (Phase 2)
- `monai.valve_landmarks` — provider `monai`, entry_type `model` (Phase 2)
- `monai.retinalOCT_RPD_segmentation` — provider `monai`, entry_type `model` (Phase 2)
- `monai.brain_image_synthesis_latent_diffusion_model` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.brats_mri_generative_diffusion` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.brats_mri_axial_slices_generative_diffusion` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.maisi_ct_generative` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.cxr_image_synthesis_latent_diffusion_model` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.mednist_ddpm` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `monai.mednist_gan` — provider `monai`, entry_type `generative_model` (Phase 2 Task 3)
- `braindecode.Deep4Net` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.ShallowFBCSPNet` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.EEGConformer` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.BENDR` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.BIOT` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.Labram` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.REVE` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.USleep` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.AttnSleep` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.DeepSleepNet` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `braindecode.SignalJEPA` — provider `braindecode`, entry_type `model` (Phase 3 Task 2)
- `external.synthseg` — provider `external_cli`, entry_type `external_engine` (Phase 4)
- `external.synthstrip` — provider `external_cli`, entry_type `external_engine` (Phase 4)
- `external.hdbet` — provider `external_cli`, entry_type `external_engine` (Phase 4)
- `external.fastsurfer` — provider `external_cli`, entry_type `external_engine` (Phase 4)
- `external.tractseg` — provider `external_cli`, entry_type `external_engine` (Phase 4)
- `foundation.medsam` — provider `medsam`, entry_type `promptable_model` (Phase 5, point/box prompts only, `MedSAMAdapter`; `qortex_status="checkpoint_unresolved"` until a verified checkpoint resolver and real inference fixture exist)
- `foundation.sam_med3d` — provider `sam_med3d`, entry_type `promptable_model` (Phase 5, point/box prompts only, `SAMMed3DAdapter`; `qortex_status="checkpoint_unresolved"` until a verified checkpoint resolver and real inference fixture exist)

---

## 1. Product boundary

Qortex and Qortex Atlas are separate products.

Qortex is the NeuroAI execution and reproducibility layer:

- model registry
- model inspection
- source-model compatibility
- preprocessing compilation
- training / inference / replay / benchmark
- model execution adapters
- external engine wrappers
- artifact, provenance, risk, and reproducibility contracts

Qortex Atlas is the data intelligence and visual exploration layer:

- OpenNeuro / BIDS discovery
- dataset inspection
- modality / cohort / subject / file graph
- readiness scoring
- visual audit
- selective download planning
- dataset-to-model recommendation UI

The Model Zoo belongs to Qortex Core. Atlas may consume it for model
suggestions, but Atlas must not own model loading, model execution, or
runtime contracts.

## 2. Goal

Expand Qortex's NeuroAI model registry from a small curated list into a
production-grade, contract-validated model zoo covering:

- MONAI Model Zoo bundles
- Braindecode models and EEG foundation models
- promptable medical segmentation models
- 2D detection / segmentation models from Ultralytics
- external CLI neuroimaging and medical-imaging engines
- generative medical imaging bundles
- future multimodal medical VLMs, only when Qortex has a proper
  conversational / VLM contract

The zoo must not become a scraped directory of random models. It must be a
verified capability registry.

A model is allowed into the Qortex zoo only when Qortex can represent:

- source URL
- provider
- modality
- task
- input contract
- output contract
- preprocessing requirements
- interaction / prompt contract if applicable
- execution mode
- license
- evidence status
- provenance
- security requirements
- runtime risk
- artifact policy

Unknown fields are allowed. Fabricated fields are not allowed.

## 3. Source-of-truth policy

The registry uses official upstream sources first.

Primary sources:

- MONAI Model Zoo catalog
- MONAI Hugging Face organization
- MONAI model-zoo GitHub repository
- Braindecode documentation
- Hugging Face model repositories
- official model GitHub repositories
- official papers / arXiv preprints
- official CLI documentation

MONAI Model Zoo is a first-class registry source because each listed model
ships as a MONAI Bundle containing weights, training config, and inference
code in one reproducible unit.

Braindecode is a first-class EEG model source because its models share a
consistent PyTorch `nn.Module` interface, use standard EEG signal parameters
such as `n_chans`, `n_times`, `sfreq`, `chs_info`, and assume canonical EEG
tensors shaped as `(batch_size, n_chans, n_times)`.

TotalSegmentator is treated as an external engine, not an in-process tensor
model. It exposes task-level capabilities such as `total` with 117 CT
classes and `total_mr` with 50 MR classes, and provides task / class
discovery commands that Qortex can wrap without downloading model weights.

## 4. Non-negotiable invariants

1. No guessed contracts.
2. No fake model support.
3. No silent prompt support.
4. No automatic `trust_remote_code=True`.
5. No external CLI engine represented as a normal in-process tensor model.
6. No model without license metadata.
7. No model without source URL.
8. No model without evidence status.
9. No model without provider dispatch.
10. No registry entry that cannot be inspected offline.
11. No runtime execution without compatibility check.
12. No output without artifact provenance.
13. No segmentation output without geometry validation when spatial data is
    involved.
14. No foundation model claims without paper / upstream documentation.
15. No Roboflow model ingestion into Model Zoo. Roboflow belongs to
    dataset / annotation ingestion.

## 5. Existing architecture to preserve

Do not duplicate or replace these modules:

```text
src/qortex/neuroai/models/_base.py
src/qortex/neuroai/models/_contracts.py
src/qortex/neuroai/models/_registry.py
src/qortex/neuroai/models/zoo.py
src/qortex/neuroai/models/huggingface.py
src/qortex/neuroai/models/onnx.py
src/qortex/neuroai/models/torch.py
src/qortex/neuroai/models/monai.py
src/qortex/neuroai/models/braindecode.py
src/qortex/neuroai/models/ultralytics.py
src/qortex/neuroai/models/torchvision_adapter.py
src/qortex/neuroai/models/keras_adapter.py
src/qortex/neuroai/models/plugin.py
src/qortex/neuroai/external.py
src/qortex/neuroai/contracts.py
src/qortex/neuroai/spec.py
src/qortex/cli/app.py
```

The new design extends the existing provider / adapter / contract system. It
does not introduce a second registry.

## 6. Required directory layout

```text
src/qortex/neuroai/models/
  zoo/
    __init__.py
    registry.py
    schema.py
    sources.py
    validate.py

    monai_imaging.py
    monai_generative.py
    braindecode_eeg.py
    vision_2d.py
    foundation_segmentation.py
    external_engines.py
    watchlist.py

  prompt.py
  promptable.py
  cache.py
  license.py
  security.py
  provenance.py
  extractors/
    __init__.py
    monai_bundle.py
    braindecode_model.py
    huggingface_repo.py
    ultralytics_model.py
    external_cli.py
```

Purpose:

- `zoo/schema.py` defines the registry entry schema.
- `zoo/registry.py` registers and lists entries.
- `zoo/sources.py` stores source registries and URL metadata.
- `zoo/validate.py` runs offline self-checks.
- domain files register curated entries.
- `extractors/` derive contracts from real upstream metadata where possible.
- `license.py` normalizes licenses and enforces runtime gates.
- `security.py` blocks unsafe model loading patterns.
- `provenance.py` records per-field evidence and source references.

## 7. Corrected registry model

The current draft adds new fields directly to `ModelContractEntry`. That is
acceptable for small expansion, but not enough for a mature zoo. The
corrected model is a layered schema.

```python
class EvidenceStatus(str, Enum):
    confirmed = "confirmed"
    inferred = "inferred"
    unknown = "unknown"
    unsupported = "unsupported"
    contradicted = "contradicted"


class ExecutionMode(str, Enum):
    in_process = "in_process"
    external_cli = "external_cli"
    remote_api = "remote_api"
    bundle = "bundle"
    pipeline_app = "pipeline_app"


class ZooEntryType(str, Enum):
    model = "model"
    foundation_model = "foundation_model"
    external_engine = "external_engine"
    generative_model = "generative_model"
    promptable_model = "promptable_model"
    template = "template"
    watchlist = "watchlist"


class ProvenancedValue(BaseModel):
    value: Any
    evidence_status: EvidenceStatus
    source_url: str | None = None
    source_field: str | None = None
    checked_at: str | None = None
    note: str | None = None


class LicenseInfo(BaseModel):
    name: str | None = None
    url: str | None = None
    commercial_use: bool | None = None
    redistribution_allowed: bool | None = None
    requires_registration: bool = False
    requires_citation: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    notes: list[str] = []


class SecurityPolicy(BaseModel):
    trust_remote_code_required: bool = False
    allow_remote_code: bool = False
    requires_sandbox: bool = False
    allowed_imports: list[str] = []
    blocked_imports: list[str] = []
    executable_names: list[str] = []
    network_required_at_runtime: bool = False
    network_required_for_download: bool = False


class ZooEntry(BaseModel):
    id: str
    display_name: str
    entry_type: ZooEntryType
    provider: str
    execution_mode: ExecutionMode

    source_url: str
    paper_url: str | None = None
    model_url: str | None = None
    docs_url: str | None = None
    maintainer: str | None = None

    modality: list[str]
    task: list[str]

    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None
    preprocessing_contract: PreprocessingContract | None = None
    interaction_contract: InteractionContract | None = None
    external_engine_contract: ExternalEngineContract | None = None

    license: LicenseInfo
    security: SecurityPolicy

    evidence_status: EvidenceStatus
    provenance: dict[str, ProvenancedValue]

    qortex_status: str
    priority: str
    notes: list[str] = []
```

This separates model identity, execution mode, provenance, license, runtime
risk, and scientific contracts.

## 8. Contract extensions

### 8.1 Do not put prompt support inside `InputContract`

The earlier draft proposed `prompt_contract: PromptContract | None = None`
inside `InputContract`. Rejected: a prompt is not the same as the biomedical
input tensor. A prompt is an interaction constraint. It belongs in a
separate interaction contract.

Correct design:

```python
class PromptType(str, Enum):
    point = "point"
    box = "box"
    text = "text"
    mask = "mask"
    scribble = "scribble"
    class_label = "class_label"


class PromptCoordinateFrame(str, Enum):
    image_2d = "image_2d"
    voxel_3d = "voxel_3d"
    world_mm = "world_mm"
    normalized = "normalized"


class InteractionContract(BaseModel):
    supported_prompt_types: list[PromptType]
    prompt_coordinate_frame: PromptCoordinateFrame | None = None
    max_points: int | None = None
    max_boxes: int | None = None
    supports_negative_points: bool = False
    supports_multiclass_prompting: bool = False
    supports_automatic_mode: bool = False
    supports_iterative_refinement: bool = False
    requires_label_set: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed
```

Implication:

- VISTA3D can support automatic mode and interactive segmentation where
  confirmed.
- MedSAM and SAM-Med3D support prompt-driven segmentation.
- YOLO-World and YOLOE expose text-conditioned detection / segmentation
  through the Ultralytics adapter.
- Do not give text prompts to MedSAM or SAM-Med3D unless upstream explicitly
  supports them.

VISTA3D targets both automatic and interactive 3D medical segmentation and
reports support for 127 automatic classes in its paper
([arXiv:2406.05285](https://arxiv.org/abs/2406.05285)).

SAM-Med3D targets general-purpose volumetric medical segmentation using 3D
prompt points and was trained on 22K 3D images with 143K masks
([arXiv:2310.15161](https://arxiv.org/abs/2310.15161)).

MedSAM3 is watchlist-only until Qortex verifies code, weights, license, and
stable APIs; its paper claims text-promptable medical segmentation and says
code/model release is planned at the linked GitHub repository
([arXiv:2511.19046](https://arxiv.org/abs/2511.19046)).

### 8.2 External engines require capability contracts

External CLI tools do not expose normal tensor-shaped model contracts. Do
not force them into `ModelContractEntry`.

```python
class ExternalEngineContract(BaseModel):
    engine: str
    executable: str
    input_file_types: list[str]
    output_file_types: list[str]
    supported_modalities: list[str]
    supported_tasks: list[str]
    command_builder: str
    list_capabilities_command: list[str] | None = None
    output_manifest_supported: bool = False
    geometry_preservation_known: bool | None = None
    license_required: bool = False
    docker_supported: bool = False
    evidence_status: EvidenceStatus
```

TotalSegmentator should use this because it exposes a CLI, task list, class
list, output report, Docker command, and offline task metadata commands
([GitHub: wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)).

## 9. Adapter design

### 9.1 Base adapter

Keep:

```python
class ModelAdapter(ABC):
    def inspect(self) -> ModelProfile: ...
    def required_input(self) -> InputContract | None: ...
    def output_schema(self) -> OutputContract | None: ...
    def load(self, runtime: RuntimeSpec) -> None: ...
    def predict(self, batch: Any) -> ModelOutput: ...
```

### 9.2 Promptable adapter

```python
class PromptableModelAdapter(ModelAdapter):
    @abstractmethod
    def interaction_contract(self) -> InteractionContract:
        ...

    @abstractmethod
    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        ...

    def predict(self, batch: Any) -> ModelOutput:
        contract = self.interaction_contract()
        if contract.supports_automatic_mode:
            return self.predict_automatic(batch)
        raise ModelExecutionError(
            "This model requires an interaction prompt. Use predict_with_prompt()."
        )
```

### 9.3 External engine adapter

Do not pretend external engines are normal PyTorch models.

```python
class ExternalEngineAdapter:
    def inspect_engine(self) -> ExternalEngineContract: ...
    def check_executable(self) -> AvailabilityReport: ...
    def build_command(self, request: ExternalSegmentationRequest) -> list[str]: ...
    def run(self, request: ExternalSegmentationRequest) -> ExternalSegmentationResult: ...
    def parse_manifest(self, output_dir: Path) -> dict[str, Any] | None: ...
```

## 10. Provider dispatch

Provider strings must remain stable.

Required providers:

```text
monai_bundle
monai_generative
braindecode
ultralytics
medsam
sam_med3d
vista3d
torch
torchscript
onnx
huggingface
external_cli
plugin
custom
```

Provider dispatch must pass offline validation. It is acceptable for
optional dependencies to be unavailable. It is not acceptable for a provider
string to have no dispatch target.

## 11. Model extraction pipelines

### 11.1 MONAI Bundle extractor

Required file sources: MONAI Model Zoo catalog, Hugging Face model repo,
MONAI bundle config files, metadata, inference config, network definition,
label map, preprocessing graph, postprocessing graph.

Extractor outputs:

```python
class ExtractedMONAIContract(BaseModel):
    model_id: str
    bundle_version: str | None
    input_contract: InputContract | None
    output_contract: OutputContract | None
    preprocessing_contract: PreprocessingContract | None
    label_map: dict[str, int] | None
    sliding_window: dict[str, Any] | None
    unresolved_transforms: list[str]
    evidence: dict[str, ProvenancedValue]
```

Rules:

- If affine / spacing / orientation requirements are missing, mark them
  `unknown`.
- If MONAI transform graph contains custom callable transforms, mark the
  graph `partially_supported`.
- If model requires spacing/orientation but source lacks geometry metadata,
  compatibility must block or become `uncertain`, depending on the model's
  declared requirement.
- Do not run MONAI bundles with implicit preprocessing unless Qortex has
  compiled the transform plan.

MONAI's catalog provides enough bundle names and high-level input/output
descriptions for seed entries such as `brats_mri_segmentation`, `vista3d`,
`wholeBody_ct_segmentation`, and `wholeBrainSeg_Large_UNEST_segmentation`
([MONAI Model Zoo catalog](https://project-monai.github.io/model-zoo.html)).

### 11.2 Braindecode extractor

Extractor sources: Braindecode API docs, model class signatures, optional
Pydantic config if `braindecode[pydantic]` is available, Hugging Face
BrainDecode organization for pretrained weights.

Rules:

- Base input shape is `(batch, n_chans, n_times)`.
- Required fields: `n_chans`, `n_times` or `input_window_seconds`, `sfreq`
  where supported, `chs_info` where required.
- Pretrained model entries must use actual Hugging Face repo IDs.
- Architecture-only entries are allowed but must be marked as `template` or
  `untrained_architecture`.

Braindecode's documentation lists available models and states that several
pretrained models can be loaded through Hugging Face Hub integration,
including BIOT, CBraMod, CodeBrain, LaBraM, REVE, LUNA, BENDR, SignalJEPA,
and EEGPT ([Braindecode API reference](https://braindecode.org/stable/api.html)).

### 11.3 Ultralytics extractor

Extractor sources: Ultralytics official docs, model YAML / task metadata,
pretrained weight metadata if available.

Rules:

- YOLO models are 2D vision models.
- Use them for detection, instance segmentation, pose, OBB, and image-level
  classification.
- Do not treat YOLO as a volumetric MRI/CT model.
- For neuroimaging, YOLO entries are secondary: microscopy, endoscopy,
  screenshots, pathology patches, 2D slices, annotation bootstrapping.
- License gate is mandatory.

### 11.4 External CLI extractor

Extractor sources: executable discovery, `--help`, version command,
capability/list command where available, upstream docs, output manifest if
supported.

Rules:

- External engines must expose command builders and output parsers.
- Executable absence is not a unit-test failure.
- Runtime execution is integration-test only.
- CLI output geometry must be validated before Qortex writes a trusted
  artifact.

## 12. Registry content

### 12.1 P0 MONAI imaging bundles

```text
monai.brats_mri_segmentation
monai.wholeBrainSeg_Large_UNEST_segmentation
monai.vista3d
monai.swin_unetr_btcv_segmentation
monai.wholeBody_ct_segmentation
monai.spleen_ct_segmentation
monai.multi_organ_segmentation
monai.pancreas_ct_dints_segmentation
monai.prostate_mri_anatomy
monai.renalStructures_CECT_segmentation
monai.renalStructures_UNEST_segmentation
monai.ventricular_short_axis_3label
monai.valve_landmarks
monai.retinalOCT_RPD_segmentation
```

Rationale: `brats_mri_segmentation` is canonical brain tumor segmentation
from T1/T1c/T2/FLAIR; `wholeBrainSeg_Large_UNEST_segmentation` is T1w
whole-brain structural segmentation with 133 structures; `vista3d` is
foundation-style 3D CT segmentation and annotation;
`swin_unetr_btcv_segmentation` is a transformer CT segmentation baseline;
`wholeBody_ct_segmentation` is broad CT anatomy segmentation; the smaller
MONAI bundles are useful for tests, demos, and integration coverage.

### 12.2 P0 external engines

```text
external.nnunet_v2
external.totalsegmentator
external.synthseg
external.synthstrip
external.hdbet
external.fastsurfer
external.tractseg
```

Rationale: nnU-Net remains a mandatory biomedical segmentation baseline —
the 2024 "nnU-Net Revisited" paper argues that strong U-Net variants inside
the nnU-Net framework, scaled and validated rigorously, remain a
state-of-the-art recipe
([arXiv:2404.09556](https://arxiv.org/abs/2404.09556)). TotalSegmentator
provides broad CT/MR anatomical segmentation tasks and has machine-readable
task/class discovery Qortex can exploit for capability inspection. SynthSeg,
SynthStrip, HD-BET, FastSurfer, and TractSeg are practical neuroimaging
engines and should be wrapped as external CLI engines, not reimplemented.

### 12.3 P0 EEG / biosignal entries

```text
braindecode.EEGNet
braindecode.Deep4Net
braindecode.ShallowFBCSPNet
braindecode.EEGConformer
braindecode.BENDR
braindecode.BIOT
braindecode.Labram
braindecode.REVE
braindecode.USleep
braindecode.AttnSleep
braindecode.DeepSleepNet
braindecode.SignalJEPA
```

Register these as two classes: architecture entries (usable for training)
and pretrained entries (usable only when a real upstream checkpoint is
known).

LaBraM is a high-priority EEG foundation model because it targets generic
EEG representations through large-scale pretraining and reports pretraining
on about 2,500 hours of EEG from around 20 datasets
([arXiv:2405.18765](https://arxiv.org/abs/2405.18765)). REVE is a
high-priority watchlist/pretrained entry because it targets arbitrary EEG
setups and reports pretraining on over 60,000 hours of EEG from 92 datasets
and 25,000 subjects
([arXiv:2510.21585](https://arxiv.org/abs/2510.21585)).

### 12.4 P0 promptable segmentation

```text
foundation.vista3d
foundation.medsam
foundation.sam_med3d
```

VISTA3D is both a MONAI bundle and a promptable/foundation segmentation
entry. Use one canonical entry ID with two capabilities instead of
duplicate entries:

```yaml
id: monai.vista3d
entry_type: promptable_model
provider: vista3d
execution_mode: bundle
capabilities:
  - automatic_segmentation
  - interactive_segmentation
  - zero_shot_target_adaptation
```

MedSAM and SAM-Med3D should use direct promptable adapters, not generic
Hugging Face pipelines, unless the specific checkpoint is exposed through a
stable HF-compatible interface.

Current implementation status: these three promptable entries are
contract-inspectable but not production-executable. Their
`InteractionContract` is usable for validation and model selection, but
runtime prompted inference remains blocked until Qortex has verified
checkpoint resolution, preprocessing, prompt-coordinate transforms, output
geometry restoration, and real end-to-end fixtures.

### 12.5 P1 MONAI generative bundles

```text
monai.brain_image_synthesis_latent_diffusion_model
monai.brats_mri_generative_diffusion
monai.brats_mri_axial_slices_generative_diffusion
monai.maisi_ct_generative
monai.cxr_image_synthesis_latent_diffusion_model
monai.mednist_ddpm
monai.mednist_gan
```

Generative models must not be treated as segmentation/classification models:

```yaml
task:
  - image_generation
  - synthesis
output_type: synthetic_image
clinical_use: prohibited
research_use: allowed
artifact_policy:
  watermark_synthetic: true
  require_generation_metadata: true
  require_prompt_or_condition_record: true
```

MAISI is important because MONAI describes it as a diffusion-based model for
synthetic 3D CT with anatomical control and outputs up to 512×512×768 voxels
conditioned on organ segmentations.

### 12.6 P2 Ultralytics / 2D vision

```text
ultralytics.yolo11_detect
ultralytics.yolo11_segment
ultralytics.yolo26_detect
ultralytics.yolo26_segment
ultralytics.yolo_world
ultralytics.yoloe
ultralytics.fastsam
ultralytics.mobilesam
```

Use cases: 2D medical frame analysis, pathology patches, endoscopy frames,
microscopy, screenshots / UI visual QA, annotation bootstrapping, 2D slice
experiments. Do not use these as primary volumetric neuroimaging models.

### 12.7 Explicitly excluded from Model Zoo

```text
Roboflow
FreeSurfer
fMRIPrep
Nilearn
MNE raw pipelines
general LLM/VLM chat models
unverified GitHub repos
paper-only models without code/weights
```

Reason: Roboflow is dataset / annotation ingestion. FreeSurfer and fMRIPrep
are whole-pipeline BIDS-App style systems, not model entries. Nilearn and
MNE are analysis/preprocessing libraries, not model zoo entries. General
chat/VLM models require a separate conversational NeuroAI contract.
Paper-only models belong in `watchlist.py`.

## 13. Required entry examples

### 13.1 MONAI BraTS

```yaml
id: monai.brats_mri_segmentation
display_name: BraTS MRI Segmentation
entry_type: model
provider: monai_bundle
execution_mode: bundle
source_url: https://huggingface.co/MONAI/brats_mri_segmentation
catalog_url: https://project-monai.github.io/model-zoo.html
maintainer: Project MONAI
priority: P0

modality:
  - mri
task:
  - segmentation
  - brain_tumor_segmentation

input_contract:
  modality: mri
  spatial_dims: 3
  required_channels:
    - T1
    - T1c
    - T2
    - FLAIR
  voxel_sizes_mm:
    value: [1.0, 1.0, 1.0]
    evidence_status: confirmed

output_contract:
  output_type: segmentation_mask
  labels:
    - tumor_core
    - whole_tumor
    - enhancing_tumor

license:
  evidence_status: unknown
  requires_manual_check: true

security:
  trust_remote_code_required: false
  network_required_for_download: true
  network_required_at_runtime: false

qortex_status: runnable_after_contract_validation
```

### 13.2 Braindecode EEGNet

```yaml
id: braindecode.EEGNet
display_name: EEGNet
entry_type: model
provider: braindecode
execution_mode: in_process
source_url: https://braindecode.org/stable/generated/braindecode.models.EEGNet.html
priority: P0

modality:
  - eeg
task:
  - classification
  - eeg_decoding
  - bci

input_contract:
  modality: eeg
  axis: batch_channels_time
  required_metadata:
    - n_chans
    - n_times
  sampling_rate_hz:
    evidence_status: runtime_required

output_contract:
  output_type: class_logits
  produces_probabilities: false

qortex_status: architecture_available
```

### 13.3 TotalSegmentator external engine

```yaml
id: external.totalsegmentator
display_name: TotalSegmentator
entry_type: external_engine
provider: external_cli
execution_mode: external_cli
source_url: https://github.com/wasserth/TotalSegmentator
priority: P0

modality:
  - ct
  - mri
task:
  - anatomical_segmentation

external_engine_contract:
  executable: TotalSegmentator
  supported_tasks:
    - total
    - total_mr
  list_capabilities_command:
    - totalseg_info
    - --json
  output_file_types:
    - nifti
    - json
  output_manifest_supported: true

qortex_status: runnable_if_executable_available
```

## 14. CLI surface

Add these commands under `qortex neuroai`.

```text
qortex neuroai zoo list
qortex neuroai zoo show <model_id>
qortex neuroai zoo validate
qortex neuroai zoo validate-entry <model_id>
qortex neuroai zoo pull <model_id>
qortex neuroai zoo cache list
qortex neuroai zoo cache verify <model_id>
qortex neuroai zoo sources
qortex neuroai zoo licenses
qortex neuroai zoo risks
```

Filtering:

```text
--provider
--modality
--task
--entry-type
--priority
--runnable-only
--cached-only
--promptable-only
--external-only
--license-safe-only
--evidence confirmed|inferred|unknown
```

Prompt inference:

```text
qortex neuroai prompt-predict input.nii.gz \
  --model monai.vista3d \
  --point 42,80,31 \
  --point-label foreground \
  --output out/
```

External engine run:

```text
qortex neuroai run-external-segmentation input.nii.gz \
  --engine totalsegmentator \
  --task total_mr \
  --output out/
```

Model recommendation:

```text
qortex neuroai suggest-models \
  --source /data/ds000117 \
  --goal "T1w whole-brain segmentation" \
  --require-compatible \
  --explain-blockers
```

## 15. Cache and provenance

`ModelCache` is not a downloader. It is a provenance layer above backend
caches.

Default: `~/.qortex/model_cache`. Environment override: `QORTEX_CACHE_DIR`.

Manifest:

```json
{
  "schema_version": "1.0",
  "entries": [
    {
      "model_id": "monai.brats_mri_segmentation",
      "provider": "monai_bundle",
      "local_path": "...",
      "size_bytes": 0,
      "sha256": null,
      "downloaded_at": "2026-07-09T00:00:00Z",
      "source_url": "https://huggingface.co/MONAI/brats_mri_segmentation",
      "backend_cache": "huggingface_hub",
      "verified": false
    }
  ]
}
```

Required methods:

```python
class ModelCache:
    def is_cached(self, model_id: str) -> bool: ...
    def record(self, entry: CacheEntry) -> None: ...
    def verify(self, model_id: str) -> bool: ...
    def list_cached(self) -> list[CacheEntry]: ...
    def disk_usage(self) -> int: ...
    def remove(self, model_id: str) -> None: ...
```

## 16. Security policy

### 16.1 Remote code

Default: `allow_remote_code: false`. Any entry requiring remote code must be
blocked unless the user explicitly enables it:

```text
Blocked:
Model requires remote Python code execution.
Use --allow-remote-code only in a trusted environment.
```

### 16.2 External executable execution

External engines must: use allowlisted executable names; use list-based
argv construction; never use shell string execution; validate input/output
paths; write command manifest; capture stdout/stderr; capture executable
version; record return code; fail if output is missing; validate output
geometry when spatial.

### 16.3 Licenses

License must be a runtime gate.

States: `safe_for_open_use`, `research_only`, `non_commercial_only`,
`registration_required`, `unknown`, `blocked`.

If license is unknown, `qortex neuroai zoo pull <model>` may download only
if the user passes `--accept-unknown-license-risk`. Execution should still
mark artifact risk.

## 17. Compatibility and suggestion logic

`zoo list` shows entries. `suggest-models` ranks entries.

Ranking must use: modality match, task match, axis match, channel match,
sampling-rate match, spatial-shape match, voxel-spacing match,
orientation / coordinate frame match, required metadata availability, label
availability, preprocessing feasibility, prompt availability, license
acceptability, backend availability, cache availability, evidence
confidence, runtime cost, known blockers.

Output example:

```text
1. monai.wholeBrainSeg_Large_UNEST_segmentation
   status: compatible_with_transforms
   required transforms:
     - reorient RAS
     - resample_spatial 1mm
   risk: medium
   reason: T1w MRI source matches model modality and task, but spacing differs.

2. external.fastsurfer
   status: compatible
   execution: external_cli
   risk: low
   reason: T1w MRI source and whole-brain segmentation goal match.

3. monai.brats_mri_segmentation
   status: blocked
   reason: model requires T1/T1c/T2/FLAIR; source contains only T1w.
```

## 18. Artifact requirements

Every zoo-driven run must write:

```text
artifact_manifest.json
artifact_contract.json
model_zoo_entry.json
model_source_provenance.json
license_report.json
security_report.json
compatibility_report.json
preprocess_plan.json
geometry_ledger.json
runtime_report.json
latency_report.json
warnings.json
```

External engine runs additionally write:

```text
external_command.json
external_stdout.txt
external_stderr.txt
external_version.txt
external_output_manifest.json
```

Promptable runs additionally write:

```text
prompt_contract.json
prompt_used.json
interaction_trace.json
```

Generative runs additionally write:

```text
generation_conditions.json
synthetic_data_notice.json
```

## 19. Testing plan

### 19.1 Offline unit tests

No network. No downloads. No real external binaries.

```text
test_zoo_registry_imports
test_all_entry_ids_unique
test_all_provider_strings_dispatch
test_all_source_urls_parse
test_all_entries_have_license_info
test_all_entries_have_evidence_status
test_promptable_entries_have_interaction_contract
test_external_entries_have_external_engine_contract
test_external_command_builders_use_argv_lists
test_cache_manifest_roundtrip
test_license_gate_unknown_blocks_by_default
test_remote_code_blocks_by_default
```

### 19.2 Optional live tests

Network allowed. Marked separately.

```text
test_monai_hf_repo_exists
test_braindecode_docs_entry_exists
test_external_engine_help_if_installed
test_totalseg_info_json_if_installed
test_medsam_checkpoint_metadata_if_available
```

### 19.3 Integration tests

Small fixtures only.

```text
test_monai_spleen_bundle_inspect
test_braindecode_eegnet_construct
test_ultralytics_model_inspect_if_installed
test_prompt_object_validation
test_external_totalseg_command_builder
```

### 19.4 Golden registry test

A frozen JSON snapshot must be generated at
`tests/golden/model_zoo_registry.v1.json`. CI checks that accidental
registry changes are explicit.

## 20. Implementation phases

See §0 for the live checklist. Phase definitions:

1. **Registry hardening** — `ZooEntry`, `LicenseInfo`, `SecurityPolicy`,
   `InteractionContract`, `ExternalEngineContract`, registry loading,
   offline validation, CLI `zoo list`/`zoo show`/`zoo validate`.
2. **MONAI integration** — MONAI bundle extractor, P0 MONAI entries, MONAI
   generative entries, MONAI model cache recording, compatibility bridge
   into existing `suggest-models`.
3. **Braindecode expansion** — expanded Braindecode entries,
   architecture vs. pretrained separation, HF pretrained registry support,
   EEG shape/channel/sampling contract validation.
4. **External engines** — SynthSeg, SynthStrip, HD-BET, FastSurfer,
   TractSeg wrappers; TotalSegmentator task discovery integration; external
   command provenance.
5. **Promptable segmentation** — `Prompt`, `InteractionContract`,
   `PromptableModelAdapter`, VISTA3D prompt path, MedSAM adapter, SAM-Med3D
   adapter, `prompt-predict` CLI.
6. **Security, license, and artifacts** — license gate, remote-code gate,
   executable allowlist, model zoo artifact integration, geometry ledger
   requirement, synthetic data notice for generative models.

## 21. Deferred work

Do not implement these in this phase:

```text
Roboflow ingestion
FreeSurfer/fMRIPrep BIDS-App orchestration
Nilearn/MNE analysis workflow registry
chat/VLM conversational NeuroAI
automatic scraping of arbitrary Hugging Face models
automatic scraping of arbitrary GitHub repositories
clinical deployment claims
```

These require separate contracts.

## 22. Final success criteria

The implementation is acceptable only when:

1. `qortex neuroai zoo list` shows MONAI, Braindecode, promptable, vision,
   and external-engine entries.
2. `qortex neuroai zoo validate` passes offline without network.
3. `qortex neuroai suggest-models` uses the expanded registry.
4. No entry fabricates unavailable contract fields.
5. Every unknown field is explicitly marked `unknown`.
6. Every model has license metadata.
7. Every promptable model has an interaction contract.
8. Every external engine has an external engine contract.
9. Every provider string resolves through adapter dispatch.
10. Every run writes model-zoo provenance into the Qortex artifact.
11. Remote code is blocked by default.
12. Unknown license execution is blocked or explicitly acknowledged.
13. Spatial outputs require geometry validation.
14. Generative outputs are marked synthetic.
15. The registry can be exported as deterministic JSON.

## 23. Strategic rule

Qortex Model Zoo is not a model list. It is a contract-validated execution
registry for NeuroAI.

The model zoo must answer: Can this model run on this source? What does it
require? What does it output? What preprocessing is scientifically
required? What is unknown? What is unsafe? What is blocked? What will be
written into the artifact? Can the result be reproduced?

Anything that cannot answer these questions does not belong in the
production zoo.

## 24. Current validation record — 2026-07-10

The current implementation has been validated beyond registry existence:

- Public API, CLI, and installed wheel import paths work.
- Real OpenNeuro manifest, metadata, event-table, NIfTI-header, catalog,
  Dataset facade, conversion, EDA/QC, visualization, and decision-workflow
  scenarios pass through `python test/run_all.py` with 43/43 projects passing.
- The NeuroAI runtime scenario writes and validates real artifact files:
  compatibility report, preprocessing plan, provenance, runtime report,
  predictions CSV/JSONL, marker records, and artifact manifest.
- The compiler profiles a real OpenNeuro T1w NIfTI header, writes a deterministic
  execution plan, and `qortex execute` verifies plan hash and source SHA drift.
- Focused correctness lint over `src/qortex/neuroai`, `src/qortex/cli/app.py`,
  and NeuroAI tests passes for `F` and `E9` classes.
- Validation also corrected two concrete implementation defects outside the
  model-zoo core: readiness type resolution for `LogicalRecording`, and MSD
  Brain `seed` propagation into the MONAI loading path.

This record does not convert unresolved checkpoint entries into runnable
entries. Entries without verified checkpoints, executables, licenses, or
security permission remain non-runnable by design.
