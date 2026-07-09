# Model Zoo Expansion — Design

Status: approved
Date: 2026-07-09

## 1. Goal

Expand Qortex's existing NeuroAI model registry from 13 curated entries to a
production-grade model zoo covering MONAI Model Zoo bundles, more Braindecode
models, the 2D vision (YOLO/SAM) family, promptable foundation segmentation
models, and additional external CLI segmentation engines — while preserving
Qortex's no-fabrication contract discipline (every field is `confirmed`,
`inferred`, or `unknown`; nothing is guessed).

Success condition: `qortex neuroai suggest-models` and `zoo list` surface the
expanded catalog with real (not fabricated) contracts; promptable models are
usable end-to-end via `prompt-predict`; new external engines follow the exact
pattern already proven by the nnU-Net/TotalSegmentator wrappers.

## 2. Existing architecture (do not duplicate)

- `neuroai/models/_base.py` — `ModelAdapter` ABC: `inspect()`,
  `required_input()`, `output_schema()`, `load(runtime)`, `predict(batch)`.
- `neuroai/models/_contracts.py` — curated `ModelContractEntry` registry with
  `lookup()`, `list_entries()`, `register()`, `get_model_card()`.
- `neuroai/models/_registry.py` — `make_model_adapter(spec)` factory,
  provider-string dispatch to adapter classes.
- `neuroai/models/zoo.py` — `backend_availability()`, real importlib probes
  of optional dependencies (unrelated to the new `zoo/` package below; kept
  as-is).
- `neuroai/models/{huggingface,onnx,torch,monai,braindecode,ultralytics,
  torchvision_adapter,keras_adapter,plugin}.py` — provider adapters.
- `neuroai/external.py` — CLI-based segmentation engine runner
  (`run_external_segmentation`, `_build_external_command` dispatch,
  `available_external_segmentation_engines()`), currently supporting
  `totalsegmentator` and `nnunet`.
- `neuroai/contracts.py` — `InputContract`, `OutputContract`, `ModelProfile`,
  `EvidenceStatus`.
- `neuroai/spec.py` — `ModelSpec`, `RuntimeSpec`.
- `cli/app.py` — `neuroai_app` Typer sub-app; commands registered with
  `@neuroai_app.command(...)`.

This spec extends these, it does not replace them.

## 3. Directory layout

```
src/qortex/neuroai/models/
  zoo/
    __init__.py            # imports each domain module, registers its entries
                            # into _contracts._REGISTRY via _contracts.register()
    eeg.py                  # Braindecode models
    vision.py                # YOLO/SAM 2D vision family
    medical_imaging.py       # MONAI segmentation bundles
    generative.py             # MONAI generative bundles
    foundation.py             # VISTA3D, MedSAM, SAM-Med3D (promptable)
    external_engines.py       # ExternalEngineEntry metadata (not ModelContractEntry)
  prompt.py                # Prompt, PromptType, PromptContract
  promptable.py             # PromptableModelAdapter ABC + VISTA3DAdapter,
                            # MedSAMAdapter, SAMMed3DAdapter
  cache.py                  # ModelCache manifest/provenance layer
```

`zoo/__init__.py` is imported once from `neuroai/models/__init__.py` (or lazily
on first `list_entries()`/`suggest-models` call) so the registry is fully
populated without every caller needing to know about the `zoo` subpackage.

## 4. Contract extensions

### 4.1 `InputContract` (in `contracts.py`)

Add one new optional field, default `None`, no impact on existing entries:

```python
prompt_contract: PromptContract | None = None
```

### 4.2 `prompt.py` (new)

```python
class PromptType(str, Enum):
    point = "point"
    box = "box"
    text = "text"

@dataclass
class Prompt:
    points: list[tuple[float, ...]] | None = None   # (x, y[, z])
    point_labels: list[int] | None = None             # 1=foreground, 0=background
    boxes: list[tuple[float, ...]] | None = None       # xyxy or xyzxyz
    text: str | None = None

class PromptContract(BaseModel if _PYDANTIC else object):
    supported_prompt_types: list[PromptType]
    max_points: int | None = None
    max_boxes: int | None = None
    supports_automatic_mode: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed
```

Each registry entry declares only the prompt types it actually supports —
VISTA3D/MedSAM/SAM-Med3D declare `point` + `box` (VISTA3D also
`supports_automatic_mode=True` for whole-organ segmentation without a
prompt); YOLO-World/YOLOE declare `text` only. No model is given a prompt
type it does not really support.

### 4.3 `promptable.py` (new)

```python
class PromptableModelAdapter(ModelAdapter):
    @abstractmethod
    def prompt_contract(self) -> PromptContract: ...

    @abstractmethod
    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput: ...

    def predict(self, batch: Any) -> ModelOutput:
        raise NotImplementedError(
            "This model requires a prompt — use predict_with_prompt()."
        )
```

- `VISTA3DAdapter(MONAIBundleAdapter, PromptableModelAdapter)` — VISTA3D is a
  MONAI bundle; subclass `MONAIBundleAdapter` for loading/inspection, add
  prompt-based inference on top.
- `MedSAMAdapter`, `SAMMed3DAdapter` — direct checkpoint loading (`torch.load`),
  prompt encoder + mask decoder inference. Not on the `transformers` pipeline
  API, so not routed through `HuggingFaceAdapter`.

`_registry.py` gains dispatch for providers `vista3d`, `medsam`, `sam_med3d`.

YOLO-World/YOLOE text-prompt support extends the existing `UltralyticsAdapter`
(`model.set_classes([...])`) — no new adapter class.

## 5. External CLI engine expansion

`external.py` gains, following the exact pattern of
`_build_totalsegmentator_command`/`_build_nnunet_command`:

- `_build_synthseg_command`
- `_build_synthstrip_command`
- `_build_hdbet_command`
- `_build_fastsurfer_command`
- `_build_tractseg_command`

Each is added to the `ExternalSegmentationEngine` literal and
`available_external_segmentation_engines()`. Executable-not-found is handled
by the existing `_require_executable` graceful-degrade path — no binaries
are required to write or unit-test this code (subprocess calls are mocked in
tests, exactly as the existing nnU-Net/TotalSegmentator tests do).

`zoo/external_engines.py` holds an `ExternalEngineEntry` metadata table
(engine name, executable, upstream URL, license, install hint) — kept
separate from `ModelContractEntry` because these tools have no in-process
tensor `InputContract`/`OutputContract`; representing them as
`ModelContractEntry` would fabricate a contract they don't have.

## 6. Cache / provenance layer

`cache.py` — a manifest on top of each backend's own cache (HF hub cache,
MONAI bundle dir), not a replacement downloader:

```python
@dataclass
class CacheEntry:
    model_id: str
    provider: str
    local_path: Path
    size_bytes: int
    sha256: str | None
    downloaded_at: str
    source_url: str | None

class ModelCache:
    def __init__(self, cache_dir: Path | None = None): ...  # default ~/.qortex/model_cache,
                                                              # override via QORTEX_CACHE_DIR
    def is_cached(self, model_id: str) -> bool: ...
    def record(self, entry: CacheEntry) -> None: ...
    def verify(self, model_id: str) -> bool: ...             # recompute sha256 vs manifest
    def list_cached(self) -> list[CacheEntry]: ...
    def disk_usage(self) -> int: ...
```

Manifest persisted as JSON at `{cache_dir}/manifest.json`. `zoo pull` calls
`adapter.load()` then `ModelCache.record()` on success.

## 7. CLI surface (`qortex neuroai ...`)

- `zoo list [--provider] [--modality] [--task] [--cached-only]` — table of
  `list_entries()` cross-referenced with `backend_availability()` and
  `ModelCache.is_cached()`.
- `zoo pull <model_id>` — resolve via registry, force `adapter.load()`,
  record into `ModelCache`.
- `prompt-predict <input> --model <id> [--point x,y,z ...] [--box ...]
  [--text ...]` — builds a `Prompt`, calls `predict_with_prompt`, writes
  output through the existing `nifti_out`/`overlay_out` writers (no new
  output format).

## 8. Registry content plan

### In scope

- **MONAI segmentation bundles**: `brats_mri_segmentation`,
  `wholeBrainSeg_Large_UNEST_segmentation`, `swin_unetr_btcv_segmentation`,
  `spleen_ct_segmentation`, `pancreas_ct_dints_segmentation`,
  `prostate_mri_anatomy`, `renalStructures_CECT_segmentation`,
  `renalStructures_UNEST_segmentation`, `retinalOCT_RPD_segmentation`,
  `ventricular_short_axis_3label`, `valve_landmarks`,
  `multi_organ_segmentation` (existing `wholeBody_ct_segmentation` stays).
- **MONAI generative bundles**: `brain_image_synthesis_latent_diffusion_model`,
  `brats_mri_generative_diffusion`,
  `brats_mri_axial_slices_generative_diffusion`, `maisi_ct_generative` —
  registered with `output_type="image_generation"`,
  `produces_probabilities=False` (a data convention on the existing free-form
  `output_type: str` field, not a schema change).
- **Braindecode**: BENDR, BIOT, LaBraM, USleep, AttnSleep, ATCNet, TIDNet,
  FBCNet, EEGITNet, EEGTCNet, ContraWR, SignalJEPA, DeepSleepNet.
- **2D vision**: YOLO11/YOLO26 (incl. `-seg` variants), YOLO-World, YOLOE,
  FastSAM, MobileSAM.
- **Promptable foundation segmentation**: VISTA3D, MedSAM, SAM-Med3D.
- **External CLI engines**: SynthSeg, SynthStrip, HD-BET, FastSurfer,
  TractSeg (metadata + `external.py` command builders).

Every new `ModelContractEntry` also carries `source_url`, `license`,
`paper_url` (optional), and `maintainer` fields (added to
`ModelContractEntry` in `_contracts.py`) — the single place to check/edit
provenance per model, matching the existing dataclass style.

### Explicitly out of scope (found during design, deliberately deferred)

- **FreeSurfer, fMRIPrep, Nilearn** — hours-long whole-pipeline BIDS-Apps,
  not single models; a different scope than a model zoo entry.
- **Roboflow** — dataset/annotation ingestion, belongs under
  `neuroai/sources/`, not `neuroai/models/`.
- **`hf_ct_chat`, `hf_llama3_vila_m3_*`** — multimodal chat/LLM bundles do not
  fit the tensor-shaped `InputContract`/`OutputContract` model; registering
  them would require fabricating a contract, which violates the existing
  no-guessing rule. Left out; revisit only if a concrete "conversational
  NeuroAI" use case emerges.

## 9. Testing

One offline registry self-check (no network, no weight downloads), matching
the fixture-based style of `test/project_21_neuroai_runtime`:

- Every `ModelContractEntry.provider` in the full registry resolves through
  `make_model_adapter`'s provider dispatch (raises neither `ValueError` for
  unknown provider nor requires network — `ImportError` for a missing
  optional dependency is acceptable and caught).
- Every populated `source_url` / `paper_url` is a well-formed URL
  (`urllib.parse.urlparse` scheme/netloc check).
- `PromptContract.supported_prompt_types` on every promptable entry is
  non-empty.

External CLI command builders (`_build_synthseg_command`, etc.) get the same
mocked-subprocess unit tests as the existing `_build_totalsegmentator_command`
tests — asserting the constructed argv, not a real binary run.

## 10. Non-goals

- No new output file formats — reuse `nifti_out`/`overlay_out`.
- No replacement of HF/MONAI's own download caching — `ModelCache` is a thin
  provenance layer on top.
- No text-prompt support claimed for models that don't have it (VISTA3D,
  MedSAM, SAM-Med3D stay point/box only).
