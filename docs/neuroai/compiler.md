# Compiler

`qortex.neuroai.compiler` is an **offline execution-plan compiler**. Given a
local source file/directory and a task, it profiles the source, scores every
matching model-zoo candidate against it, and emits a deterministic, hashable
execution plan as JSON.

!!! warning "Planner, not executor"
    The compiler does not download datasets, does not fetch model weights, and
    does not run inference. It reads local file headers and the model-zoo
    registry, then reports what *would* run and why. Use [Pipeline](pipeline.md)
    to actually execute a model.

## What it does

- Profiles the source: local file/directory existence, size, SHA-256, modality
  guess from suffix, and — for NIfTI and EEG files — real header geometry (see
  below). No voxel/sample data is loaded.
- Looks up every model-zoo entry (`qortex.neuroai.models.zoo.registry.list_entries`)
  that declares the requested `task`.
- Scores each entry against the source: license status, security/remote-code
  policy, runtime status, modality/geometry compatibility, and an estimated
  resource plan.
- Emits a `CompilationResult` with per-candidate blockers, warnings, and repair
  options, plus a stable `plan_hash` over the whole payload.

## CLI

```bash
qortex compile <source> --task <task> [options]
```

Real flags, from `compile_cmd` in `src/qortex/cli/app.py`:

| Flag | Default | Meaning |
|---|---|---|
| `source` (argument) | — | Dataset id, remote source, or local source path |
| `--task` | required | NeuroAI task the candidate models must declare, e.g. `segmentation`, `whole_brain_segmentation`, `foundation_segmentation` |
| `--device` | `cpu` | Target runtime device label, used only for the resource estimate |
| `--max-download-gb` | none | Blocks the plan if the acquisition estimate exceeds this |
| `--max-vram-gb` | none | Blocks a candidate if its estimated VRAM exceeds this |
| `--output` / `-o` | `execution-plan.json` | Where the plan JSON is written |
| `--accept-unknown-license-risk` | `false` | Explicitly accept an unresolved license instead of blocking |
| `--allow-remote-code` | `false` | Allow candidates that require `trust_remote_code` |
| `--require-open-license` / `--allow-restricted-license` | require | Block non-commercial, registration-only, or research-only licenses |
| `--include-plan-only` / `--runnable-only` | include | Include blocked/non-runnable candidates with their repair evidence, instead of only runnable ones |

### Real example

Run against a real, synthetic 64×64×32 NIfTI file created with `nibabel`
(the same construction `tests/test_neuroai_compiler_geometry.py` uses):

```bash
$ qortex compile demo_t1w.nii.gz --task whole_brain_segmentation \
    --accept-unknown-license-risk --output execution-plan.json
Compiled 3 candidate(s); runnable=false runnable_candidates=0 plan_hash=8f2f8bb8fdd28de63e80def78827cb526a8335fb017b2fcf87e67a18c6613f9a
Plan saved to execution-plan.json
```

The saved `source_profile` contains real header evidence, not a guess:

```json
{
  "source_type": "local_file",
  "modality": "mri",
  "spatial_shape": [64, 64, 32],
  "voxel_sizes_mm": [1.0, 1.0, 1.0],
  "orientation": "RAS",
  "evidence_status": "confirmed"
}
```

Every candidate in this run was non-runnable, for three different real
reasons: the MONAI candidate's compatibility status is `uncertain` (its input
contract's evidence status is `unknown`, so geometry cannot be fully proven
offline even though modality matches), and the two external-CLI candidates
(`external.fastsurfer`, `external.synthseg`) report a `requires_local_executable`
capability state with a blocker naming the missing executable (`run_fastsurfer.sh`,
`mri_synthseg`) because it is not on `PATH`.

## Python API

```python
from qortex.neuroai.compiler import CompilationRequest, compile_neuroai

result = compile_neuroai(CompilationRequest(
    source="demo_t1w.nii.gz",
    task="whole_brain_segmentation",
    device="cpu",
    accept_unknown_license_risk=True,
))

result.runnable          # bool: any candidate runnable AND acquisition has no blockers
result.plan_hash          # sha256 over the full canonical payload
result.candidates         # list[ModelCandidate]
result.source_profile     # SourceProfileSummary
result.save("execution-plan.json")
```

`profile_source(source)` can be called standalone to inspect a source without
compiling a plan:

```python
from qortex.neuroai.compiler import profile_source

profile = profile_source("demo_t1w.nii.gz")
profile.spatial_shape, profile.voxel_sizes_mm, profile.orientation
# (64, 64, 32), (1.0, 1.0, 1.0), "RAS"
```

## Source profiling

`profile_source()` (in `compiler.py`) never loads voxel or sample arrays. For
a local file it computes size and SHA-256, and additionally reads:

- **NIfTI** (`.nii`, `.nii.gz`, via `nibabel`): `spatial_shape` (first three
  header dims), `voxel_sizes_mm` (header zooms), `orientation` (`aff2axcodes`
  string like `"RAS"`).
- **EEG** (`.edf`, `.bdf`, `.set`, `.vhdr`, `.fif`, via `mne.io.read_raw(...,
  preload=False)`): `n_channels`, `sampling_rate_hz`, `duration_s`.

If the optional dependency (`nibabel`/`mne`) is missing, or the file fails to
parse, the compiler degrades gracefully: the geometry fields stay `None` and a
human-readable note is appended to `SourceProfileSummary.notes` instead of
raising. This is covered by
`tests/test_neuroai_compiler_geometry.py::test_profile_source_corrupted_nifti_degrades_without_raising`.

For a directory, the profile reports total size and the set of file suffixes
present, and classifies it as `local_bids_directory` when a
`dataset_description.json` is found, else `local_directory`. For anything that
is not a path that exists locally, the source is classified
`remote_or_catalog_source` with `evidence_status="unknown"` — the compiler does
not resolve OpenNeuro dataset IDs or other catalog references into files (see
[Known limitations](#known-limitations)).

## `CompilationResult` structure

```python
class CompilationResult:
    request: dict            # the CompilationRequest, serialized
    created_at: str
    source_profile: SourceProfileSummary
    evidence_graph: EvidenceGraph
    acquisition_plan: AcquisitionPlan
    candidates: list[ModelCandidate]
    runnable: bool            # any candidate runnable AND no acquisition blockers
    plan_hash: str             # sha256 of the canonical JSON payload
```

`SourceProfileSummary` carries the real geometry fields:

```python
source: str
source_type: str              # local_file | local_directory | local_bids_directory | remote_or_catalog_source
exists: bool
size_bytes: int | None
sha256: str | None
modality: str | None
available_suffixes: list[str]
evidence_status: EvidenceStatus
notes: list[str]
spatial_shape: tuple[int, ...] | None
voxel_sizes_mm: tuple[float, ...] | None
orientation: str | None
n_channels: int | None
sampling_rate_hz: float | None
duration_s: float | None
```

Each `ModelCandidate` carries: `capability_state`, `runnable`, `compatibility`
(`CompatibilityProof`), `geometry_plan` (`GeometryPlan`), `resource_plan`
(`ResourcePlan`), `license_report` (`LicenseReport`), `security_report`
(`SecurityReport`), `artifact_contract`, `repair_options`, `blockers`, and
`warnings`.

## How a candidate is scored

`build_candidates()` (in `candidates.py`) runs these checks per zoo entry that
declares the requested task, in this order:

1. **Runtime status** (`qortex.neuroai.models.zoo.status.runtime_status`).
   `blocked`, `checkpoint_unresolved`, `architecture_available`, and `unknown`
   all produce a blocker and, for the first two, a `RepairOption` (e.g.
   `resolve_checkpoint_contract`). Verified in
   `tests/test_neuroai_compiler.py::test_compile_marks_checkpoint_unresolved_promptable_models_unavailable`.
2. **License gate** (`qortex.neuroai.models.license.evaluate_license`).
   `blocked` always blocks. `unknown` blocks unless
   `--accept-unknown-license-risk` is set — this is the default behavior and
   is covered by
   `test_compile_blocks_unknown_license_by_default`. With
   `--require-open-license` (the default), `non_commercial_only`,
   `registration_required`, and `research_only` licenses are also blockers.
3. **Security/remote-code gate**. A candidate that requires
   `trust_remote_code` is blocked unless `--allow-remote-code` is passed (or
   the entry itself allows it). For `external_cli` entries, the declared
   executable is resolved with `shutil.which`; if missing, the candidate is
   blocked with a named `install_external_executable` repair. Covered by
   `test_compile_external_engine_records_missing_executable_requirement`.
4. **Compatibility** (`_compatibility`). Compares source modality against the
   model's declared modalities (`compatible` / `incompatible` / `uncertain`),
   and records the real header-geometry evidence (`spatial_shape`,
   `voxel_sizes_mm`, `orientation`, `n_channels`, `sampling_rate_hz`,
   `duration_s`) when present. `uncertain` — not `compatible` — is the result
   whenever the model's input contract evidence status is `unknown`, even if
   modality matches.
5. **Geometry plan** (`_geometry_plan`). Notes the confirmed source geometry
   and blocks if an external engine explicitly declares
   `geometry_preservation_known=False`.
6. **Resource plan** (`estimate_resource_plan`, in `resources.py`). Estimates
   VRAM from the model's declared `input_contract.spatial_shape` when every
   dimension is resolved and positive; otherwise falls back to the local file
   size as an `inferred`-evidence proxy. Blocks if the estimate exceeds
   `--max-vram-gb`.

A candidate's overall `capability_state` is `executable` only when it has no
blockers and its runtime is truly executable. `runnable` additionally requires
`compatibility.status == "compatible"` — so an `executable`-state candidate can
still be non-`runnable` if compatibility is only `uncertain` (see the real
example above).

## Acquisition plan

`build_acquisition_plan()` (in `acquisition.py`) is intentionally simple today:

- Local sources (`source_type` starting with `local_`) always report
  `required_download=False`, `estimated_download_gb=0.0`,
  `evidence_status="confirmed"`.
- Any other source type is treated as needing a download, with
  `estimated_download_gb` left `None` (or based on a locally-known size if one
  happens to be available) and `evidence_status="unknown"`. A note is always
  attached: *"Remote source size is not known without manifest inspection; no
  download is performed by compile."* `--max-download-gb` only blocks when a
  size estimate actually exists to compare against.

## Deterministic plan hash

`CompilationResult.build()` serializes `request`, `source_profile`,
`evidence_graph`, `acquisition_plan`, and `candidates` through
`serialization.canonical_json()` (sorted keys, fixed separators,
`ensure_ascii=True`) and hashes it with SHA-256 into `plan_hash`. Calling
`compile_neuroai()` twice with the same `CompilationRequest` produces the same
`plan_hash` and the same JSON — this is exactly what
`test_compile_plan_hash_is_stable_and_saved_json_contains_required_sections`
asserts, and it holds across separate process invocations since nothing
timestamp-dependent feeds the hash (the payload used for hashing excludes
`created_at`).

## Known limitations

Documented honestly, not aspirationally:

- **Acquisition planning is not connected to real OpenNeuro manifests.**
  Remote/catalog sources are reported with `evidence_status="unknown"` and no
  real size; there is no manifest fetch, no companion-file closure, and no
  actual bytes-to-download computation for anything that isn't already a local
  path.
- **There is no model ranking or `selected_model` field.** `candidates` is a
  sorted list (executable candidates first, then by blocker count, then by id)
  for readability, but the compiler does not pick or recommend a single model.
- **There is no `qortex execute` command.** A saved `execution-plan.json` is
  not consumed by any command yet — running a model still means using
  [Pipeline](pipeline.md) directly.
- **Compatibility checks stay `uncertain` whenever the model's input contract
  evidence is unknown**, even when header geometry was read successfully from
  the source. The compiler does not infer or guess an unconfirmed contract.
