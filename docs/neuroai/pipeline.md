# Pipeline

`Pipeline` is the top-level object for the NeuroAI runtime. It wraps the full check → plan → load → run → artifact sequence and exposes each step as an explicit method.

## YAML format

A pipeline YAML has five sections: `source`, `window`, `model`, `outputs`, and `runtime`. All keys are documented below.

```yaml
name: eeg_classifier          # used in provenance and artifact filenames

source:
  type: bids                  # see Sources page for all types
  path: data/ds004130
  modality: eeg
  subject: "01"               # optional — alias for subjects: ["01"]
  session: null               # optional — alias for sessions
  task: null                  # optional

window:
  duration_s: 4.0             # also accepts duration: "4s"
  step_s: 2.0                 # also accepts step: "2s" or "500ms"
  overlap_frac: 0.0
  drop_short: true

model:
  provider: huggingface       # huggingface | onnx | torch | torchscript | braindecode | monai | ultralytics | plugin
  id: braindecode/EEGNet      # HF repo ID, local path, or model identifier
  task: eeg_classification    # task hint passed to the model adapter
  revision: main              # HF revision (branch, tag, commit)
  trust_remote_code: false    # must be true for plugin provider

preprocessing:
  mode: auto                  # auto | explicit | none
  allow: [resample, channel_select, cast_dtype, to_tensor]
  deny: []                    # explicit transform names to forbid
  normalize: false            # boolean gate for normalize
  resample: true              # boolean gate for resample and resample_spatial
  channel_select: true        # boolean gate for channel_select

outputs:
  - type: jsonl
    path: predictions.jsonl
    append: false
  - type: lsl_marker
    stream_name: qortex_markers
  - type: parquet
    path: predictions.parquet

runtime:
  device: cpu                 # cpu | cuda | cuda:0 | mps
  fp16: false                 # half precision — cuda only
  latency_budget_ms: 50.0    # warning threshold for p95 latency
  optimize: safe              # safe | speed | memory
  batch_size: 1
  num_workers: 0
  cache_model: true
```

`SourceSpec.from_dict()` accepts `subject`/`session` as scalar aliases and
normalizes them to `subjects`/`sessions` lists. `WindowSpec.from_dict()` accepts
both user-facing keys (`duration_s`, `step_s`, `overlap_frac`) and serialized
keys (`duration`, `step`, `overlap`). String durations such as `"2s"` and
`"500ms"` are parsed to seconds.

Boolean fields are parsed deliberately. `"false"`, `"0"`, `"no"`, and `"off"`
mean `False`; `"true"`, `"1"`, `"yes"`, and `"on"` mean `True`. Invalid boolean
strings fail validation instead of relying on Python truthiness.

## Validation

`PipelineSpec.validate()` returns a list of error strings before anything is loaded. The CLI runs it automatically; call it explicitly in Python before `check()`:

```python
errors = pipe._spec.validate()
if errors:
    for e in errors:
        print(e)
```

Checks performed:
- `source.type` is present
- `model.id` and `model.provider` are present
- `model.provider` is a known value (`huggingface`, `onnx`, `torch`, `torchscript`, `monai`, `braindecode`, `ultralytics`, `custom`, `plugin`)
- `trust_remote_code=True` is flagged as a security warning
- At least one output is declared
- `window.duration_s > 0` and `window.step_s > 0` (when provided)
- `window.step_s <= window.duration_s`
- All values in `preprocessing.allow` and `preprocessing.deny` are valid `TransformKind` names
- `preprocessing.allow` and `preprocessing.deny` do not contain the same transform
- Boolean gates do not contradict `allow` (`normalize=False` cannot allow `normalize`)
- `trigger.when.probability_gte` is numeric and between 0 and 1
- `trigger.when.stable_for` is a positive integer

`Pipeline.from_yaml()` and `Pipeline.from_dict()` raise
`qortex.core.ContractValidationError` when parsing or validation fails. The
exception carries `code="contract.validation_failed"` and a structured
`context["violations"]` list for CLI, notebook, or service error handling.

## Python API

### Construction

```python
from qortex.neuroai import Pipeline

pipe = Pipeline.from_yaml("pipeline.yaml")

# Or from a dict:
pipe = Pipeline.from_dict({
    "name": "ad_hoc",
    "source": {"type": "local_file", "path": "data.edf"},
    "model": {"provider": "huggingface", "id": "braindecode/EEGNet"},
    "outputs": [{"type": "jsonl", "path": "out.jsonl"}],
})
```

### `check()` — compatibility gate

```python
report = pipe.check()
```

`check()` probes the source (header only, no data loaded) and inspects the model (config.json only, no weights). It returns a `CompatibilityReport` with:

- `status`: `compatible` | `compatible_with_transforms` | `uncertain` | `incompatible`
- `is_runnable`: bool — True when status is `compatible` or `compatible_with_transforms`
- `required_transforms`: list of transforms the runtime will apply automatically
- `blockers`: list of hard incompatibilities (wrong modality, incompatible shape, missing dependency)
- `warnings`: non-fatal issues (unknown channel labels, inferred sampling rate)
- `unknowns`: fields that could not be determined without loading data
- `explain()`: structured source-vs-model comparison rows
- `to_markdown()` / `to_json()`: exportable compatibility evidence

```python
print(report.summary())
# CompatibilityReport: COMPATIBLE_WITH_TRANSFORMS
#   Required transforms (1):
#     • cast_dtype(from=float64, to=float32)  [irreversible]

for b in report.blockers:
    print(b.code, b.message)

if not report.is_runnable:
    # Do not call run() — it will raise RuntimeExecutionError.
    pass
```

The distinction between `uncertain` and `incompatible` matters: `uncertain` means a required property (e.g., model input shape) could not be read from the config — the pipeline might work, but `check()` cannot guarantee it. `incompatible` means a hard blocker was found (modality mismatch, insufficient channels, etc.).

Denied required transforms are hard blockers. For example, if the source is
`float64`, the model contract requires `float32`, and `preprocessing.deny`
contains `cast_dtype`, the report is `incompatible` with a `DTYPE_MISMATCH`
blocker. The runtime will not silently cast against policy.

Scientific geometry checks also fail closed. Declared model `intensity_range`
is compared with explicit source `intensity_range` / `value_range` metadata and
requires `rescale_intensity` when mismatched. Voxel-spacing mismatch must be
handled by an allowed concrete `resample_spatial` plan or the run is blocked.
Coordinate-frame mismatches are handled symmetrically with `reorient`; denied or
unsupported orientation pairs are blockers. fMRI TR mismatch is incompatible
until an explicit temporal-resampling transform exists.

Missing required channels are not mapped heuristically. `channel_map` is planned
only when the pipeline declares an executable mapping:

```yaml
preprocessing:
  allow: [channel_map]
  channel_map:
    Cz: C3     # target required by model -> available source channel
```

Without the explicit map, missing model-required channels are blockers.

### Explicit Model Contracts

Any provider can receive a universal contract override. This is required for
raw PyTorch checkpoints and useful for research models whose config files do
not carry reliable neuro/medical metadata:

```yaml
model:
  provider: torch
  id: model.pt
  task: eeg_classification
  input_contract:
    modality: eeg
    axis_convention: batch_channels_time
    n_channels: 22
    sampling_rate_hz: 250
    window_duration_s: 4
    dtype: float32
  output_contract:
    output_type: classification
    classes: [left_hand, right_hand]
    n_classes: 2
    produces_probabilities: true
```

### `plan_preprocessing()` — inspect the transform chain

```python
plan = pipe.plan_preprocessing()  # calls check() if not already done

for t in plan.transforms:
    print(f"{t.kind.value}: {t.required_by} (reversible={t.reversible})")
```

Each `TransformDescriptor` has:
- `kind`: `TransformKind` enum (`resample`, `channel_select`, `channel_reorder`, `channel_map`, `bandpass`, `normalize`, `window`, `cast_dtype`, `rescale_intensity`, `reorient`, `resample_spatial`, `pad_or_crop`, `add_batch_dim`, `add_channel_dim`, `transpose_axes`, `to_tensor`)
- `required_by`: why this transform is in the plan
- `params`: dict of transform parameters
- `reversible`: whether the transform can be undone
- `evidence_status`: `EvidenceStatus` for the requirement

### Transform execution semantics

The `TransformExecutor` that `run()` and `benchmark()` use applies each planned
transform with behaviour chosen for scientific correctness and streaming
throughput:

- **`pad_or_crop` is centre-aligned.** When the source volume is larger than
  the model's required spatial shape the crop is taken from the centre, and
  when it is smaller the padding is distributed symmetrically. A corner-aligned
  crop would silently discard one side of the anatomy (e.g. the top slices of a
  brain); centre alignment keeps the field of view centred.
- **`resample` uses a rational approximation of the true rate ratio.** The
  ratio `to_hz / from_hz` is converted with `Fraction(...).limit_denominator()`
  before `scipy.signal.resample_poly`, so fractional acquisition rates such as
  `512.03 Hz → 256 Hz` are resampled accurately instead of being rounded to the
  nearest integer Hz first.
- **`bandpass` filter coefficients are cached.** The Butterworth SOS design
  depends only on `(low_hz, high_hz, sfreq, order)`, so it is memoised on the
  executor and designed once per streaming session rather than once per window.
- **`normalize: exponential_moving_standardize`** is vectorised across channels
  (numerically identical to the per-sample recurrence) so per-window cost scales
  with the number of time steps, not channels × time steps.
- **Critical transforms fail loud.** `resample`, `resample_spatial`, `reorient`,
  `normalize`, `rescale_intensity`, `cast_dtype`, `bandpass`, `channel_select`,
  `channel_map`, `channel_reorder`, `pad_or_crop`, and `transpose_axes` raise
  `TransformError` on failure instead of passing data through unchanged. Only
  purely structural transforms (`add_batch_dim`, `add_channel_dim`, `to_tensor`)
  degrade to a logged warning.

### `run()` — execute the pipeline

```python
run = pipe.run(artifact_dir="artifacts/run_001")
```

`run()` loads model weights, streams windows from the source, applies the preprocessing plan, runs inference, writes all outputs, then unloads the model. If `check()` has not been called yet, `run()` calls it automatically.

`run()` raises if `report.is_runnable` is False.

`artifact_dir` is optional. When provided, file-backed outputs are written under
`artifact_dir/outputs/` and `ArtifactWriter` hashes both sidecars and output
files (see [Outputs](outputs.md#artifact-directory)).

Artifact writing is strict by default when `artifact_dir` is provided. Add this
only for exploratory runs where a missing artifact should not fail the command:

```yaml
artifact:
  failure_policy: warn     # default: strict
```

The runtime processes windows in batches up to `runtime.batch_size`. Adapters
that implement `predict_batch()` can run true batched inference; other adapters
use the base sequential fallback. Output metadata preserves source/window
details such as shape, dtype, axes, channel names, sampling frequency, voxel
metadata, and source provenance when the source adapter provides them.

`runtime.num_workers` enables ordered parallel preprocessing for each batch.
Source iteration and output writes remain ordered and single-owner, so record
ordering is deterministic.

Runtime failure policy is explicit:

```yaml
runtime:
  source_failure_policy: strict        # strict | skip_window | continue_recording
  preprocess_failure_policy: strict    # strict | drop_failed
  max_windows: 100
  max_duration_s: 60
  idle_timeout_s: 10
  fail_on_no_windows: true
```

`strict` preserves fail-fast scientific behavior. `drop_failed` keeps valid
windows from a partially bad preprocessing batch and records failed windows in
the latency/report error counts.

The returned `PipelineRunReport` contains:

```python
run.n_windows_processed   # int
run.n_outputs_written     # int
run.errors                # list[str]
run.latency_report        # LatencyReport
run.artifact_contract     # ArtifactContract | None
run.outputs               # per-adapter prediction/marker record counts
```

### `benchmark()` — latency profiling

```python
bench = pipe.benchmark(n_windows=50)
print(bench.summary())
# p50=12.3ms  p95=18.7ms  p99=23.1ms  budget=50ms  status=ok
```

`benchmark()` loads weights, runs `n_windows` windows through source →
preprocess → inference → the standard output-write stage, and records per-stage
timing. It uses an in-memory output sink, so the same runtime output lifecycle
is exercised without writing files, LSL markers, or network callbacks. For
`batch_size > 1`, Qortex stores real batch timings and reports derived
per-window latency separately, instead of assigning the whole batch latency to
the first window.

`LatencyReport` fields:

```python
bench.p50_ms, bench.p95_ms, bench.p99_ms
bench.mean_ms
bench.breakdown          # LatencyBreakdown with source/preprocess/inference/postprocess/output timings
bench.n_batches
bench.batch_p50_ms, bench.batch_p95_ms, bench.batch_p99_ms
bench.throughput_windows_per_s
bench.budget_ms          # from spec.runtime.latency_budget_ms
bench.status             # PASS | FAIL | UNKNOWN
bench.n_windows
bench.n_dropped
bench.summary()          # human-readable string
```

### `replay()` — replay a recorded session

```python
pipe.replay("recording.xdf", speed=2.0, output_dir=Path("replay_out/"))
```

`replay()` constructs a temporary pipeline with `source_path` as the source,
then re-runs compatibility and preprocessing planning against that replay
source. The original pipeline state is not mutated. `output_dir` redirects file
outputs. Runtime iteration calls the source adapter's `replay(speed=...)`
method, so adapters with time-aware replay control playback pacing directly;
nonpositive speeds are rejected before the model is loaded.

## Trigger system

A `trigger` block causes the runtime to evaluate a condition against each prediction and fire a structured `EventMarkerOutput` when the condition is met.

```yaml
trigger:
  when:
    class: motor_imagery
    probability_gte: 0.85
    stable_for: 3          # must fire on 3 consecutive windows
  emit:
    lsl_value: "1"
    action: send_cue
```

When the trigger fires, the runtime:

1. Calls `out_adapter.write_marker(EventMarkerOutput(...))` on every output adapter that exposes `write_marker`.
2. Sets `metadata["trigger_fired"] = True` in the normal `write()` call for all adapters.
3. Resets the consecutive-window counter.

Supported `when` keys: `class`, `probability_gte`, `stable_for`.

The `EventMarkerOutput` emitted contains: `event_type`, `label`, `confidence`, `window_index`, `source_id`, `timestamp_utc`, `emit_payload`.

Adapters that implement `write_marker`: `OverlayOutputAdapter` (writes a `.json` trigger sidecar), `JSONLOutputAdapter` (appends a marker record), `LSLMarkerOutputAdapter` (pushes the trigger label as a marker string).

## Accessing intermediate state

After `check()` or `run()`, the pipeline caches all intermediate results:

```python
pipe.source_profile       # SourceProfile
pipe.model_profile        # ModelProfile
pipe.compatibility_report # CompatibilityReport
pipe.preprocess_plan      # PreprocessPlan
```

These are also written into the artifact directory, so you don't need to keep the Pipeline object alive to inspect them later.

## Shortcut: `qortex.neuroai.check()`

For a quick compatibility check without a YAML file:

```python
import qortex.neuroai

report = qortex.neuroai.check(
    source="data.edf",
    model="braindecode/eegnet",
    task="eeg_classification",
)
print(report.summary())
```

## CLI

```bash
# Compatibility check — no weights downloaded.
qortex neuroai check pipeline.yaml
qortex neuroai check pipeline.yaml --json
qortex neuroai check pipeline.yaml --markdown

# Print the executable preprocessing plan.
qortex neuroai plan pipeline.yaml
qortex neuroai plan pipeline.yaml --json

# Run the pipeline and write a complete artifact directory.
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001

# Validate hashes, sidecars, output records, markers, and runtime counts.
qortex neuroai validate-artifact artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001 --strict --markdown

# Latency benchmark.
qortex neuroai benchmark pipeline.yaml --windows 100

# Replay from a file.
qortex neuroai replay pipeline.yaml --source recording.xdf --speed 2.0

# Probe a source and rank compatible models from the curated contract registry.
qortex neuroai suggest-models data.edf --task classification --top-k 10 --json

# Inspect a source without any model.
qortex neuroai inspect-source data.edf
```
