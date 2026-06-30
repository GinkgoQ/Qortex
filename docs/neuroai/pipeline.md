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

### `plan_preprocessing()` — inspect the transform chain

```python
plan = pipe.plan_preprocessing()  # calls check() if not already done

for t in plan.transforms:
    print(f"{t.kind.value}: {t.required_by} (reversible={t.reversible})")
```

Each `TransformDescriptor` has:
- `kind`: `TransformKind` enum (`resample`, `channel_select`, `channel_reorder`, `channel_map`, `bandpass`, `normalize`, `window`, `cast_dtype`, `rescale_intensity`, `reorient`, `resample_spatial`, `pad_or_crop`, `add_batch_dim`, `add_channel_dim`, `to_tensor`)
- `required_by`: why this transform is in the plan
- `params`: dict of transform parameters
- `reversible`: whether the transform can be undone
- `evidence_status`: `EvidenceStatus` for the requirement

### `run()` — execute the pipeline

```python
run = pipe.run(artifact_dir="artifacts/run_001")
```

`run()` loads model weights, streams windows from the source, applies the preprocessing plan, runs inference, writes all outputs, then unloads the model. If `check()` has not been called yet, `run()` calls it automatically.

`run()` raises if `report.is_runnable` is False.

`artifact_dir` is optional. When provided, file-backed outputs are written under
`artifact_dir/outputs/` and `ArtifactWriter` hashes both sidecars and output
files (see [Outputs](outputs.md#artifact-directory)).

The runtime processes windows in batches up to `runtime.batch_size`. Adapters
that implement `predict_batch()` can run true batched inference; other adapters
use the base sequential fallback. Output metadata preserves source/window
details such as shape, dtype, axes, channel names, sampling frequency, voxel
metadata, and source provenance when the source adapter provides them.

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

`benchmark()` loads weights, runs `n_windows` windows through source → preprocess → inference, and records per-stage timing. No output adapters are opened — nothing is written to disk or LSL.

`LatencyReport` fields:

```python
bench.p50_ms, bench.p95_ms, bench.p99_ms
bench.mean_ms
bench.breakdown          # LatencyBreakdown with source/preprocess/inference/postprocess/output timings
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

`replay()` swaps the pipeline's source adapter to read from `source_path` and runs the full pipeline. `speed=2.0` plays back twice as fast as real-time (for XDF/EDF sources that simulate timing via `source.replay(speed)`). `output_dir` redirects output paths.

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

# Run the pipeline.
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001

# Latency benchmark.
qortex neuroai benchmark pipeline.yaml --n-windows 100

# Replay from a file.
qortex neuroai replay pipeline.yaml --source recording.xdf --speed 2.0

# Probe a source and rank compatible HuggingFace models.
qortex neuroai suggest-models data.edf --task eeg_classification --limit 10

# Inspect a source without any model.
qortex neuroai inspect-source data.edf
```
