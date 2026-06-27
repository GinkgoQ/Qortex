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
  subject: "01"               # optional — single subject
  session: null               # optional
  task: null                  # optional

window:
  duration_s: 4.0             # window length in seconds
  step_s: 2.0                 # step between windows (overlap = duration - step)
  max_windows: null           # null = no limit

model:
  provider: huggingface       # huggingface | onnx | torch | torchscript | braindecode | monai | ultralytics | plugin
  id: braindecode/EEGNet      # HF repo ID, local path, or model identifier
  task: eeg_classification    # task hint passed to the model adapter
  revision: main              # HF revision (branch, tag, commit)
  trust_remote_code: false    # must be true for plugin provider

preprocessing:
  resample_hz: 128            # target sampling rate (null = no resampling)
  z_score: true               # z-score per channel
  bandpass_hz: [1.0, 40.0]   # [low, high] Hz, null = skip
  channel_select: null        # list of channel names to keep, null = all

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
  num_threads: null           # ONNX/Torch thread count, null = default
```

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

- `status`: `runnable` | `runnable_with_transforms` | `uncertain` | `not_runnable`
- `is_runnable`: bool — True when status is `runnable` or `runnable_with_transforms`
- `required_transforms`: list of transforms the runtime will apply automatically
- `blockers`: list of hard incompatibilities (wrong modality, incompatible shape, missing dependency)
- `warnings`: non-fatal issues (unknown channel labels, inferred sampling rate)
- `unknowns`: fields that could not be determined without loading data

```python
print(report.summary())
# status: runnable_with_transforms
# transforms: [resample_250→128Hz, z_score_per_channel]
# blockers: []
# warnings: [channel labels inferred from index, not names]
# unknowns: []

for b in report.blockers:
    print(b.code, b.message)

if not report.is_runnable:
    # Do not call run() — it will raise.
    pass
```

The distinction between `uncertain` and `not_runnable` matters: `uncertain` means a required property (e.g., model input shape) could not be read from the config. The pipeline might work, but check cannot guarantee it. `not_runnable` means a hard blocker was found.

### `plan_preprocessing()` — inspect the transform chain

```python
plan = pipe.plan_preprocessing()  # calls check() if not already done

for t in plan.transforms:
    print(f"{t.kind.value}: {t.required_by} (reversible={t.reversible})")
```

Each `TransformDescriptor` has:
- `kind`: `TransformKind` enum (`resample`, `z_score`, `bandpass`, `channel_select`, `transpose`, `to_float32`, `normalize`, `window`)
- `required_by`: why this transform is in the plan
- `params`: dict of transform parameters
- `reversible`: whether the transform can be undone
- `evidence`: `EvidenceStatus` for the requirement

### `run()` — execute the pipeline

```python
run = pipe.run(artifact_dir="artifacts/run_001")
```

`run()` loads model weights, streams windows from the source, applies the preprocessing plan, runs inference, writes all outputs, then unloads the model. If `check()` has not been called yet, `run()` calls it automatically.

`run()` raises if `report.is_runnable` is False.

`artifact_dir` is optional. When provided, `ArtifactWriter` writes 9 files to that directory (see [Outputs](outputs.md#artifact-directory)).

The returned `PipelineRunReport` contains:

```python
run.n_windows_processed   # int
run.n_outputs_written     # int
run.errors                # list[str]
run.latency_report        # LatencyReport
run.artifact_contract     # ArtifactContract | None
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
bench.per_stage          # dict: stage_name → {p50, p95, p99}
bench.budget_ms          # from spec.runtime.latency_budget_ms
bench.budget_met         # bool
bench.summary()          # human-readable string
```

### `replay()` — replay a recorded session

```python
pipe.replay("recording.xdf", speed=2.0, output_dir=Path("replay_out/"))
```

`replay()` swaps the pipeline's source adapter to read from `source_path` and runs the full pipeline. `speed=2.0` plays back twice as fast as real-time (for XDF/EDF sources that simulate timing via `source.replay(speed)`). `output_dir` redirects output paths.

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
