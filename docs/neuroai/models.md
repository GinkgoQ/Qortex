# Models

Every model adapter implements `inspect()` (no weights), `required_input()`, `output_schema()`, `load(runtime_spec)`, `predict(data)`, and `unload()`. `inspect()` returns a `ModelProfile` from config files only — weights are not downloaded until `load()` is called.

Model provider is set by `model.provider` in the pipeline YAML.

## Contract registry

Qortex ships a curated **contract registry** (`qortex.neuroai.models._contracts`) that maps known model IDs to verified `InputContract` / `OutputContract` pairs. Every adapter consults the registry at the start of `inspect()`:

- **Registry hit** → contract is returned immediately with `evidence_status=confirmed` (or `inferred` when the architecture allows flexibility). No network call is made.
- **Registry miss** → adapter falls back to reading `config.json` from HuggingFace Hub or the ONNX graph.

This means `suggest-models` and compatibility checking produce **confirmed** evidence instead of guesses for the 13 models currently in the registry.

### Current coverage

| Model | Provider | Modality | Task |
|---|---|---|---|
| `braindecode/EEGNet_8_2` | braindecode | EEG | classification |
| `braindecode/ShallowFBCSPNet` | braindecode | EEG | classification |
| `braindecode/Deep4Net` | braindecode | EEG | classification |
| `braindecode/EEGConformer` | braindecode | EEG | classification |
| `google/vit-base-patch16-224` | huggingface | image | classification |
| `microsoft/resnet-50` | huggingface | image | classification |
| `facebook/deit-base-patch16-224` | huggingface | image | classification |
| `openai/whisper-base` | huggingface | audio | transcription |
| `wholeBody_ct_segmentation` | monai | CT | segmentation |
| `msd_brain_tumor` | monai | MRI | segmentation |
| `ultralytics/yolov8n` | ultralytics | image | detection |

### Querying the registry in Python

```python
from qortex.neuroai.models import list_model_contracts, lookup_model_contract

# List all EEG models
eeg_models = list_model_contracts(modality="eeg")

# Look up a specific model
entry = lookup_model_contract("braindecode/EEGNet_8_2")
print(entry.input_contract.n_channels)   # 64
print(entry.input_contract.sampling_rate_hz)  # 250.0
print(entry.estimated_memory_mb)         # 120.0
```

### suggest-models CLI

`suggest-models` uses the registry for zero-cost, offline compatibility ranking — no model weights are downloaded:

```bash
qortex neuroai suggest-models data.edf --task classification --modality eeg
```

Results are scored on two axes: compatibility status (`compatible` > `compatible_with_transforms` > `uncertain` > `incompatible`) and evidence quality (`confirmed` > `inferred` > `unknown`). Models with `confirmed` contracts always rank above equivalently compatible models with `inferred` evidence.

## HuggingFace

```yaml
model:
  provider: huggingface
  id: braindecode/EEGNet
  task: eeg_classification
  revision: main
```

Uses `transformers.AutoConfig` for `inspect()` and `AutoModel.from_pretrained()` for `load()`. The `task` hint is passed to `AutoModel.from_pretrained(task)`. If the model repo has a `config.json`, `inspect()` reads input shape, hidden size, and label map from it.

For models that are not in the `transformers` ecosystem (e.g., braindecode models hosted on HF Hub), use `provider: braindecode` instead.

## ONNX

```yaml
model:
  provider: onnx
  id: model.onnx              # local path or HF path like hf://org/repo/model.onnx
  task: segmentation
  output_decoder:
    type: segmentation        # classification | segmentation | regression | embedding
    output_name: logits
    activation: softmax
    argmax_axis: 1
```

`inspect()` reads the ONNX graph without creating an inference session. Input names, output names, and shapes are extracted from `onnx.load_model()`.

`load()` creates an `onnxruntime.InferenceSession` with `ExecutionProviders` selected from `runtime.device`:
- `cpu` → `CPUExecutionProvider`
- `cuda` / `cuda:N` → `CUDAExecutionProvider`

`predict()` runs `session.run(output_names, {input_name: data})`. Input must be
`float32` numpy array. Semantic decoding is explicit:

- `classification` applies `softmax` unless `activation: none` / `probabilities` is declared.
- `segmentation` can apply `softmax + argmax_axis` or `sigmoid + threshold`.
- `regression` returns a scalar value.
- `embedding` returns the raw embedding tensor.
- Unknown decoder/task returns `ModelOutput(output_type="raw")` instead of
  pretending the output is classification.

## PyTorch / TorchScript

```yaml
model:
  provider: torch             # or: torchscript / ts for TorchScript
  id: model.pt                # or model.ts for TorchScript
  task: classification
```

For TorchScript (`provider: torchscript` or path ending in `.ts`), loads with `torch.jit.load()`. For standard PyTorch, loads with `torch.load(weights_only=False)`.

Input shape is inferred from the TorchScript graph or from the first layer's attributes (`in_features` for linear, `in_channels` for conv). FP16 is applied with `.half()` when `runtime.fp16=True` and device contains `"cuda"`.

`predict()` output parsing:
- 1D output → softmax → `ClassificationOutput`
- Multi-dimensional output → argmax → segmentation mask

## Braindecode

```yaml
model:
  provider: braindecode
  id: braindecode/EEGNet      # or ShallowFBCSPNet, DeepFBCSPNet, EEGConformer, etc.
  input:
    n_channels: 22
    n_times: 1000
  output:
    n_classes: 4
    classes: [left_hand, right_hand, feet, tongue]
```

Loads `config.json` from HF Hub to get `n_chans`, `n_times`, `n_outputs`, and `id2label`. Known model names map to the corresponding braindecode class:

| HF model name | Braindecode class |
|---|---|
| `eegnet` | `EEGNetv4` |
| `shallowfbcspnet` | `ShallowFBCSPNet` |
| `deepfbcspnet` | `Deep4Net` |
| `eegconformer` | `EEGConformer` |
| `tidnet` | `TIDNet` |

Built-in Braindecode classes require confirmed dimensions from the curated
registry, HuggingFace config, or explicit YAML fields under `model.input` and
`model.output`. Qortex does not instantiate a 64-channel / 512-sample / 2-class
fallback model, because that would produce scientifically meaningless outputs.

Unknown names fall back to `AutoModel.from_pretrained()`. Input is always
`[batch, channels, time]`. Output has softmax applied.

## MONAI bundle

```yaml
model:
  provider: monai
  id: wholeBody_ct_segmentation   # MONAI Hub bundle name, local path, or local ZIP
```

Bundle resolution order: local directory → local ZIP (extracted to tempdir) → MONAI Hub download.

`inspect()` reads `configs/metadata.json` and `configs/inference.json` for network parameters. `load()` uses `monai.bundle.ConfigParser` for the network definition and `torch.load` for weights.

`predict()` uses `monai.inferers.sliding_window_inference(roi_size=(96, 96, 96))` for volumetric data. Returns `VolumePredictionOutput`.

## Ultralytics (YOLOv8)

```yaml
model:
  provider: ultralytics
  id: yolov8n.pt              # local path or Ultralytics model name
  task: detect                # detect | segment | classify
```

`inspect()` loads the YOLO model (Ultralytics always loads weights at init — no separate inspect). Task dispatch:
- `detect` → `DetectionOutput` with `BoundingBox` list
- `segment` → `SegmentationOutput` with argmax mask
- `classify` → `ClassificationOutput`

Input conversion: numpy CHW float → HWC uint8.

Box parsing: `result.boxes.xyxy`, `result.boxes.conf`, `result.boxes.cls`.

## Custom plugin

```yaml
model:
  provider: plugin
  id: my_model.py             # path to a Python file
  trust_remote_code: true     # required — must be set explicitly
```

Loads `my_model.py` with `importlib.util.spec_from_file_location`. Validates that the module defines a `QortexPlugin` class with all required methods: `inspect`, `required_input`, `output_schema`, `load`, `predict`.

`trust_remote_code: true` must be set in the pipeline spec. Without it, the adapter raises `CompatibilityError` before loading anything.

All plugin calls are wrapped in try/except with structured error messages so one bad prediction does not crash the full run.

### Plugin interface

```python
class QortexPlugin:
    def inspect(self) -> dict:
        """Return model metadata as a dict (no weights loaded)."""
        return {
            "model_id": "my_model",
            "task": "eeg_classification",
            "input_shape": [None, 64, 512],  # [batch, channels, time]
            "n_outputs": 4,
        }

    def required_input(self) -> dict:
        return {"shape": [None, 64, 512], "dtype": "float32"}

    def output_schema(self) -> dict:
        return {"type": "classification", "n_classes": 4}

    def load(self, runtime_spec) -> None:
        """Load weights. Called once before streaming starts."""
        self.model = load_my_model()

    def predict(self, data) -> object:
        """Run inference on one window. data is [batch, channels, time] float32."""
        return self.model(data)
```

## Model profile

`inspect()` returns a `ModelProfile`:

```python
profile.model_id              # str — HF ID, local path, or model name
profile.provider              # str — "huggingface" | "onnx" | "torch" | ...
profile.task                  # str | None — "eeg_classification" | "segmentation" | ...
profile.revision              # str | None — git revision or None
profile.model_hash            # str | None — SHA-256 of weights file when available
profile.license               # str | None — SPDX license identifier
profile.trusted               # bool — True when trust_remote_code was accepted
profile.input_contract        # InputContract | None — formal input requirements
profile.output_contract       # OutputContract | None — formal output schema
profile.estimated_params      # int | None — number of model parameters
profile.estimated_memory_mb   # float | None — estimated GPU/CPU memory in MB
profile.supported_devices     # list[str] — declared supported devices
profile.warnings              # list[WarningItem] — non-fatal inspection warnings
```

The **`InputContract`** inside `profile.input_contract` carries what the model formally requires:

```python
contract = profile.input_contract

contract.modality             # "eeg" | "mri" | "image" | ...
contract.axis_convention      # AxisConvention — e.g. batch_channels_time
contract.n_channels           # int | None — required channel count
contract.sampling_rate_hz     # float | None — required sampling frequency
contract.window_duration_s    # float | None — required window length
contract.spatial_shape        # tuple[int, ...] | None — (H, W) or (Z, Y, X)
contract.voxel_sizes_mm       # tuple[float, ...] | None — spatial resolution
contract.dtype                # str — "float32" (default)
contract.evidence_status      # EvidenceStatus — confirmed | inferred | unknown
```

The **`OutputContract`** inside `profile.output_contract` declares what the model produces:

```python
out = profile.output_contract

out.output_type               # "classification" | "segmentation" | "detection" | ...
out.classes                   # list[str] — class names (empty when unknown)
out.n_classes                 # int | None
out.output_dtype              # str — "float32"
out.produces_probabilities    # bool — True when output is a probability distribution
```

The compatibility engine reads `input_contract` to check modality, channel count, sampling rate, spatial shape, and axis convention against `SourceProfile`. Fields with `evidence_status=unknown` produce `uncertain` compatibility rather than a blocker.

The plugin adapter raises `ModelAdapterError` (not `CompatibilityError`) when `trust_remote_code` is not set.
