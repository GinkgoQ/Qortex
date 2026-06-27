# Models

Every model adapter implements `inspect()` (no weights), `required_input()`, `output_schema()`, `load(runtime_spec)`, `predict(data)`, and `unload()`. `inspect()` returns a `ModelProfile` from config files only — weights are not downloaded until `load()` is called.

Model provider is set by `model.provider` in the pipeline YAML.

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
  task: classification
```

`inspect()` reads the ONNX graph without creating an inference session. Input names, output names, and shapes are extracted from `onnx.load_model()`.

`load()` creates an `onnxruntime.InferenceSession` with `ExecutionProviders` selected from `runtime.device`:
- `cpu` → `CPUExecutionProvider`
- `cuda` / `cuda:N` → `CUDAExecutionProvider`

`predict()` runs `session.run(output_names, {input_name: data})`. Input must be `float32` numpy array.

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
```

Loads `config.json` from HF Hub to get `n_chans`, `n_times`, `n_outputs`, and `id2label`. Known model names map to the corresponding braindecode class:

| HF model name | Braindecode class |
|---|---|
| `eegnet` | `EEGNetv4` |
| `shallowfbcspnet` | `ShallowFBCSPNet` |
| `deepfbcspnet` | `Deep4Net` |
| `eegconformer` | `EEGConformer` |
| `tidnet` | `TIDNet` |

Unknown names fall back to `AutoModel.from_pretrained()`. Input is always `[batch, channels, time]`. Output has softmax applied.

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
profile.model_id            # str — HF ID, path, or name
profile.provider            # str
profile.task                # str | None
profile.input_shape         # tuple[int | None, ...] | None
profile.output_shape        # tuple[int | None, ...] | None
profile.input_dtype         # str — "float32" etc.
profile.modality            # expected modality, if declared
profile.n_outputs           # int | None — number of classes or output dims
profile.label_map           # dict[int, str] | None
profile.framework           # "torch" | "onnx" | "huggingface" | ...
profile.evidence            # dict[str, EvidenceStatus]
```

As with `SourceProfile`, every field carries an `EvidenceStatus`. The compatibility engine uses these to distinguish hard blockers from uncertainties.
