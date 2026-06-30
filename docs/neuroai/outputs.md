# Outputs

Every output adapter implements `open()`, `write(output, metadata)`, and `close()` as a context manager. Multiple outputs can be listed in one pipeline — all are opened before streaming starts and closed after the run finishes.

Output type is set by `outputs[n].type` in the pipeline YAML.

## File outputs

### JSONL

```yaml
outputs:
  - type: jsonl
    path: predictions.jsonl
    append: false
```

One JSON object per prediction window. Schema varies by output type (classification, detection, segmentation, etc.) but always includes timestamp, index, `window_index`, source, pipeline hash, and `output_type` when available.

### Parquet

```yaml
outputs:
  - type: parquet
    path: predictions.parquet
```

Columnar output for offline analytics. Core columns include timestamp, output type, class, class index, top probability, regression value, pipeline hash, and run metadata. Probability vectors are flattened into per-class columns for direct filtering and aggregation.

### CSV

```yaml
outputs:
  - type: csv
    path: predictions.csv
    append: true
```

Appendable CSV for lightweight analytics and spreadsheet inspection. It writes stable columns for timestamp, index, pipeline hash, output type, class, class index, top probability, regression value, window index, trigger state, source, probabilities JSON, runtime metadata JSON, model-output metadata JSON, and compact summaries for raw arrays, masks, and embeddings. `append: true` opens in append mode so multiple runs accumulate in one file.

## Streaming outputs

### LSL marker

```yaml
outputs:
  - type: lsl_marker
    stream_name: qortex_markers
```

Creates a `pylsl.StreamInfo` with `type="Markers"`, `channel_count=1`, `channel_format=cf_string`. Pushes one string per prediction: `"{class_name}:{class_index}"`. When `metadata["trigger_value"]` is set, that value is pushed instead.

This is the standard interface for BCI applications — BCI2000, OpenViBE, and similar platforms can receive these markers on the LSL network.

### WebSocket

```yaml
outputs:
  - type: websocket
    path: ws://localhost:8765/predictions
```

Uses `websocket-client` (synchronous). Sends a JSON payload per `write()`. Connection is opened in `open()` and closed in `close()`. Does not retry on write failure — errors are logged.

### HTTP callback

```yaml
outputs:
  - type: http
    path: https://api.example.com/predictions
    extra:
      auth:
        type: bearer
        token: "${API_TOKEN}"
      retry_max: 3
```

Uses `requests.Session`. Sends a JSON POST per window. Retries up to 3 times with `sleep(0.5 * attempt)` backoff. Supports bearer token and basic auth via `spec.extra["auth"]`.

## Imaging outputs

### NIfTI

```yaml
outputs:
  - type: nifti
    path: mask.nii.gz
```

Writes a `nibabel.Nifti1Image` from a `SegmentationOutput` or `VolumePredictionOutput`. Validates that the affine determinant is positive before writing (a negative determinant indicates a flipped orientation). Writes a JSON sidecar (`mask_provenance.json`) with pipeline reference and model ID.

### DICOM-SEG

```yaml
outputs:
  - type: dicom_seg
    path: output_seg/
```

Creates a `highdicom.seg.Segmentation` with `SegmentDescription` and `AlgorithmIdentificationSequence`. Validates mask shape against the source series dimensions. Falls back to saving as `.npy` if the `highdicom` API raises (API changes across versions).

### DICOM-SR

```yaml
outputs:
  - type: dicom_sr
    path: output_sr/
```

Creates a `highdicom.sr.EnhancedSR` structured report using TID 1500 MeasurementReport. Used for `RegressionOutput` and `ReportOutput` types. Falls back to `.json` on `highdicom` API failure.

## Image overlay

### Overlay

```yaml
outputs:
  - type: overlay
    path: annotated_frames/
    extra:
      format: png        # png | jpg (default: png)
      alpha: 0.45        # mask transparency, 0–1 (default: 0.45)
      line_width: 2      # bounding-box border width in pixels
```

Renders model predictions on top of source images and saves annotated frames to disk. Requires Pillow or OpenCV (checked at runtime; falls back automatically between the two).

The source image must be passed through `metadata["source_image"]` as a `[H, W]` or `[H, W, C]` array when calling `out_adapter.write(output, metadata={"source_image": frame, ...})`.

What gets drawn per output type:

| Output type | Rendered annotation |
|---|---|
| `DetectionOutput` | Coloured bounding boxes with class label and confidence |
| `SegmentationOutput` | Per-class colour overlay at configured `alpha` transparency |
| `ClassificationOutput` | Black banner with `class_name: confidence%` in the top-left corner |

When no `source_image` is provided (pure signal pipelines), the adapter writes a JSON sidecar per window instead of a frame.

When a trigger fires, `write_marker(EventMarkerOutput)` writes a companion `trigger_{frame_idx}.json` next to the annotated frame.

## Annotation outputs

### BIDS derivative

```yaml
outputs:
  - type: bids
    path: derivatives/qortex/
    extra:
      subject: "01"
      session: null
      task: rest
      run: "01"
```

Creates a BIDS derivative directory structure. Writes `dataset_description.json` with `GeneratedBy: [{Name: "Qortex"}]`. Filenames follow BIDS entity naming:

```
sub-01[_ses-01][_task-rest][_run-01]_{suffix}.json
```

For segmentation output, writes a NIfTI via nibabel instead of JSON. On `close()`, writes `provenance.json` summarizing all outputs in the session.

### COCO JSON

```yaml
outputs:
  - type: coco
    path: predictions_coco.json
```

Accumulates `images`, `annotations`, and `categories` in memory across all windows. Writes the full COCO JSON on `close()`. Bounding boxes are in COCO format: `[x, y, width, height]`.

### YOLO text

```yaml
outputs:
  - type: yolo
    path: yolo_labels/
```

One `.txt` file per image: `{class_id} {cx} {cy} {w} {h}` with normalized coordinates (0–1). On `close()`, writes `classes.txt` listing all class names.

## Canonical output types

All adapters receive one of these typed objects:

```python
ClassificationOutput(
    class_name="motor_imagery",
    class_index=1,
    confidence=0.92,
    top_k=[("motor_imagery", 0.92), ("rest", 0.08)],
)

DetectionOutput(
    boxes=[
        BoundingBox(x1=0.1, y1=0.2, x2=0.4, y2=0.6,
                    class_name="lesion", class_index=0, confidence=0.87)
    ],
    image_shape=(512, 512),
)

SegmentationOutput(
    mask=np.array(..., dtype=np.int16),   # [z, y, x] or [h, w]
    n_classes=5,
    class_labels={0: "background", 1: "WM", 2: "GM", 3: "CSF", 4: "tumor"},
    affine=np.eye(4).tolist(),
    voxel_sizes=(1.0, 1.0, 1.0),
)

# Convenience: build from a list of class names
SegmentationOutput.from_label_list(
    mask=np.array(..., dtype=np.int16),
    class_labels=["background", "WM", "GM", "CSF", "tumor"],
    affine=np.eye(4).tolist(),
    voxel_sizes=(1.0, 1.0, 1.0),
)

RegressionOutput(value=3.72, units="mm", confidence_interval=(3.1, 4.3))

EmbeddingOutput(vector=np.array(...), dimensionality=128, model_layer="encoder")

TimeSeriesPredictionOutput(
    predictions=np.array(...),    # [time, n_classes]
    timestamps=np.array(...),
    sampling_rate_hz=250.0,
    label_map={0: "N2", 1: "N3", 2: "REM", 3: "Wake"},
)

EventMarkerOutput(
    event_type="trigger",
    label="motor_imagery",
    timestamp_utc="2024-06-28T12:00:00+00:00",
    confidence=0.92,
    window_index=42,
    source_id="bids:ds004130",
    emit_payload={"lsl_value": "1", "action": "send_cue"},
)

VolumePredictionOutput(
    mask=np.array(...),
    affine=np.eye(4),
    voxel_sizes=(1.0, 1.0, 1.0),
    n_classes=3,
    class_labels=["background", "lesion", "edema"],
)

ReportOutput(
    title="Brain Lesion Analysis",
    findings=["Single lesion detected in right hemisphere"],
    measurements={"volume_ml": 4.2},
    confidence=0.87,            # numeric score in [0, 1]
    warnings=[],
    source_id="dicom:study_001",
    model_id="org/lesion-detector",
)
# report.confidence_level → "high" | "medium" | "low"
```

`BoundingBox` has helpers for format conversion:

```python
box.to_coco()              # [x, y, width, height]
box.to_yolo(img_w=512, img_h=512)   # [cx, cy, w, h] normalized
```

`ClassificationOutput.from_probs(probs, top_k=5)` builds from a raw probability array.

## Artifact directory

When `pipe.run(artifact_dir="...")` is called, file-backed outputs are routed
under `artifact_dir/outputs/` and `ArtifactWriter` writes a self-contained
artifact:

```
artifacts/run_001/
  provenance.json           pipeline spec, source, model, git hash, timestamps
  compatibility_report.json CompatibilityReport fields
  preprocess_plan.json      PreprocessPlan — transforms applied and why
  runtime_report.json       windows processed, outputs written, errors
  latency_report.json       p50/p95/p99 per stage
  warnings.json             non-fatal issues during the run
  pipeline.yaml             copy of the pipeline spec
  artifact_contract.json    schema, hash, provenance summary
  artifact_manifest.json    SHA-256 and file size for every file below
  outputs/
    predictions.jsonl
    predictions.parquet
    predictions.csv
```

`artifact_manifest.json` is recursive: it includes sidecars, provenance files,
and files under `outputs/`. This lets downstream consumers verify prediction
files without re-running inference.

Validate a run artifact before handing it to another workflow:

```python
from qortex.neuroai import validate_artifact

report = validate_artifact("artifacts/run_001")
print(report.summary())
```

```bash
qortex neuroai validate-artifact artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001 --strict --json
```

The validator checks required sidecars, recursive manifest file size and
SHA-256 entries, JSONL prediction records, trigger marker records, CSV columns,
Parquet metadata, NIfTI mask readability/shape, COCO JSON structure, YOLO box
normalization, DICOM output headers when `pydicom` is installed, and
consistency between observed output records and `runtime_report.json.outputs`.
