# Guides

Guides are task-oriented. They answer "how do I do X?" without explaining why things are designed this way — see [Get Started](../getting-started/index.md) for that.

Qortex works as a pipeline. Each guide covers one stage.

---

<div class="tq-pipeline-steps" markdown>

## 1 → Inspect

**Examine a dataset without downloading it.**

Fetch the manifest, count subjects and modalities, read labels from sidecar files, and get a size estimate — all from the OpenNeuro API. No data transferred.

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor()
print(report.to_text())
```

→ [Inspect a dataset](../dataset/index.md)

---

## 2 → Assess readiness

**Decide if the dataset is usable before committing to a download.**

Check label coverage, class balance, event file completeness, minimum sample counts, and split feasibility. Get a binary yes/no with an actionable explanation.

```python
ok = ds.can_train(target_col="trial_type", min_classes=2, min_per_class=10)
report = ds.label_readiness(target_col="trial_type")
```

→ [Assess readiness](../readiness/index.md)

---

## 3 → Download

**Fetch exactly what you need — nothing more.**

Plan before downloading. Filter by subject, task, modality, or suffix. Resume interrupted transfers. Download metadata only for offline inspection.

```python
plan = ds.minimum(goal="first-batch")
ds.download_paths(plan.files, data_dir="data/")
```

→ [Download](../download/index.md)

---

## 4 → Visualize

**Inspect local files before running anything.**

Render center slices, channel plots, event timelines, and QC panels without loading full volumes. Catch bad files, wrong orientations, and mismatched labels before conversion.

```python
ds.visualize(data_dir="data/ds004130/", output_dir="qa/")
```

→ [Visualize](../visualization/index.md)

---

## 5 → Convert

**Transform raw BIDS data into ML-ready formats.**

One command writes Parquet, Zarr, HDF5, WebDataset, HuggingFace Dataset, or TFRecord with subject-level splits, reproducible seeds, and a full provenance record.

```python
from qortex.convert import ConversionPipeline

pipeline = ConversionPipeline(data_dir="data/ds004130/", format="parquet")
artifact = pipeline.run(splits={"train": 0.7, "val": 0.15, "test": 0.15})
```

→ [Convert](../conversion/index.md)

---

## 6 → Use artifacts

**Load, inspect, and bridge to your training framework.**

Open an artifact, check for subject leakage, visualize sample distributions, and connect to PyTorch, Lightning, or HuggingFace `datasets` in one line.

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130_parquet/")
art.inspect()
loader = art.to_torch_dataset(split="train")
```

→ [Artifacts](../artifacts/index.md)

---

## NeuroAI runtime

**Run a full inference pipeline declaratively.**

Connect a data source, a pre-trained model, and output sinks through a single YAML. The runtime checks source/model compatibility before loading any weights.

```yaml
source:
  type: bids
  path: data/ds004130
  modality: eeg
model:
  hub: braindecode/eegnet-motor-imagery
outputs:
  - type: parquet
    path: predictions/
```

→ [NeuroAI runtime](../neuroai/index.md)

</div>

---

## Which guide do I need?

| Situation | Go to |
|---|---|
| I have a dataset ID and want to know if it's usable | [Inspect](../dataset/index.md) + [Assess readiness](../readiness/index.md) |
| I want the smallest download to test a pipeline | [Download → Minimum subset](../download/plan.md) |
| Something looks wrong with my local data | [Visualize](../visualization/index.md) |
| I want to train a model and need Parquet/HDF5 | [Convert](../conversion/index.md) |
| I want to run inference with a pre-trained model | [NeuroAI runtime](../neuroai/index.md) |
| I need to verify my splits have no leakage | [Artifacts → Leakage check](../artifacts/ml-bridge.md) |
