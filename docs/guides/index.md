# Guides

Use guides when you know the job and need the shortest correct route. Concept pages explain design; API pages list objects; guides tell you what to run and why that step exists.

## Pipeline At A Glance

| Stage | Input | Output | Mistake it prevents |
|---|---|---|---|
| [Inspect](../dataset/index.md) | Dataset ID or local BIDS root | Manifest, entities, sidecars, participants, events, size estimates | Downloading a dataset whose structure cannot support your task |
| [Assess readiness](../readiness/index.md) | Manifest plus optional local files | Doctor, can-train, label, leakage, content, and minimum-subset reports | Training on missing labels, weak evidence, or unsafe splits |
| [Download](../download/index.md) | A goal or file selection | Resumable local subset with companions | Moving terabytes when metadata or one batch would answer the question |
| [Visualize](../visualization/index.md) | Local BIDS files or converted artifact | QC HTML, overlays, thumbnails, plots, sample previews | Converting unreadable files or trusting unseen geometry |
| [Convert](../conversion/index.md) | Local BIDS subset | ML artifact with splits and provenance | Losing source lineage or mixing subjects across splits |
| [Use artifacts](../artifacts/index.md) | Artifact directory | Arrays, datasets, split summaries, framework bridges | Rebuilding loaders by hand and silently changing the training set |
| [NeuroAI](../neuroai/index.md) | Source, model, output contract | Compatibility report, plan, run artifact | Loading weights before the source can satisfy the model contract |

## Example Evidence From A Real Dataset

The guide examples use public OpenNeuro dataset `ds000001`. The event figure
below comes from `Dataset.events(subject="01", task="balloonanalogrisktask",
run="01")` and contains 158 real `events.tsv` rows.

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Real ds000001 events.tsv timeline and trial-type counts">
  <figcaption>Qortex uses event tables like this to reason about label candidates, class balance, and subject coverage. For `ds000001`, labels are visible but still marked as candidate until local columns are confirmed.</figcaption>
</figure>

## The Core Route

<div class="tq-pipeline-steps" markdown>

### 1 · Inspect

Read structure first. Qortex can usually count files, subjects, modalities, events, participants, BIDS entities, and rough size before raw data is local.

```python
from qortex import Dataset

ds = Dataset("ds000001")
print(ds.doctor().to_text())
```

Go to [Inspect datasets](../dataset/index.md).

---

### 2 · Assess Readiness

Ask whether a specific workflow is possible. Readiness reports separate confirmed facts from inferred or missing evidence.

```python
print(ds.can_train(target="trial_type").to_text())
print(ds.minimum(goal="first-batch").to_text())
```

Go to [Assess readiness](../readiness/index.md).

---

### 3 · Download

Download a plan, not a wish. Use `minimum()` when you need the smallest valid subset; use selection filters when you already know the subjects, tasks, or datatypes.

```python
plan = ds.minimum(goal="first-batch")
ds.download_paths(plan.files, output_dir="data/ds000001")
```

Go to [Download](../download/index.md).

---

### 4 · Visualize

Run visual QC before conversion. Fast previews catch corrupt files, strange orientations, empty masks, bad overlays, missing companions, and signal problems while the dataset is still easy to fix.

```python
report = ds.visual_audit(local_path="data/ds000001", output_dir="qa/ds000001")
print(report.to_markdown())
```

Go to [Visualization](../visualization/index.md).

---

### 5 · Convert

Write a durable artifact. Conversion should decide windowing, labels, split policy, format, and provenance in one place.

```python
result = ds.convert(
    output_dir="artifacts/ds000001",
    output_format="parquet",
    split_strategy="subject",
)
```

Go to [Conversion](../conversion/index.md).

---

### 6 · Use Artifacts

Open the artifact instead of re-parsing raw BIDS files in every notebook. Use framework bridges only after checking the split manifest.

```python
from qortex import Artifact

artifact = Artifact.open("artifacts/ds000001")
print(artifact.leakage_check().to_text())
X_train, y_train = artifact.sklearn(split="train")
```

Go to [Artifacts](../artifacts/index.md).

---

### NeuroAI Runtime

Use NeuroAI when your task is inference rather than dataset conversion. The runtime probes sources and model contracts before loading weights.

```bash
qortex neuroai check pipeline.yaml
qortex neuroai run pipeline.yaml --artifact-dir artifacts/run_001
qortex neuroai validate-artifact artifacts/run_001
```

Go to [NeuroAI runtime](../neuroai/index.md).

</div>

## Which Guide Do I Need?

| Situation | Best route |
|---|---|
| I have an OpenNeuro ID and no local files | [Inspect](../dataset/index.md) → [Readiness](../readiness/index.md) → [Minimum subset](../readiness/minimum.md) |
| I need labels for supervised learning | [Label readiness](../readiness/label-readiness.md) → [Can train](../readiness/can-train.md) |
| I want a small smoke-test download | [Minimum subset](../readiness/minimum.md) → [Selective download](../download/selective-download.md) |
| I suspect local data is incomplete | [Content status](../readiness/content-status.md) → [Local index](../download/local-index.md) |
| I need to inspect images or signals | [Visual audit](../visualization/visual-audit.md) → modality-specific QC page |
| I need Parquet, HDF5, WebDataset, or HuggingFace | [Conversion pipeline](../conversion/pipeline.md) → [Formats](../conversion/formats.md) |
| I need model inference and provenance | [NeuroAI pipeline](../neuroai/pipeline.md) → [Outputs](../neuroai/outputs.md) |
