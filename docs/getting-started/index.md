# Get Started

Qortex helps you answer a practical question before you spend time or bandwidth:

> Can this neuroscience dataset support my analysis, and what is the smallest safe path to a usable artifact?

Start here if you have an OpenNeuro dataset ID, a local BIDS directory, or a model pipeline that needs source compatibility checks.

## Fastest Path

| Step | Do this | What you learn |
|---|---|---|
| 1 | [Install Qortex](install.md) | Which optional extras match your modality. |
| 2 | [Run the quickstart](quickstart.md) | How `doctor`, `minimum`, download, conversion, and artifacts fit together. |
| 3 | [Choose a tutorial](../tutorials/index.md) | How the same workflow changes for EEG, sleep, seizure, MRI, fMRI, and segmentation tasks. |
| 4 | [Read the guide map](../guides/index.md) | Which subsystem owns the next problem you need to solve. |

## Mental Model

Qortex is a readiness layer. It sits between raw neurodata and model code.

```text
OpenNeuro / local BIDS
        ↓
inspect       manifest, sidecars, participants, events, headers
assess        labels, companions, sample counts, split safety
plan          exact files and bytes for one goal
download      selective, resumable transfer
visualize     QC before conversion
convert       artifact formats with provenance
use           sklearn, torch, HuggingFace, NeuroAI runtime
```

The important design choice is that inspection comes before transfer. Qortex uses the manifest and lightweight metadata first, then asks you to download only when a real workflow needs local bytes.

## Pick Your Entry Point

<div class="tq-card-grid tq-card-grid-3">
  <div class="tq-card">
    <h3><a href="quickstart/">Dataset ID</a></h3>
    <p>Start with `Dataset("ds000001")`, inspect readiness, plan a first batch, convert, and open an artifact.</p>
  </div>
  <div class="tq-card">
    <h3><a href="first-visual-audit/">Local BIDS data</a></h3>
    <p>Run a visual audit before conversion so broken images, missing files, and geometry issues show up early.</p>
  </div>
  <div class="tq-card">
    <h3><a href="../neuroai/">Inference pipeline</a></h3>
    <p>Use NeuroAI when your question is source/model compatibility and reproducible inference output.</p>
  </div>
</div>

## Common First Questions

| Question | Best page |
|---|---|
| Which optional packages do I need? | [Install](install.md) |
| Can this dataset train a model? | [Can train](../readiness/can-train.md) |
| How do I avoid downloading the full dataset? | [Minimum subset](../readiness/minimum.md) |
| How do I check labels and events? | [Label readiness](../readiness/label-readiness.md) |
| How do I inspect images before conversion? | [First visual audit](first-visual-audit.md) |
| How do I load converted data into ML code? | [ML bridge](../artifacts/ml-bridge.md) |

## What Not To Skip

Run `doctor()` before conversion. Run `leakage_check()` before training. Keep the artifact manifest with your model results. These three habits catch the most expensive mistakes: unusable labels, unsafe subject splits, and unreproducible training data.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-manifest-composition.png" alt="Bar charts showing OpenNeuro ds000001 file suffix counts and bytes by BIDS datatype.">
  <figcaption>Real `Dataset.manifest()` output from OpenNeuro ds000001: suffix counts and bytes by BIDS datatype.</figcaption>
</figure>

```python
ds = Dataset('ds000001', snapshot='1.0.0')
manifest = ds.manifest()
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->
