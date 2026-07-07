# Minimum

`ds.minimum()` computes the smallest real download that enables a specific goal. It returns a `DownloadPlan` with exactly the files needed — no padding, no extras beyond required companions.

## Python

```python
from qortex import Dataset

ds = Dataset("ds000001")

report = ds.minimum(goal="first-batch", output_dir="data/ds000001")
print(report.to_text())
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-minimum-plan.png" alt="Qortex minimum first-batch plan for ds000001 showing the selected metadata, events, sidecar, and BOLD file">
  <figcaption>Real output from `ds.minimum(goal="first-batch")` on OpenNeuro `ds000001`. The plan contains 7 files and about 0.05 GB: enough to load one BOLD run with its interpretation files.</figcaption>
</figure>

## Goals

| Goal | What it means |
|------|--------------|
| `first-batch` | Enough subjects to run one complete pipeline pass (inspect → download → convert → train step). Usually 3–5 subjects. |
| `label-check` | Only the events.tsv and sidecar JSON files. No imaging data. |
| `validation` | A representative subset for BIDS validation — one subject per modality/task combination. |
| `metadata` | All JSON sidecars, TSV, bval/bvec, root-level files. No imaging data. |

## CLI

```bash
qortex minimum ds000001 --goal first-batch
qortex minimum ds000001 --goal label-check
qortex minimum ds000001 --goal metadata
```

Observed output:

```
Dataset : ds000001 (1.0.0)
Goal    : first-batch
Status  : possible
Reason  : A first batch needs one loadable primary recording plus required companions.
Files   : 7
Size    : 0.05 GB (estimated)
```

Add `--download` to execute the plan immediately:

```bash
qortex minimum ds000001 --goal first-batch --download --output-dir data/ds000001
```

## How first-batch is computed

`first-batch` selects subjects that maximize label class coverage within the minimum count:

1. Fetch the manifest and events.tsv for all subjects
2. For each subject, count label occurrences per class
3. Select the minimum number of subjects such that every class appears at least once
4. Add one extra subject as a buffer for val/test split
5. Include all companions (JSON, events, bval/bvec) for the selected subjects

If the dataset has no label column (no events.tsv), `first-batch` falls back to selecting 3 subjects arbitrarily.

## Using the plan

```python
report = ds.minimum(goal="first-batch", output_dir="data/ds000001")

# Inspect before committing
for f in report.plan.files:
    print(f.path, f.size)

# Download
ds.download_paths(report.plan.files, output_dir="data/ds000001")
```

## After first-batch succeeds

A successful first-batch run confirms:

- The download engine works for this dataset
- The conversion pipeline can extract windows and labels
- The ML bridge (PyTorch/sklearn) can load the artifact

If first-batch fails, the error is almost always a data issue (missing events, malformed sidecar, LFS pointers) rather than a library bug.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-minimum-plan.png" alt="Horizontal bar chart of the ds000001 first-batch download plan and file sizes.">
  <figcaption>Real `minimum(goal='first-batch')` plan: metadata, sidecar, events, and one BOLD run.</figcaption>
</figure>

```python
plan = ds.minimum(goal='first-batch', output_dir=Path('data/ds000001'))
print(plan.to_text())
```

Result artifact: [ds000001-minimum-first-batch.txt](/Qortex/assets/results/ds000001-minimum-first-batch.txt)

<!-- qortex-evidence:end -->

## Related

- [Plan](../download/plan.md) — for custom file selections
- [First batch](first-batch.md) — detailed description of the first-batch diagnostic
