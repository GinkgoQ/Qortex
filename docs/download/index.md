# Download

Qortex downloads files from the OpenNeuro CDN to a local directory. The download engine supports resume, companion file inclusion, and selective filtering.

## Core principle: plan before you download

Qortex separates planning from execution. `ds.plan()` returns a `DownloadPlan` that lists exact files with target paths. You can inspect the plan before committing to the transfer.

```python
plan = ds.plan(subjects=["01", "02"], tasks=["rest"])
print(f"{len(plan.files)} files, {plan.size_gb:.1f} GB")
ds.download_paths(plan.files)
```

Or combine planning and execution:

```python
ds.download(subjects=["01", "02"], tasks=["rest"], data_dir="data/ds004130/")
```

## What you can do

[**Plan**](plan.md) — Create a DownloadPlan and inspect it before downloading.

[**Selective download**](selective-download.md) — Filter by subject, session, task, run, suffix, and size.

[**Metadata only**](metadata-only.md) — Download only sidecar JSON, events.tsv, bval/bvec, and participants files.

[**Cache**](cache.md) — Where files are stored, how the cache works, and how to clear it.

[**Local index**](local-index.md) — Scan a pre-existing BIDS directory and build a local manifest.

[**Resume**](resume.md) — How interrupted downloads recover.

---

**Next →** [Visualize](../visualization/index.md) — QC your local files before conversion to catch bad data early.








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
