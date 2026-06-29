# Get Started

Qortex turns an OpenNeuro dataset ID into a training-ready artifact. It checks whether a dataset is actually usable — before you download a single file.

---

## New to Qortex?

Follow this path. It takes about 20 minutes.

**1 · Install**

```bash
pip install qortex
```

→ [Full install instructions and optional extras](install.md)

**2 · Run the quickstart**

Inspect a real dataset, check its readiness, download a minimal subset, and convert to Parquet — 20 lines of code.

→ [Quickstart](quickstart.md)

**3 · Try a hands-on tutorial**

Pick a modality you work with and follow a complete end-to-end example with a real open dataset.

→ [Tutorial index](../tutorials/index.md)

---

## Already installed?

| I want to… | Go to |
|---|---|
| Work through a real dataset end-to-end | [Tutorials](../tutorials/index.md) |
| Check if a dataset is usable | [Assess readiness](../readiness/index.md) |
| Download only what I need | [Download guide](../download/index.md) |
| Convert data to Parquet / Zarr / HDF5 | [Conversion guide](../conversion/index.md) |
| Look up what a function returns | [API reference](../api/index.md) |
| Fix a specific error | [Troubleshooting](../troubleshooting/index.md) |

---

## How Qortex works

Qortex is a **readiness layer** — not a downloader, not a training framework. It sits between OpenNeuro and your pipeline.

```
OpenNeuro API
     ↓
  inspect      ← manifest only, no download
  assess       ← label coverage, class balance, split feasibility
  plan         ← exact file list + byte count for your goal
     ↓
  download     ← selective, resumable
  visualize    ← QC before converting
  convert      ← Parquet / Zarr / HDF5 / HuggingFace / TFRecord
     ↓
ML artifact    ← subject-level splits, leakage verified, provenance recorded
```

The key principle: **every check runs on the manifest and sidecar files — not on imaging data.** You know whether a dataset is usable before paying for the download.

→ [Core concepts](../concepts/index.md) — readiness-first design, data model, the full workflow

---

## Visual audit (first practical step)

After install, the fastest way to understand what Qortex does is to run a visual audit on a local BIDS dataset.

→ [First visual audit](first-visual-audit.md)
