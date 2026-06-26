# Readiness First

The central design decision in Qortex is that readiness checks happen before any large data transfer. This is not a convenience — it is the reason Qortex exists.

## Why readiness checks matter

Downloading an OpenNeuro dataset takes time and disk space. Many datasets, when fully downloaded, turn out to be unusable for the intended task. The events.tsv files may be missing. The label column may have only one class. The subjects may be too few to split. The NIfTI files may be pointer-like Git LFS objects that were not properly fetched.

Running these checks against the remote manifest — before transfer — costs almost nothing. The manifest contains the file tree. The sidecar JSON files are small. Qortex reads exactly what it needs.

## The five readiness states

A dataset in Qortex moves through distinct states:

**Not inspected.** Only the dataset ID is known. No manifest has been fetched.

**Manifest-only.** The remote file tree has been fetched. Qortex knows which subjects, sessions, tasks, and modalities exist. It knows which files have events.tsv, bval/bvec, or JSON sidecar companions. No data is on local disk.

**Downloaded.** Files exist locally. Content may still be incomplete (partial download, Git LFS pointers).

**Validated.** The local BIDS structure has passed the official BIDS Validator. File counts match the manifest.

**Conversion-ready.** All required companions are present. Windows can be extracted. Splits are feasible.

## What readiness checks return

Each check returns a structured report with a state, findings, and a recommended next action. The reports are machine-readable (Pydantic models, JSON-serializable) so they can be embedded in pipelines.

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor()
print(report.to_text())
# State: manifest_only
# Subjects: 88  Sessions: 1  Modalities: eeg  Events: yes (88 files)
# Next action: download — use ds.minimum(goal="first-batch") to get the smallest subset
```

## The minimum download principle

`ds.minimum(goal="first-batch")` computes the smallest real download that would let you extract one batch of training samples. This is not a rough estimate — it lists the exact files by path.

Goals:
- `first-batch` — enough data to verify the full pipeline end-to-end
- `label-check` — enough events.tsv files to verify label coverage
- `validation` — a representative subset for BIDS validation
- `metadata` — metadata and sidecar files only, no large imaging files

## Companion-file awareness

BIDS companion files travel with their primary files. A DWI NIfTI is not useful without its bval/bvec files. An fMRI run is not label-ready without its events.tsv. Qortex tracks companion relationships in the manifest and includes them automatically in any download plan.

When you call `ds.download(suffixes=["bold"])`, the plan automatically includes the JSON sidecar and events.tsv for each selected BOLD file.
