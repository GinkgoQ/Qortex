# Doctor

`ds.doctor()` is the first readiness check to run on any new dataset. It fetches the manifest and a sample of sidecar files, then produces a structured report covering structural properties and potential blockers.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor()
print(report.to_text())
```

Example output:

```
Dataset:      ds004130 — Resting-state EEG
Snapshot:     1.2.0
State:        manifest_only
Subjects:     88    Sessions: 1
Modalities:   eeg
Tasks:        rest
Events:       yes — 88 files, columns: onset duration trial_type
Labels:       trial_type → 3 classes (rest, eyes-open, task)
Size:         4.2 GB
Split check:  ok — 88 subjects supports 70/15/15 split (≥ 3 per class in test)

Next action:  download
  Use ds.minimum(goal="first-batch") to download 3 subjects for a pipeline test.
```

## CLI

```bash
qortex doctor ds004130
qortex doctor ds004130 --snapshot 1.0.0
qortex doctor ds004130 --json
```

## What doctor checks

**Manifest availability.** Can the manifest be fetched from OpenNeuro? If not, the state is `api_error`.

**Subject count.** Are there enough subjects to form train/val/test splits? Threshold defaults to 10. Configurable with `--min-subjects`.

**Companion coverage.** For BOLD files: are events.tsv present? For DWI: are bval/bvec present? For EEG: is channels.tsv present?

**Events file validity.** Are events.tsv files non-empty? Do they have an `onset` column? Is there a parseable label column?

**Label class coverage.** After selecting the best label column, how many classes are there? Are there enough samples per class?

**Split feasibility.** Can the subject count support a 70/15/15 train/val/test split with at least 3 subjects per class in test?

**Size estimate.** Total compressed size from manifest records.

## Findings and severity levels

Each finding has a `severity`:

- `info` — informational, not a blocker
- `warning` — may cause issues; review before proceeding
- `error` — blocks the next pipeline stage

```python
for f in report.findings:
    print(f"{f.severity.upper()}: [{f.code}] {f.message}")
```

Common error codes:

| Code | Meaning |
|------|---------|
| `NO_EVENTS` | No events.tsv files found |
| `EMPTY_EVENTS` | Events files exist but are empty or have no trial_type |
| `NO_COMPANIONS` | DWI files missing bval/bvec |
| `TOO_FEW_SUBJECTS` | Not enough subjects for splits |
| `ONE_CLASS` | Only one label class in events |
| `LFS_POINTERS` | Files appear to be Git LFS pointers (local only) |

## Doctor after download

Running `doctor()` after a local download also checks content:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
report = ds.doctor()
# Checks file count vs manifest, LFS pointer detection, JSON sidecar parsability
```

## Related

- [Minimum](minimum.md) — smallest download for a specific goal
- [Can train](can-train.md) — binary label readiness check
- [Content status](content-status.md) — post-download content integrity
