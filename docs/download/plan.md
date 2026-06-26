# Plan

`ds.plan()` constructs a `DownloadPlan` without starting any transfers. The plan lists every file that would be downloaded, its target local path, and the CDN URL.

## Basic usage

```python
from qortex import Dataset

ds = Dataset("ds004130")
plan = ds.plan(subjects=["01", "02"], tasks=["rest"], suffixes=["eeg"])

print(f"Files: {len(plan.files)}")
print(f"Size:  {plan.size_gb:.2f} GB")
print(f"Target: {plan.target_dir}")

for f in plan.files[:5]:
    print(f.path, f.size)
```

## Plan properties

| Property | Type | Description |
|----------|------|-------------|
| `files` | `list[FileRecord]` | Files that will be downloaded |
| `target_dir` | `Path` | Root directory for downloads |
| `size_gb` | `float` | Total size in GB |
| `n_subjects` | `int` | Number of subjects in plan |
| `includes_companions` | `bool` | Whether companions were added |

## Serialize the plan

```python
import json

plan_dict = plan.to_dict()
with open("plan.json", "w") as f:
    json.dump(plan_dict, f, indent=2)
```

Reload and execute later:

```python
from qortex.download import DownloadPlan

plan = DownloadPlan.from_file("plan.json")
ds.download_paths(plan.files)
```

## CLI

```bash
qortex plan ds004130 --subjects 01 02 --tasks rest --suffixes eeg
```

Output shows file count and total size. Add `--output plan.json` to save:

```bash
qortex plan ds004130 --subjects 01 02 --output plan.json
```

## Minimum plan

`ds.minimum()` builds a plan for the smallest set of subjects that enables a specific goal:

```python
plan = ds.minimum(goal="first-batch")
plan = ds.minimum(goal="label-check")
plan = ds.minimum(goal="validation")
plan = ds.minimum(goal="metadata")
```

`first-batch` picks enough subjects (usually 3–5) to run one full pipeline pass: download → convert → train step.

## Companion file inclusion

By default, any imaging file in the plan brings its companions with it. A BOLD NIfTI automatically includes its JSON sidecar and events.tsv. To suppress this:

```python
plan = ds.plan(subjects=["01"], include_companions=False)
```

Excluding companions is rarely useful — conversion will fail without sidecars and events.

## Related

- [Selective download](selective-download.md) — filter options
- [Resume](resume.md) — what happens if a transfer is interrupted mid-plan
