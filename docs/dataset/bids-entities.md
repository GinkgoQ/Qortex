# BIDS Entities

BIDS entity labels are parsed from file paths in the manifest. Qortex uses these labels for filtering files, computing coverage, and checking companion relationships.

## Entity labels

A BIDS file path encodes several key-value pairs called entities. For example:

```
sub-01/func/sub-01_ses-01_task-rest_run-01_bold.nii.gz
```

This file has:

| Entity | Key | Value |
|--------|-----|-------|
| Subject | `sub` | `01` |
| Session | `ses` | `01` |
| Task | `task` | `rest` |
| Run | `run` | `01` |
| Datatype | (directory) | `func` |
| Suffix | (before extension) | `bold` |
| Extension | | `.nii.gz` |

## Entities Qortex parses

- `subject` — always present; extracted from `sub-{label}/` in the path
- `session` — present if the path contains `ses-{label}/`
- `task` — present if the filename contains `task-{label}_`
- `run` — present if the filename contains `run-{label}_`
- `acquisition` — present if the filename contains `acq-{label}_`
- `direction` — present if the filename contains `dir-{label}_`
- `echo` — present if the filename contains `echo-{label}_`
- `part` — present if the filename contains `part-{label}_`
- `datatype` — directory containing the file (`anat`, `func`, `eeg`, `dwi`, `perf`, `pet`, `meg`, `ieeg`, `beh`)
- `suffix` — the last underscore-delimited segment before the file extension
- `extension` — file extension including the dot (`.nii.gz`, `.tsv`, `.json`, `.set`, `.fif`, `.edf`, `.bval`, `.bvec`)

## Accessing entity labels

From a FileRecord:

```python
manifest = ds.manifest()
for f in manifest.files:
    print(f.subject, f.task, f.suffix)
```

From a filter call:

```python
bold_files = ds.files(subjects=["01", "02"], tasks=["rest"], suffixes=["bold"])
```

## Filtering rules

When you pass a list to `subjects=`, `tasks=`, `suffixes=`, or `sessions=`:

- `None` means "no filter on this entity" (include all values)
- An empty list `[]` means "exclude all" (returns nothing)
- A list of strings means "include only these values"

```python
# Only resting-state BOLD from subjects 01–05
ds.files(
    subjects=["01", "02", "03", "04", "05"],
    tasks=["rest"],
    suffixes=["bold"],
)

# All EEG files regardless of task
ds.files(datatypes=["eeg"])

# All anatomical files
ds.files(datatypes=["anat"])
```

## Companion file relationships

Qortex uses entity labels to resolve companions. The companion of `sub-01_task-rest_bold.nii.gz` is:

- `sub-01_task-rest_bold.json` — same entities, suffix `bold`, extension `.json`
- `sub-01_task-rest_events.tsv` — same entities, suffix `events`, extension `.tsv`
- `sub-01_task-rest_desc-confounds_timeseries.tsv` — same entities (if present)

For DWI, the companions of `sub-01_dwi.nii.gz` are:

- `sub-01_dwi.bval`
- `sub-01_dwi.bvec`
- `sub-01_dwi.json`

When you download a NIfTI file, Qortex includes its companions in the download plan by default. Suppress this with `include_companions=False` in the download call.

## Entities that Qortex does not use

Qortex does not parse `space`, `res`, `den`, `hemi`, `label`, or `desc` entities. Files with these entities are still included in the manifest and can be downloaded, but the entities are not parsed for filtering or companion matching.
