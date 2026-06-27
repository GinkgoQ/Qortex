# OpenNeuro and BIDS

Qortex is built specifically for OpenNeuro datasets that follow the BIDS standard. Understanding how these two relate clarifies what Qortex can and cannot do.

## OpenNeuro

OpenNeuro is a platform for sharing neuroimaging data. Every dataset has a unique ID (e.g., `ds004130`) and is versioned through snapshots.

Qortex communicates with OpenNeuro through its GraphQL API via `OpenNeuroClient`. This covers:

- Full dataset metadata (name, DOI, license, authors, funding, associated paper DOI)
- Community engagement (views, downloads, stars, followers)
- Snapshot summaries (subject count, session list, task list, file count, total size, BIDS version)
- Subject demographics (age, sex, group) from the API-level summary
- README text (the description shown on the OpenNeuro dataset page)
- BIDS validation issues (errors and warnings)
- CDN download URLs for individual files

You do not need an account to read public datasets. Private datasets or embargo-period access require an API token.

## Snapshots

A snapshot is a versioned, immutable copy of a dataset. Once published, a snapshot's content does not change. Snapshots are tagged with version strings like `1.0.0` or `2.1.3`.

When you create a `Dataset` object without specifying a snapshot, Qortex uses the latest published snapshot. To pin a specific version:

```python
from qortex import Dataset
ds = Dataset("ds000001", snapshot="1.0.3")
```

Pinning a snapshot ensures reproducibility. The same snapshot always has the same files at the same CDN URLs.

## BIDS structure

BIDS (Brain Imaging Data Structure) is a naming and organization convention for neuroimaging data. A BIDS dataset has a predictable directory tree:

```
dataset/
  participants.tsv
  dataset_description.json
  sub-01/
    anat/
      sub-01_T1w.nii.gz
      sub-01_T1w.json
    func/
      sub-01_task-rest_bold.nii.gz
      sub-01_task-rest_bold.json
      sub-01_task-rest_events.tsv
    eeg/
      sub-01_task-rest_eeg.set
      sub-01_task-rest_eeg.json
      sub-01_task-rest_channels.tsv
  sub-02/
    ...
```

Each file's name encodes BIDS entities: subject (`sub-`), session (`ses-`), task (`task-`), run (`run-`), and suffix (the last `_word` before the extension).

## How Qortex uses BIDS

Qortex parses BIDS entity labels from file paths in the manifest. It does not run a full BIDS parser on the remote tree — it extracts entities from path components using string rules.

When you call `ds.files(subjects=["01", "02"], tasks=["rest"])`, Qortex filters the manifest by matching the extracted subject and task labels. The results contain all files whose paths include `sub-01/` or `sub-02/` and contain `task-rest_`.

Companion files are resolved by matching the stem of the primary file. A file named `sub-01_task-rest_bold.nii.gz` has companions at paths like `sub-01_task-rest_bold.json`, `sub-01_task-rest_events.tsv`, and `sub-01_task-rest_desc-confounds_timeseries.tsv`.

## BIDS derivatives

Derivatives (preprocessed outputs from tools like fMRIPrep or FreeSurfer) live in a `derivatives/` subdirectory. Qortex excludes derivatives by default. To include them:

```python
ds.download(include_derivatives=True)
# or
ds.plan(include_derivatives=True)
```

## What requires an account

Reading public datasets: no account needed.

Downloading files: no account needed for public datasets.

Accessing private or embargoed datasets: requires an API token stored with `qortex login`.
