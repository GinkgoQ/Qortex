# Inspect

Qortex provides two levels of dataset inspection: a lightweight manifest-level profile built from the file tree (`DatasetInspector`), and a rich API-level metadata object sourced directly from the OpenNeuro GraphQL API (`RichDatasetInfo`). Use them together to get the full picture before downloading anything.

## Rich API metadata

`OpenNeuroClient.get_dataset_rich()` returns a `RichDatasetInfo` object covering every field visible on an OpenNeuro dataset page — without downloading or iterating the file tree.

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    info = client.get_dataset_rich("ds008039")

# Identity
info.id                               # "ds008039"
info.name                             # dataset name from dataset_description.json
info.doi                              # "10.18112/openneuro.ds008039.v1.0.0"
info.license                          # "CC0"
info.authors                          # ["Doe J", "Smith A", ...]
info.senior_author                    # last/senior author name

# Scientific context
info.modalities                       # ["MRI"]
info.tasks                            # ["rest", "nback"]
info.species                          # "human"
info.study_domain                     # e.g. "cognitive neuroscience"
info.study_design                     # e.g. "longitudinal"
info.data_processed                   # True/False — preprocessed data included

# Publication & funding
info.associated_paper_doi             # linked journal article DOI
info.grant_funder                     # funding agency name
info.grant_id                         # grant number

# Dates
info.created                          # ISO 8601 string: "2024-03-15T10:22:00.000Z"
info.publish_date                     # ISO 8601 string

# Community engagement
info.engagement.views                 # total page views
info.engagement.downloads             # total file downloads
info.engagement.stars                 # number of stars
info.engagement.followers             # number of followers
info.engagement.popularity_score      # composite 0–100 score

# Snapshot summary (subject counts, file counts — no file tree needed)
snap = info.latest_snapshot_summary
snap.tag                              # "1.0.0"
snap.n_subjects                       # 42
snap.n_sessions                       # 2
snap.n_tasks                          # 3
snap.subjects                         # ["sub-01", "sub-02", ...]
snap.sessions                         # ["ses-01", "ses-02"]
snap.tasks                            # ["rest", "nback"]
snap.total_files                      # 1847
snap.total_size_gb                    # 14.23
snap.bids_version                     # "1.8.0"
snap.funding                          # ["NIH R01 MH123456", ...]
snap.references                       # ["https://doi.org/...", ...]
snap.ethics_approvals                 # ["IRB approval #2024-001"]
snap.how_to_acknowledge               # citation string

# Subject demographics
snap.demographics_dataframe()         # Polars DataFrame: participant_id, age, sex, group
snap.age_stats()                      # {"n": 42, "mean": 28.4, "min": 18, "max": 65, ...}
```

Export to dict or JSON:

```python
import json
print(json.dumps(info.to_dict(), indent=2))
```

## README / description text

The text shown as "Description" on the OpenNeuro dataset page comes from the README file. Fetch it in one call:

```python
with OpenNeuroClient() as client:
    readme = client.get_readme("ds008039")           # latest snapshot
    readme = client.get_readme("ds008039", "1.0.0")  # specific version

print(readme)  # full README text
```

Returns `None` if the dataset has no README.

## BIDS validation issues

```python
with OpenNeuroClient() as client:
    issues = client.get_validation_issues("ds008039", "1.0.0")

for issue in issues:
    print(issue["severity"], issue["key"], issue["reason"])
    # "error"   MISSING_FILE   "Required file /participants.tsv not found"
```

Each issue dict has: `severity` (`"error"` or `"warning"`), `key`, `reason`, `files`, `helpUrl`.

## Manifest-level profile

`DatasetInspector.inspect()` fetches the file tree for a snapshot and builds a structural profile with BIDS entity parsing, modality breakdowns, and ML readiness scoring.

```python
from qortex import Dataset

ds = Dataset("ds004130")
profile = ds.inspect()

print(profile)           # human-readable summary
```

The returned `DatasetProfile` has:

```python
profile.dataset_ref        # DatasetRef (id, name, doi, license, authors, modalities, tasks)
profile.snapshot           # SnapshotRef (tag, hexsha, doi)
profile.all_snapshots      # list[SnapshotRef] — all available versions

# Structural counts
profile.n_subjects         # integer
profile.n_sessions         # integer
profile.n_tasks            # integer
profile.subjects           # ["sub-01", ...]
profile.sessions           # ["ses-01", ...]
profile.tasks              # ["rest", ...]

# Modality breakdown
profile.modality_breakdown  # {"eeg": ModalityBreakdown(...), ...}

# Cross-subject coverage matrices
profile.subject_task_matrix     # {"sub-01": ["rest", "nback"], ...}
profile.subject_session_matrix  # {"sub-01": ["ses-01", "ses-02"], ...}

# File counts
profile.n_signal_files      # signal-type files only
profile.n_events_files      # *_events.tsv count
profile.n_sidecar_files     # *.json sidecar count
profile.n_derivative_files  # derivatives/* count
profile.total_size_gb       # float
profile.per_subject_avg_gb  # float

# Companion coverage (fraction of signal files with companion)
profile.events_coverage     # 0.0–1.0
profile.channels_coverage   # 0.0–1.0
profile.sidecar_coverage    # 0.0–1.0

# Structure checks
profile.has_participants_tsv        # bool
profile.has_dataset_description     # bool
profile.has_readme                  # bool

# ML readiness
profile.ml_readiness        # MLReadinessScore
profile.ml_readiness.total  # 0–100
profile.ml_readiness.grade  # "A" | "B" | "C" | "D" | "F"

# Recommendations
profile.recommendations     # list[str] — human-readable suggestions

# Rich API metadata (populated when OpenNeuroClient is available)
profile.rich_info           # RichDatasetInfo or None
```

## CLI

```bash
qortex inspect ds004130
```

Output:

```
Dataset:   ds004130 — EEG resting-state
Snapshot:  1.2.0
DOI:       10.18112/openneuro.ds004130.v1.2.0
Subjects:  88
Sessions:  1
Modalities: eeg
Tasks:     rest
Files:     1,056
Size:      4.2 GB
Events:    yes (88 files, columns: onset duration trial_type)
```

Add `--json` for machine-readable output:

```bash
qortex inspect ds004130 --json
```

## All available snapshots

```python
with OpenNeuroClient() as client:
    snapshots = client.get_snapshots("ds008039")

for s in snapshots:
    print(s.tag, s.created, s.size)
```

## Local dataset

If you have a local BIDS directory and want to build a manifest from disk rather than OpenNeuro:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
ds.index_local()     # scan local tree, write .qortex/manifest.json
profile = ds.inspect()
```

CLI equivalent:

```bash
qortex local-index data/ds004130/ --dataset-id ds004130
qortex inspect ds004130 --local
```

## Related

- [`DatasetQuery`](search-catalog.md) — search and filter across all public datasets
- [`ds.metadata()`](metadata.md) — read sidecar files, events.tsv, participants.tsv
- [`ds.doctor()`](../readiness/doctor.md) — full ML readiness report
