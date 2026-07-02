## Research conclusion

Qortex should focus on one powerful promise:

> **“Tell me whether this OpenNeuro/BIDS dataset is worth using, what exactly I need to download, whether it will load, whether it has usable labels, and give me training-ready data without custom glue code.”**

That is the real user need.

The current README is strong, but it reads like a broad feature inventory. To make Qortex more impactful, the product should be organized around **decision workflows**, not just modules.

Users do not primarily want:

```text
manifest, graph, cache, parser, adapter, registry
```

They want:

```text
Can I use this dataset?
What should I download?
Will it load?
Does it have labels?
Can I train a model?
Can I reproduce this later?
Can I avoid wasting 300 GB and 2 days?
```

---

# 1. What users actually need

## 1.1 ML/AI users

AI users want to go from OpenNeuro to model training.

Their pain:

- finding usable datasets
- knowing whether labels exist
- knowing whether the dataset is classification/regression/self-supervised-ready
- avoiding huge downloads
- converting BIDS into tensors
- avoiding dataset leakage
- using PyTorch/Hugging Face/Lightning without manual parsers
- comparing datasets quickly

This is strongly supported by the EEGDash paper: public neurophysiology datasets are accessible, but turning them into trained models still requires large amounts of custom code for download, loading, repair, windowing, and evaluation; metadata-compliant datasets can still fail to load. ([arXiv][1])

## 1.2 Neuroscientists

Neuroscientists want trust and interpretability.

Their pain:

- BIDS validation is technical
- metadata inheritance is hard to reason about
- event files may exist but may not mean usable labels
- derivatives can confuse raw-data workflows
- local copies can become stale
- DataLad/git-annex can confuse non-expert users
- file symlinks/placeholders can look like broken data

The DataLad OpenNeuro guide explicitly warns that users may see confusing errors if file content has not been retrieved yet, and says the first thing to check is whether the content is actually present. ([handbook.datalad.org][2])

## 1.3 Data engineers

Data engineers want reliable, resumable, inspectable pipelines.

Their pain:

- large datasets
- partial failures
- retry/backoff behavior
- corrupted/incomplete files
- reproducibility
- cache lifecycle
- cloud/object storage
- provenance and manifests
- structured logs

OpenNeuro and openneuro-py issue discussions show real download failures, DataLad failures, S3/version-ID problems, corrupted-looking files, concurrency/backoff concerns, and size mismatch failures.

## 1.4 BIDS/OpenNeuro power users

Power users want more precise tools than the repository gives them.

Their pain:

- subset validation
- snapshot diffs
- semantic search
- annotation/search over participant metadata
- richer dataset metadata
- improved partial workflows

The BIDS Validator has an open request to validate only selected files/folders, motivated by avoiding full validation when fixing events files or adding one subject/session among hundreds.

OpenNeuro discussions also mention NeuroBagel-style annotations/search over OpenNeuro datasets, including participant annotations and richer search infrastructure.

---

# 2. What the current README already does well

Your README already contains the right core thesis:

- Qortex is not a thin downloader.
- It builds a semantic layer over OpenNeuro manifests.
- It reasons about logical recordings, companions, sidecars, labels, loadability, provenance, and ML outputs.
- It treats BIDS files as semantic units, not isolated paths.
- It separates manifest-level label candidates from confirmed local labels.
- It includes planning, readiness, EDA, conversion, artifact manifests, and training adapters.

That is correct.

The best part is this concept:

```text
Semantic logical recording graph
```

That should become the core intellectual identity of Qortex.

---

# 3. Main weakness in the README

The README is too feature-list-heavy.

It says many things, but the user value is buried.

Current style:

```text
Qortex implements catalog, manifest, graph, fetch engine, loaders, EDA, conversion...
```

Better product framing:

```text
Qortex answers practical questions before users waste time:
Can I use this dataset?
What should I download?
Will it validate?
Will it load?
Does it contain usable labels?
Can I convert it to ML-ready data?
Can I reproduce the exact pipeline later?
```

The README should move from **module inventory** to **decision-oriented workflows**.

---

# 4. The strongest product angle

## Qortex should become the “readiness and conversion layer” for neurodata

OpenNeuro is the archive.

BIDS is the standard.

DataLad is the versioned retrieval engine.

PyBIDS/MNE/NiBabel/Nilearn are domain parsers.

EEGDash is emerging as a neurophysiology ML-access layer.

Qortex should sit above all of them:

```text
OpenNeuro/BIDS → readiness intelligence → selective access → verified local lake → ML artifact
```

The word that matters is:

```text
readiness
```

Not just validation. Not just download. Not just conversion.

Qortex should tell the user:

```text
This dataset is ready for X, not ready for Y, and here is the minimum action needed.
```

---

# 5. Highest-impact feature directions

## 5.1 `qortex doctor`

This should be a signature feature.

Purpose:

```text
Diagnose an OpenNeuro/BIDS dataset before download, after download, before conversion, and before training.
```

Example:

```bash
qortex doctor ds000246 --task classification --modality meg
```

Output:

```text
Dataset: ds000246
Goal: MEG classification

Status: Partially ready

Can inspect remotely: yes
Needs download: 18.4 GB
Loadable recordings: 42/48
Event-complete recordings: 39/48
Confirmed labels: unknown until events files are downloaded
Minimum download for label check: 12 MB
Minimum download for smoke training: 1.2 GB

Recommended next step:
qortex download ds000246 --modality meg --events-only
```

Why it grabs attention:

- users immediately understand it
- simple but clever
- converts technical complexity into a decision
- useful before downloading large data

---

## 5.2 `qortex scout`

Dataset discovery should be smarter than search.

Purpose:

```text
Find datasets that match an ML/research goal.
```

Example:

```bash
qortex scout --goal eeg-classification --min-subjects 30 --max-gb 50
```

Output:

```text
Top candidates:
1. dsXXXXXX — 42 subjects, EEG, event labels likely, 28 GB, readiness 84
2. dsYYYYYY — 65 subjects, EEG, labels confirmed from cached metadata, 44 GB, readiness 79
3. dsZZZZZZ — 31 subjects, EEG, missing channels.tsv in 6 recordings, readiness 62
```

This is more useful than a normal catalog.

The EEGDash paper’s metadata-first registry, semantic search, loadability/compliance metadata, and dataset-level tags show that dataset discovery is a real gap in public neurophysiology ML workflows. ([arXiv][1])

---

## 5.3 `qortex minimum`

Users often do not need the full dataset.

Purpose:

```text
Compute the smallest useful download for a goal.
```

Examples:

```bash
qortex minimum ds000246 --goal validate
qortex minimum ds000246 --goal label-check
qortex minimum ds000246 --goal smoke-train
qortex minimum ds000246 --goal full-train --modality eeg
```

Output:

```text
Goal: label-check
Required files:
- dataset_description.json
- participants.tsv
- *_events.tsv
- event sidecars
Estimated size: 7.6 MB
```

Why it matters:

- OpenNeuro datasets can be large
- many users just need to know if labels exist
- avoids waste
- immediately valuable

This aligns with DataLad’s strength: users can retrieve only specific paths such as `sub-01/anat/`, and full dataset content is not needed immediately. ([handbook.datalad.org][2])

---

## 5.4 Label Intelligence

This should be a major differentiator.

Current README already says Qortex is conservative about labels. Keep that.

Add a full label system:

```text
No events file
Events present but no label columns
Candidate labels found
Confirmed labels found
Ambiguous labels
Continuous targets
Trial-level labels
Subject-level labels
Session-level labels
Regression target
Classification target
Self-supervised only
```

API:

```python
ds.labels().summary()
ds.labels().candidates()
ds.labels().confirmed()
ds.labels().recommend_target()
```

CLI:

```bash
qortex labels ds000246 --download-minimal
```

Why it matters:

AI users care more about labels than BIDS file names.

---

## 5.5 Loadability Index

Validation is not enough.

A dataset can be BIDS-valid but still hard to load. EEGDash explicitly makes this point: metadata-compliant datasets can still fail to load. ([arXiv][1])

Qortex should define:

```text
BIDS-valid ≠ loadable ≠ ML-ready
```

Scores:

| Score             | Meaning                           |
| ----------------- | --------------------------------- |
| BIDS score        | standard compliance               |
| Loadability score | can domain loaders open the data? |
| Label score       | usable target availability        |
| Conversion score  | can Qortex produce an artifact?   |
| Training score    | can a dataloader produce batches? |

Signature output:

```text
Qortex Readiness:
BIDS      91
Load      73
Labels    64
Convert   78
Train     70
Overall   75
```

---

## 5.6 First Batch Guarantee

This is a simple but powerful idea.

Purpose:

```text
Qortex should not claim success until it can produce the first ML batch.
```

CLI:

```bash
qortex first-batch ds000246 --modality eeg --target trial_type
```

Output:

```text
Success.
X: torch.Size([32, 64, 1000])
y: torch.Size([32])
subjects: 12
split: train
source files: 14
```

Why it matters:

- ML users trust actual tensors more than reports
- it is a memorable feature
- it proves the pipeline end-to-end
- it exposes problems early

---

## 5.7 Explainable download plans

The current README already has per-file `SelectionReason`. Make it central.

Users should see:

```text
Why is this file included?
```

Example:

```text
sub-01/eeg/sub-01_task-rest_eeg.set
  selected because: primary EEG recording

sub-01/eeg/sub-01_task-rest_events.tsv
  selected because: required event companion for selected recording

sub-01/eeg/sub-01_task-rest_channels.tsv
  selected because: channel metadata required for loadability

dataset_description.json
  selected because: essential BIDS metadata
```

This is clever because BIDS users often do not know which sidecars are necessary.

---

## 5.8 Dataset Recipe Export

Users need reproducibility.

Add:

```bash
qortex recipe ds000246 --modality eeg --task rest --target trial_type
```

Output:

```yaml
dataset: ds000246
snapshot: 1.0.0
selection:
  modality: eeg
  task: rest
  require_events: true
  require_labels: true
conversion:
  window_seconds: 10
  split: subject
  target: trial_type
output:
  format: parquet
```

Then:

```bash
qortex run recipe.yaml
```

This becomes a shareable artifact in papers and repos.

---

## 5.9 Dataset diff that users understand

OpenNeuro snapshots are versioned. OpenNeuro docs say snapshots are represented as Git tags. ([docs.openneuro.org][3])

Qortex should expose semantic diffs:

```bash
qortex diff ds000246 1.0.0 1.0.1
```

Not just files changed.

Output:

```text
Added:
- 2 subjects
- 4 EEG recordings
- 4 events files

Changed:
- participants.tsv
- task-rest EEG sidecar changed sampling frequency metadata

Impact:
- ML artifact should be regenerated
- previous train/test split is no longer valid
```

This is much more useful than raw Git diff.

---

## 5.10 Dataset Card Auto-Generation

Add:

```bash
qortex card ds000246 --output qortex-card.md
```

Sections:

- dataset identity
- snapshot
- DOI
- modalities
- subjects
- sessions
- tasks
- label candidates
- readiness score
- minimal download options
- known issues
- recommended ML tasks
- citation/provenance

This is simple but high-impact for GitHub, papers, and Hugging Face.

---

# 6. What forums/issues suggest users struggle with

## 6.1 Download reliability

Evidence:

- openneuro-py has an open issue about concurrency not behaving as expected and backoff not reducing server pressure globally.
- openneuro-py has an open issue where size mismatch crashes the entire download instead of retrying or continuing.
- openneuro-py has an open issue about creating one HTTP client per file, which can increase socket/file-descriptor pressure for large datasets.
- OpenNeuro issues include failed downloads, DataLad errors, S3/private remote/version-ID problems, and corrupted-looking files.

Qortex response:

- global backoff
- shared HTTP session
- per-file failure isolation
- resumable lockfile
- “explain failure” output
- verify local content status
- “repair incomplete download” command

---

## 6.2 Subset validation

Evidence:

- BIDS Validator users want to validate specific files/folders, for example after fixing `_events` files or adding a subject/session, without waiting for validation of unrelated data.

Qortex response:

```bash
qortex validate-subset ./ds --paths sub-01/eeg --with-inheritance
```

Add:

```text
subset validation context builder
```

This should include inherited sidecars and essential metadata automatically.

---

## 6.3 DataLad complexity

Evidence:

- DataLad is powerful, but its OpenNeuro guide has to warn users not to copy symlinks incorrectly, not to force-overwrite annexed files, and to ensure content is retrieved before opening files because tools may emit confusing errors otherwise. ([handbook.datalad.org][2])

Qortex response:

```bash
qortex materialize ds000246 --policy copy
qortex content-status ./ds
qortex explain-placeholder ./ds/sub-01/anat/file.nii.gz
```

A very useful feature:

```text
“Your file is not corrupted; it is a DataLad/git-annex pointer and the content has not been retrieved.”
```

This directly addresses a real confusion pattern.

---

## 6.4 Search and annotation

Evidence:

- OpenNeuro issue discussion mentions NeuroBagel-style annotation/search over OpenNeuro datasets and participant annotation improvements.
- OpenNeuro’s GraphQL API exposes dataset/snapshot/file metadata, recursive file trees, file IDs, sizes, directory flags, and annexed flags, which Qortex can use to create a richer discovery layer. ([docs.openneuro.org][4])

Qortex response:

```bash
qortex scout
qortex annotate-cache
qortex recommend
qortex find-labels
```

---

# 7. Features that can grab attention

These are simple but clever.

## 7.1 “Can I train?” command

```bash
qortex can-train ds000246 --modality eeg --target trial_type
```

Output:

```text
Yes, with limitations.

Minimum download: 14.2 GB
Confirmed labels: no, needs event-file inspection
Likely labels: trial_type
Suggested split: subject
Leakage risk: low
Recommended converter:
qortex convert ds000246 --modality eeg --target trial_type --split subject
```

## 7.2 “Before you download” report

```bash
qortex before-download ds000246
```

Output:

```text
Do not download full dataset yet.
First download metadata + events: 9 MB.
Reason: label readiness unknown.
```

## 7.3 “One subject smoke test”

```bash
qortex smoke ds000246 --modality meg
```

Downloads the smallest complete semantic recording and tries:

```text
download → validate subset → load → convert → produce first batch
```

## 7.4 “Dataset risk radar”

```text
Risk:
- large derivatives dominate size
- labels not confirmed
- 17 recordings missing channels.tsv
- fMRI TR inconsistent across runs
- local files stale relative to snapshot
```

## 7.5 “Semantic subset”

Instead of:

```bash
--include sub-01/**/*.tsv --include sub-01/**/*.json ...
```

Use:

```bash
qortex download ds000246 --recordings 10 --label-ready --modality eeg
```

Qortex should decide the required files.

## 7.6 “ML artifact inspector”

```bash
qortex artifact inspect ./qortex-artifact
```

Output:

```text
Format: Parquet
Samples: 12,430
Subjects: 48
Splits: train/val/test
Input shape: [channels, time]
Target: trial_type
Source dataset: ds000246@1.0.0
```

## 7.7 “Leakage guard”

Before training:

```bash
qortex leakage-check artifact/
```

Detect:

- same subject in train and test
- same session split across train/test
- duplicate files in multiple splits
- derivatives generated from all subjects before splitting
- time-window leakage

This is a strong ML-engineering differentiator.

## 7.8 “Citation/provenance helper”

```bash
qortex cite artifact/
```

Output:

```text
Dataset DOI:
Qortex recipe:
Snapshot:
Source files:
Conversion hash:
```

This matters for papers.

---

# 8. Missing features to add to README

The README should add a section called:

```text
User Questions Qortex Answers
```

Add:

| User question                         | Qortex answer                    |
| ------------------------------------- | -------------------------------- |
| Is this dataset worth using?          | `qortex doctor`                  |
| What is the smallest useful download? | `qortex minimum`                 |
| Does it have labels?                  | `qortex labels`                  |
| Will it load?                         | `qortex check --inspect-loaders` |
| Can I train?                          | `qortex can-train`               |
| Can I get one real batch?             | `qortex first-batch`             |
| What changed between snapshots?       | `qortex diff`                    |
| Why was this file downloaded?         | `DownloadPlan.explain()`         |
| Can I reproduce this later?           | `qortex recipe`                  |
| Is my local copy stale or incomplete? | `qortex content-status`          |
| Is this DataLad file only a pointer?  | `qortex explain-placeholder`     |

---

# 9. Impact ranking

## Highest impact, highest attention

| Rank | Feature                | Why                                         |
| ---: | ---------------------- | ------------------------------------------- |
|    1 | `qortex doctor`        | Instantly understandable, decision-oriented |
|    2 | `qortex can-train`     | Speaks directly to AI/ML users              |
|    3 | `qortex minimum`       | Saves time/disk/bandwidth                   |
|    4 | Label Intelligence     | ML users care about targets                 |
|    5 | First Batch Guarantee  | Proves end-to-end value                     |
|    6 | Loadability Index      | Goes beyond BIDS validation                 |
|    7 | Dataset Recipe         | Reproducibility and sharing                 |
|    8 | Leakage Guard          | Serious ML credibility                      |
|    9 | Semantic Snapshot Diff | Strong reproducibility story                |
|   10 | Explain Placeholder    | Solves DataLad/git-annex confusion          |

---

# 10. How to make Qortex more impactful

## 10.1 Own the phrase “ML-readiness”

Qortex should consistently use:

```text
ML-readiness
```

Not only:

```text
validation
```

Because BIDS validation is already owned by BIDS Validator.

Qortex owns:

```text
Can this dataset become a reliable ML artifact?
```

## 10.2 Do not compete with every tool

Qortex should not replace:

- OpenNeuro
- BIDS Validator
- DataLad
- PyBIDS
- MNE-BIDS
- NiBabel
- Nilearn
- Braindecode
- EEGDash

Qortex should orchestrate them and add missing intelligence:

```text
semantic planning + readiness + conversion + reproducibility
```

## 10.3 Make the first 5 minutes magical

The best first-user experience:

```bash
pip install qortex
qortex doctor ds000246
qortex minimum ds000246 --goal first-batch --modality eeg
qortex first-batch ds000246 --modality eeg
```

If this works, users understand Qortex immediately.

## 10.4 Be honest about uncertainty

Qortex should explicitly say:

```text
Labels are candidate-only until local event files are inspected.
```

This is excellent and should remain a product principle.

## 10.5 Build a public “Qortex Readiness Index”

This can become the community contribution.

```text
Qortex Readiness Index for OpenNeuro
```

A generated catalog:

| Dataset | Modality | Subjects | Loadability | Label readiness | ML-readiness |
| ------- | -------: | -------: | ----------: | --------------: | -----------: |

This could attract researchers because it saves everyone time.

---

# 11. README restructuring recommendation

Current README order:

```text
What Qortex Implements
Public Python Methods
Selection and Planning
Semantic Manifest Graph
Readiness Analysis
...
```

Better order:

```text
# Qortex

## What problem Qortex solves

## User questions Qortex answers

## Quick examples

## Core workflows

1. Discover
2. Diagnose
3. Plan minimum download
4. Download
5. Validate/check
6. Convert
7. Train
8. Reproduce

## Core concepts

- Manifest
- LogicalRecording
- DownloadPlan
- ReadinessReport
- Artifact
- Recipe

## Features

## Current boundaries

## Roadmap
```

This makes the README user-first, not implementation-first.

---

# 12. Revised core positioning

Use this positioning:

```text
Qortex is a readiness-first neurodata platform for OpenNeuro and BIDS datasets.

It helps users decide whether a dataset is usable before they download it, retrieves only the files needed for a goal, checks BIDS/loadability/label readiness, converts semantic recordings into ML artifacts, and produces reproducible training-ready outputs for modern ML frameworks.
```

Short version:

```text
Qortex turns OpenNeuro/BIDS datasets into verified, minimal, ML-ready artifacts.
```

Even shorter:

```text
From OpenNeuro to first ML batch.
```

That last phrase is strong.

---

# 13. What to implement next

Do not add more broad modules first.

Implement these in this order conceptually:

1. `doctor`
2. `minimum`
3. `labels`
4. `first-batch`
5. `leakage-check`
6. `recipe`
7. `content-status`
8. `snapshot diff`
9. `readiness index`
10. dashboard after the above APIs are stable

This is better than building dashboard/cloud/export too early.

---

# 14. Best next README section to add

Add this section near the top:

```markdown
## What Qortex Answers

Qortex is designed around practical dataset decisions:

| Question                                        | Qortex workflow                              |
| ----------------------------------------------- | -------------------------------------------- |
| Is this dataset worth using?                    | `qortex doctor ds000246`                     |
| What is the smallest useful download?           | `qortex minimum ds000246 --goal label-check` |
| Does it contain usable labels?                  | `qortex labels ds000246`                     |
| Will the files load with real domain libraries? | `qortex check ds000246 --inspect-loaders`    |
| Can I produce a training batch?                 | `qortex first-batch ds000246 --modality eeg` |
| Why is each file included in my download?       | `qortex plan ds000246 --explain`             |
| Can I reproduce this pipeline later?            | `qortex recipe ds000246`                     |
```

This section will immediately make Qortex feel different from `openneuro-py`.

---

# 15. Final product direction

Qortex should not be advertised as:

```text
A Python library for OpenNeuro downloads.
```

It should be advertised as:

```text
A readiness-first neurodata operating layer that turns OpenNeuro/BIDS datasets into minimal, validated, reproducible, ML-ready artifacts.
```

The most memorable product promise:

```text
From OpenNeuro to first ML batch.
```

That is the feature story that can grab attention.

[1]: https://arxiv.org/abs/2606.16041?utm_source=chatgpt.com "EEGDash: An open-source platform for machine learning on public neurophysiological data"
[2]: https://handbook.datalad.org/en/latest/usecases/openneuro.html "OpenNeuro Quickstart Guide: Accessing OpenNeuro datasets via DataLad — The DataLad Handbook"
[3]: https://docs.openneuro.org/git.html "Git access to OpenNeuro datasets - OpenNeuro documentation"
[4]: https://docs.openneuro.org/api.html "API Examples - OpenNeuro documentation"
