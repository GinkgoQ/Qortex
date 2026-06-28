# What is Qortex

Qortex is a Python library for preparing OpenNeuro neuroimaging datasets for machine learning. It is not a download manager. It is a decision layer.

## The problem it solves

OpenNeuro hosts over 1,700 public BIDS datasets. Many are not usable for supervised learning. Before you can tell whether a dataset is usable, you need to know:

- Do the fMRI files have events.tsv companions with parseable trial types?
- Are there enough subjects to split into train/val/test without leakage?
- Do the DWI files have bval/bvec companions?
- Is the total size within your storage budget?

None of this requires downloading the data. The OpenNeuro manifest contains the file tree, and the sidecar JSON files are small enough to fetch over CDN without downloading the NIfTI volumes.

Qortex reads the manifest first and answers these questions before any bulk transfer.

## What Qortex is not

**Not a download manager.** Tools like DataLad and the OpenNeuro CLI handle download protocol details. Qortex uses HTTP downloads with resume support, but its value is in the decisions around what to download, not the transfer mechanism itself.

**Not a preprocessing pipeline.** Qortex converts to ML-ready formats (Parquet, Zarr, HDF5) but does not run fMRIPrep, FreeSurfer, or MNE preprocessing. It works with whatever the dataset contains — raw or preprocessed.

**Not a BIDS validator.** Qortex wraps the official BIDS Validator for structural correctness checks. It does not reimplement BIDS validation rules.

## What it does

Qortex has five main functions:

1. **Inspect** — Read the remote manifest. Report subjects, sessions, tasks, modalities, companion coverage, and total size.

2. **Decide** — Run readiness checks (doctor, minimum, can-train, first-batch) to answer whether a dataset is usable for a specific goal and what the minimum download is.

3. **Download** — Transfer files with subject/modality/task filters, resume support, and companion-file awareness.

4. **Convert** — Write ML-format artifacts (Parquet, Zarr, HDF5, WebDataset, HuggingFace, TFRecord) with sliding or event-aligned windows, subject-level splits, and provenance metadata.

5. **Inspect visually** — Render thumbnails, QC panels, and overlays from local files without loading full volumes. One center slice per NIfTI.

## Scope

Qortex works with any BIDS-formatted data stored on OpenNeuro. It has been designed primarily around:

- fMRI (BOLD, resting-state)
- EEG/MEG/iEEG
- Structural MRI (T1w, T2w)
- DWI
- PET

Surface data (GIFTI/CIFTI) is partially supported: typed inspection and QC summaries work for GIFTI meshes/scalars/labels and CIFTI dense matrices. Advanced workbench-style surface interaction and volume-to-surface projection are still future work.
