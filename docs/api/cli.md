# CLI Reference

All commands are accessed via `qortex <command> [options]`.

## Dataset commands

### search

Search the local catalog for datasets by modality, task, and size.

```bash
qortex search [--modality MOD] [--task TASK] [--min-subjects N]
              [--max-size GB] [--query TEXT] [--json] [--csv]
```

### inspect

Fetch and display the dataset manifest summary.

```bash
qortex inspect DATASET_ID [--snapshot VER] [--local] [--json]
```

### metadata

Print dataset metadata: dataset_description.json, participants.tsv, or a specific sidecar.

```bash
qortex metadata DATASET_ID [FILE_PATH] [--participants] [--snapshots] [--json]
```

### readme

Fetch and print the README / description text for a dataset.

```bash
qortex readme DATASET_ID [--snapshot VER]
```

### validation-issues

Print BIDS validation issues for a snapshot.

```bash
qortex validation-issues DATASET_ID --snapshot VER [--errors-only] [--json]
```

### preview

Show the first N rows of a tabular file.

```bash
qortex preview DATASET_ID FILE_PATH [--n 10]
```

## Readiness commands

### doctor

Full readiness report: subjects, modalities, events, label coverage, split feasibility.

```bash
qortex doctor DATASET_ID [--snapshot VER] [--local] [--data-dir DIR]
              [--recipe NAME] [--json]
```

### minimum

Compute the smallest download for a given goal.

```bash
qortex minimum DATASET_ID --goal {first-batch,label-check,validation,metadata}
               [--download] [--data-dir DIR]
```

### can-train

Binary label readiness check.

```bash
qortex can-train DATASET_ID --label COL [--min-classes N] [--min-per-class N]
                 [--min-subjects N]
```

### first-batch

Download minimum subjects and run a full pipeline pass.

```bash
qortex first-batch DATASET_ID --label COL [--data-dir DIR] [--format FMT]
```

### leakage-check

Verify no subject appears in two splits of a converted artifact.

```bash
qortex leakage-check DATASET_ID --artifact DIR [--level {subject,session}]
```

### content-status

Check local files for LFS pointers and incomplete downloads.

```bash
qortex content-status DATASET_ID --data-dir DIR
```

### make-recipe

Create a readiness recipe file from parameters.

```bash
qortex make-recipe RECIPE_NAME [--modality MOD] [--label COL]
                   [--min-subjects N] [--output FILE]
```

### run-recipe

Run a recipe file against a dataset.

```bash
qortex run-recipe RECIPE_FILE --dataset DATASET_ID [--data-dir DIR]
```

## Download commands

### plan

Show what would be downloaded without downloading.

```bash
qortex plan DATASET_ID [--subjects S [S ...]] [--tasks T [T ...]]
            [--suffixes SUF [SUF ...]] [--output FILE]
```

### download

Download files from OpenNeuro.

```bash
qortex download DATASET_ID [--subjects S [S ...]] [--tasks T [T ...]]
                [--suffixes SUF [SUF ...]] [--datatypes D [D ...]]
                [--metadata-only] [--min-goal GOAL]
                [--data-dir DIR] [--snapshot VER]
                [--max-size GB] [--concurrency N] [--force]
```

### validate

Run BIDS Validator on a local directory.

```bash
qortex validate DIR [--dataset-id ID]
```

### local-index

Scan a local BIDS directory and build a local manifest.

```bash
qortex local-index DIR --dataset-id ID
```

## Conversion commands

### eda

Exploratory data analysis: signal statistics and label landscape.

```bash
qortex eda DATASET_ID [--data-dir DIR] [--label COL]
```

### convert

Convert a local BIDS dataset to an ML artifact.

```bash
qortex convert DATASET_ID [--data-dir DIR] --output DIR
               [--format {parquet,zarr,hdf5,webdataset,huggingface,tfrecord}]
               [--window SECONDS] [--overlap FRAC]
               [--event-aligned] [--tmin SECONDS]
               [--label COL] [--subjects S [S ...]]
               [--val-frac F] [--test-frac F] [--seed N]
               [--workers N] [--overwrite]
```

## Cache commands

### cache info

Show total cache size.

```bash
qortex cache info
```

### cache list

List cached datasets.

```bash
qortex cache list [DATASET_ID]
```

### cache remove

Remove a dataset from cache.

```bash
qortex cache remove DATASET_ID [--snapshot VER]
```

### cache clear

Remove all cached data.

```bash
qortex cache clear [--yes]
```

## Authentication

### login

Store an OpenNeuro API token for private dataset access.

```bash
qortex login [--token TOKEN]
```

## Catalog commands

### catalog-refresh

Pull the latest catalog index from OpenNeuro.

```bash
qortex catalog-refresh
```

### catalog-profile

Show catalog statistics.

```bash
qortex catalog-profile
```

## Visualization commands

### visualize

Render figures for local files.

```bash
qortex visualize DATASET_ID [--subject S] [--suffix SUF]
                 [--data-dir DIR] [--output DIR]
```

### visualize-openneuro

Render center-slice thumbnails from OpenNeuro CDN without full download.

```bash
qortex visualize-openneuro DATASET_ID [--subjects S [S ...]]
                            [--suffixes SUF [SUF ...]]
                            [--snapshot VER] [--output DIR]
                            [--format {html,png}]
```

### dicom-browser

Browse a local DICOM directory.

```bash
qortex dicom-browser DIR [--output FILE]
```

### fmri-qc

Render the fMRI QC summary for a BOLD file.

```bash
qortex fmri-qc FILE [--events FILE] [--confounds FILE] [--output FILE]
```

### dwi-qc

Render the DWI QC summary.

```bash
qortex dwi-qc FILE --bval FILE --bvec FILE [--output FILE]
```

### visual-audit

Run a visual audit on a local BIDS directory or manifest.

```bash
qortex visual-audit DATASET_ID [--data-dir DIR] [--output FILE]
                    [--mode {local,manifest}]
```

### visualize-overlay

Overlay a mask or stat map on an anatomical image.

```bash
qortex visualize-overlay BACKGROUND_FILE --overlay FILE
                          [--type {mask,labelmap,stat,pet,contour,edges}]
                          [--alpha F] [--threshold F] [--output FILE]
```

### compare-masks

Side-by-side or overlay comparison of two masks.

```bash
qortex compare-masks BACKGROUND_FILE --mask-a FILE --mask-b FILE
                     [--labels A B] [--mode {side-by-side,overlay}]
                     [--output FILE]
```

### artifact-visualize

Visualize samples from a converted artifact.

```bash
qortex artifact-visualize ARTIFACT_DIR [--split {train,val,test}]
                           [--index N] [--compare-splits]
```

### dashboard

Launch the Qortex Streamlit dashboard.

```bash
qortex dashboard [--host HOST] [--port PORT]
```

---

## Preflight command

### preflight

Run a goal-aware preflight check before data processing.

```bash
qortex preflight DATASET_DIR --goal {visualize,convert,train,neuroai-run}
                 [--modality MOD] [--target COL] [--split-unit {subject,session,run}]
                 [--pipeline YAML] [--output FILE] [--strict]
```

---

## Check commands

Goal-specific data integrity checks. Each outputs a structured report and exits 1 on BLOCK.

### check events

Validate event timing, onset ranges, and BIDS entity matching.

```bash
qortex check events DATASET_DIR [--modality MOD] [--require-trial-type]
                    [--output FILE]
```

### check units

Validate declared units against observed signal scale.

```bash
qortex check units DATASET_DIR [--modality MOD] [--output FILE]
```

### check geometry

Validate NIfTI affine, qform/sform, and DWI gradient tables.

```bash
qortex check geometry DATASET_DIR [--output FILE]
```

### check leakage

Validate ML label availability and train/test leakage risk.

```bash
qortex check leakage DATASET_DIR [--target COL]
                     [--split-unit {subject,session,run}] [--output FILE]
```

### check structure

Validate BIDS layout, companion-file closure, and entity consistency.

```bash
qortex check structure DATASET_DIR [--modality MOD] [--output FILE]
```

### check metadata

Cross-check BIDS JSON sidecars against raw file headers.

```bash
qortex check metadata DATASET_DIR [--modality MOD] [--output FILE]
```

### check dwi-gradients

Validate DWI bval/bvec integrity and volume count consistency.

```bash
qortex check dwi-gradients DATASET_DIR [--output FILE]
```

### check eeg-channels

Validate EEG channel metadata and unit consistency.

```bash
qortex check eeg-channels DATASET_DIR [--output FILE]
```

---

## NeuroAI commands

### neuroai check

Run compatibility check between a source and a model.

```bash
qortex neuroai check SOURCE_PATH --model MODEL_ID [--provider {huggingface,braindecode,ultralytics,onnx}]
                  [--task TASK]
```

### neuroai run

Run a NeuroAI inference pipeline from a YAML spec.

```bash
qortex neuroai run PIPELINE_YAML [--data-dir DIR] [--output DIR]
                   [--dry-run] [--verbose]
```

### neuroai benchmark

Benchmark a model against a source or BIDS dataset.

```bash
qortex neuroai benchmark SOURCE_PATH --model MODEL_ID [--n-runs N]
                          [--provider PROV] [--output FILE]
```

### neuroai replay

Replay a previously saved inference result.

```bash
qortex neuroai replay RESULT_PATH [--output DIR]
```

### neuroai inspect-source

Probe a source file and print its SourceProfile.

```bash
qortex neuroai inspect-source SOURCE_PATH [--type {auto,edf,nifti,bids,dicom,lsl,brainflow,nwb,xdf}]
```

### neuroai inspect-model

Inspect a model and print its ModelProfile and InputContract.

```bash
qortex neuroai inspect-model MODEL_ID [--provider {huggingface,braindecode,ultralytics,onnx}]
                              [--task TASK]
```

### neuroai suggest-models

Suggest compatible models for a given source file and task.

```bash
qortex neuroai suggest-models SOURCE_PATH --task {classification,segmentation,detection,regression,embedding}
                               [--modality {eeg,mri,ct,image,video}]
                               [--top-k N] [--provider PROV]
```

---

## Neuro-classic commands

Classical computational neuroscience methods. Requires `pip install 'qortex[neuroclassic]'`.

### neuro-classic signal-qc

Run signal QC on all EEG/MEG files (flatline, NaN, saturation, amplitude).

```bash
qortex neuro-classic signal-qc DATASET_DIR [--modality {eeg,meg}]
                                [--max-files N] [--output FILE]
```

### neuro-classic image-qc

Run image QC on all NIfTI files (NaN, constant, tSNR, volume outliers).

```bash
qortex neuro-classic image-qc DATASET_DIR [--modality {mri,fmri,ct,dwi,pet}]
                               [--max-files N] [--output FILE]
```

### neuro-classic eeg-psd

Run EEG power spectral density and spectral slope summary.

```bash
qortex neuro-classic eeg-psd DATASET_DIR [--max-files N] [--output FILE]
```

### neuro-classic connectivity

Compute connectivity matrix and graph metrics on EEG files.

```bash
qortex neuro-classic connectivity DATASET_DIR [--method correlation]
                                  [--threshold F] [--output FILE]
```

### neuro-classic cohort-anomalies

Detect cohort-level amplitude outliers across subjects.

```bash
qortex neuro-classic cohort-anomalies DATASET_DIR [--output FILE]
```
