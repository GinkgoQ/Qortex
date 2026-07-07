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
qortex inspect DATASET_ID [--snapshot VER] [--output FILE]
```

### metadata

Print dataset metadata: dataset_description.json, participants.tsv, or a specific sidecar.

```bash
qortex metadata DATASET_ID [--snapshot VER] [--output-dir DIR] [--download] [--limit N]
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
qortex preview DATASET_ID FILE_PATH [--snapshot VER] [--local-path DIR]
                [--rows N] [--max-bytes N]
```

## Readiness commands

### doctor

Full readiness report: subjects, modalities, events, label coverage, split feasibility.

```bash
qortex doctor DATASET_ID [--snapshot VER] [--local-path DIR] [--json-output FILE]
```

### minimum

Compute the smallest download for a given goal.

```bash
qortex minimum DATASET_ID [--snapshot VER]
               [--goal {first-batch,label-check,validation,metadata}]
               [--modality MOD] [--target COL]
               [--output-dir DIR] [--download] [--json-output FILE]
```

### can-train

Structured supervised-training readiness report.

```bash
qortex can-train DATASET_ID [--snapshot VER] [--modality MOD]
                 [--target COL] [--local-path DIR] [--json-output FILE]
```

### first-batch

Print first artifact rows, or print the smallest plan needed to produce a first batch.

```bash
qortex first-batch [--dataset DATASET_ID] [--snapshot VER]
                   [--artifact DIR] [--local-path DIR]
                   [--modality MOD] [--target COL] [--limit N]
                   [--json-output FILE]
```

### leakage-check

Verify no subject appears in two splits of a converted artifact.

```bash
qortex leakage-check DATASET_ID --artifact DIR [--level {subject,session}]
```

### content-status

Check local files for LFS pointers and incomplete downloads.

```bash
qortex content-status PATH [--dataset DATASET_ID] [--snapshot VER] [--json-output FILE]
```

### make-recipe

Create a readiness recipe file from parameters.

```bash
qortex make-recipe DATASET_ID OUTPUT [--snapshot VER] [--modality MOD]
                   [--target COL] [--split subject] [--goal first-batch]
                   [--output-dir DIR] [--metadata-only]
```

### run-recipe

Run a recipe file against a dataset.

```bash
qortex run-recipe RECIPE_FILE [--download]
```

## Download commands

### plan

Show what would be downloaded without downloading.

```bash
qortex plan DATASET_ID [--snapshot VER] [--subjects CSV] [--tasks CSV]
            [--modalities CSV] [--include-derivatives] [--output-dir DIR]
```

### download

Download files from OpenNeuro.

```bash
qortex download DATASET_ID [--subjects S [S ...]] [--tasks T [T ...]]
                [--modalities CSV] [--include-derivatives]
                [--output-dir DIR] [--snapshot VER] [--dry-run]
```

### validate

Run BIDS Validator on a local directory.

```bash
qortex validate DIR [--config FILE] [--json-output FILE]
                [--markdown-output FILE] [--html-output FILE]
                [--ignore-warnings] [--ignore-nifti-headers]
                [--no-cache] [--refresh-cache] [--timeout SECONDS]
```

### local-index

Scan a local BIDS directory and build a local manifest.

```bash
qortex local-index DIR [--manifest-dir DIR] [--json-output FILE] [--no-pybids]
```

## Conversion commands

### eda

Exploratory data analysis: signal statistics and label landscape.

```bash
qortex eda DATASET_ID [--snapshot VER] [--output FILE]
```

### convert

Convert a local BIDS dataset to an ML artifact.

```bash
qortex convert DATA_DIR OUTPUT_DIR
               [--format {parquet,zarr,hdf5,webdataset,huggingface,tfrecord}]
               [--window SECONDS] [--overlap FRAC]
               [--split STRATEGY] [--shard-size N]
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
qortex visualize PATH [--mode {auto,qc,static,interactive,thumbnail,summary}]
                 [--output FILE] [--colormap NAME] [--modality MOD] [--open]
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
qortex visual-audit DATASET_ID [--local DIR] [--output-dir DIR]
                    [--json] [--markdown] [--manifest-json]
                    [--subjects CSV] [--suffixes CSV] [--datatypes CSV]
                    [--max-files N] [--n-per-suffix N] [--open]
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

Launch the Qortex Streamlit dashboard. This command requires the `dashboard`
extra; importing `qortex.console.app` does not require Streamlit until the app
is launched.

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

Run compatibility check for a YAML pipeline without loading model weights.

```bash
qortex neuroai check PIPELINE_YAML [--verbose] [--json] [--markdown]
```

### neuroai plan

Run compatibility checking and print the executable preprocessing plan.

```bash
qortex neuroai plan PIPELINE_YAML [--json]
```

### neuroai run

Run a NeuroAI inference pipeline from a YAML spec.

```bash
qortex neuroai run PIPELINE_YAML [--dry-run]
                   [--artifact-dir DIR]
                   [--validate-artifact/--no-validate-artifact]
```

When `--artifact-dir` is provided, file-backed outputs are written under
`DIR/outputs/`, sidecars are written to `DIR`, and the artifact is validated by
default after the run.

### neuroai validate-artifact

Validate a completed NeuroAI run artifact.

```bash
qortex neuroai validate-artifact ARTIFACT_DIR [--strict] [--json] [--markdown]
```

### neuroai render-segmentation-showcase

Render inspectable source/mask/overlay artifacts from a source NIfTI and a predicted mask.

```bash
qortex neuroai render-segmentation-showcase IMAGE_NIFTI PREDICTION_MASK_NIFTI OUT_DIR \
  --case-id sub-04 \
  --model-id my-segmentation-model \
  --class-labels-json '{"0":"background","1":"target"}'
```

Writes source slice, prediction mask, overlay, area profile, metrics JSON, and manifest JSON. Add `--truth-mask TRUTH_NIFTI` to include Dice/IoU and an error map.

### neuroai run-external-segmentation

Run a supported external segmentation CLI and capture command provenance.

```bash
qortex neuroai run-external-segmentation totalsegmentator case_001_ct.nii.gz artifacts/case_001_total.nii.gz \
  --task total \
  --device gpu \
  --extra-arg=--ml
```

For nnU-Net:

```bash
qortex neuroai run-external-segmentation nnunet nnunet_input/case_001_0000.nii.gz nnunet_predictions \
  --model-folder /models/nnUNet_results \
  --dataset-id 501 \
  --configuration 3d_fullres \
  --trainer nnUNetTrainer \
  --plans nnUNetPlans \
  --fold 0 --fold 1 --fold 2 --fold 3 --fold 4
```

The command writes a `*.qortex.json` provenance file beside the output file, or `qortex_external_segmentation.json` inside an output directory.

### neuroai benchmark

Benchmark a YAML pipeline without writing real outputs.

```bash
qortex neuroai benchmark PIPELINE_YAML [--windows N]
```

### neuroai replay

Replay a recorded source file through a YAML pipeline.

```bash
qortex neuroai replay PIPELINE_YAML SOURCE_PATH [--speed FLOAT] [--output-dir DIR]
```

### neuroai inspect-source

Probe a source file and print its SourceProfile.

```bash
qortex neuroai inspect-source SOURCE_PATH [--modality MODALITY] [--suffix SUFFIX]
```

### neuroai inspect-model

Inspect a model and print its ModelProfile and InputContract.

```bash
qortex neuroai inspect-model MODEL_ID [--provider PROVIDER]
                              [--task TASK]
                              [--input-contract input_contract.yaml]
                              [--output-contract output_contract.yaml]
```

`--input-contract` and `--output-contract` accept JSON or YAML mappings. They
are required for raw PyTorch checkpoints and useful for any model whose config
does not expose reliable neuro/medical input and output semantics.

### neuroai suggest-models

Suggest compatible models for a given source file and task.

```bash
qortex neuroai suggest-models SOURCE_PATH --task {classification,segmentation,detection,regression,embedding}
                               [--modality {eeg,mri,ct,image,video}]
                               [--top-k N] [--provider PROV] [--json]
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
                                  [--threshold F] [--max-files N]
                                  [--output FILE]
```

### neuro-classic cohort-anomalies

Detect cohort-level amplitude outliers across subjects.

```bash
qortex neuro-classic cohort-anomalies DATASET_DIR [--output FILE]
```
