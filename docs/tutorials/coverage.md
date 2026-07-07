# Tutorial and Project Coverage

This page maps Qortex's implemented feature surface to the tutorials and
scenario projects that exercise it.  Tutorials teach the user-facing workflow;
scenario projects are runnable verification scripts under `test/`.

Run all scenario projects:

```bash
python test/run_all.py
```

Run one scenario project:

```bash
python test/project_21_neuroai_runtime/run.py
```

## Coverage Matrix

| Feature area | Tutorials | Scenario projects | What is covered |
|---|---|---|---|
| Installation, config, public imports, structured errors | [Tutorial index](index.md) | `test/0_import_config`, `test/project_01_import_and_config` | Package import, settings overrides, structured exceptions, warning behavior |
| OpenNeuro manifest and BIDS graph | [T07 fMRI readiness](t07-fmri-design-readiness.md) | `test/1_manifest_models`, `test/project_02_manifest_fetch`, `test/project_03_manifest_graph` | Manifest fetch, BIDS entity parsing, logical recordings, companion closure |
| Catalog ingestion and search | [Tutorial index](index.md) | `test/11_catalog_project`, `test/project_14_catalog`, `test/16_catalog_ingestion_project` | Catalog refresh, facets, live search, deep file summaries, semantic search surfaces |
| Remote metadata preview | [T07 fMRI readiness](t07-fmri-design-readiness.md) | `test/3_remote_preview_project`, `test/project_04_metadata_preview`, `test/project_15_remote_inspection`, `test/17_remote_inspection_project` | Participants tables, events TSV previews, sidecars, NIfTI header probes without full download |
| Download planning, metadata-only download, cache, local index | [T07 fMRI readiness](t07-fmri-design-readiness.md) | `test/2_selection_planning`, `test/4_download_specific_parts_project`, `test/project_05_selection_plan`, `test/project_06_metadata_download`, `test/project_07_local_index`, `test/10_local_index_validation_cache_project` | Companion-aware plans, exact-path closure, metadata-only materialization, local reconciliation |
| Readiness-first decisions | [T07 fMRI readiness](t07-fmri-design-readiness.md) | `test/7_readiness_report_project`, `test/project_08_readiness_report`, `test/project_12_decision_workflow`, `test/15_decision_workflows_project` | `doctor`, `minimum`, `can-train`, first-batch planning, local label confirmation |
| Event labels, label landscape, signal budget | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T03 sleep staging](t03-eeg-sleep-staging.md), [T04 seizure detection](t04-eeg-seizure-detection.md), [T07 fMRI readiness](t07-fmri-design-readiness.md) | `test/5_eda_events`, `test/8_behavior_loader_project`, `test/project_16_label_landscape`, `test/18_label_landscape_project`, `test/19_signal_budget_project` | Events loading, label candidates, imbalance, ISI jitter, class consistency, window-budget estimates |
| Windowing, splits, leakage safety | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T02 EEG connectivity](t02-eeg-connectivity.md), [T03 sleep staging](t03-eeg-sleep-staging.md), [T04 seizure detection](t04-eeg-seizure-detection.md), [T05 dementia baseline](t05-mri-dementia-baseline.md), [T06 age and sex QC](t06-mri-age-sex-qc.md) | `test/9_window_split_project`, `test/project_11_artifact_access` | Subject-safe split assignment, split summaries, artifact split metadata |
| Conversion formats and artifact writing | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T03 sleep staging](t03-eeg-sleep-staging.md), [T08 tumour segmentation](t08-brain-tumour-segmentation.md) | `test/6_conversion_artifact`, `test/project_10_events_to_parquet`, `tests/test_convert_format_writers.py` | Parquet artifacts, event-table export, HDF5/WebDataset/HuggingFace writer roundtrips |
| Artifact inspection and ML bridges | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T08 tumour segmentation](t08-brain-tumour-segmentation.md) | `test/project_11_artifact_access`, `test/project_21_neuroai_runtime` | Artifact manifests, source tracking, output validation, prediction/marker counts |
| Visualization and visual QC | [T06 age and sex QC](t06-mri-age-sex-qc.md), [T07 fMRI readiness](t07-fmri-design-readiness.md), [T08 tumour segmentation](t08-brain-tumour-segmentation.md) | `test/project_19_visualization`, `test/project_20_visualization_advanced` | Volume viewers, EEG plots, fMRI/DWI QC, overlays, masks, DICOM PHI redaction, surface summaries |
| Remote NIfTI streaming | [T07 fMRI readiness](t07-fmri-design-readiness.md), [Visual pipeline](../visualization/visual-pipeline.md) | `tests/test_stream_nifti.py`, `tests/test_console_streaming.py` | Header probing, range-backed slices, 4D timepoint validation, console streaming endpoints |
| Dataset loaders and dataset cards | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T02 EEG connectivity](t02-eeg-connectivity.md), [T03 sleep staging](t03-eeg-sleep-staging.md), [T04 seizure detection](t04-eeg-seizure-detection.md), [T05 dementia baseline](t05-mri-dementia-baseline.md), [T06 age and sex QC](t06-mri-age-sex-qc.md), [T07 fMRI readiness](t07-fmri-design-readiness.md), [T08 tumour segmentation](t08-brain-tumour-segmentation.md) | `test/13_dataset_facade_project`, `test/project_15_remote_inspection` | Dataset facade, dataset cards, local and remote inspection helpers |
| Neuroclassic methods | [T01 EEG motor imagery](t01-eeg-motor-imagery.md), [T02 EEG connectivity](t02-eeg-connectivity.md), [T03 sleep staging](t03-eeg-sleep-staging.md), [T05 dementia baseline](t05-mri-dementia-baseline.md), [T06 age and sex QC](t06-mri-age-sex-qc.md) | `tests/test_neuroclassic.py`, `tests/test_neuroclassic_advanced.py`, `tests/test_cli_neuro_classic.py` | Signal QC, PSD, named epoch feature matrices, workflow-specific EEG band sets, Pearson and PLV connectivity, graph metrics, spectral entropy, Higuchi fractal dimension, CSP spatial filters, leakage-safe splits, CLI behavior |
| NeuroAI runtime | [T08 tumour segmentation](t08-brain-tumour-segmentation.md), [NeuroAI pipeline](../neuroai/pipeline.md) | `test/project_21_neuroai_runtime`, `tests/test_neuroai_pipeline.py`, `tests/test_neuroai_transforms.py` | Contract checks, transform planning, strict failure policies, batching, outputs, triggers, artifacts, benchmark, replay |
| Console and dashboard entrypoints | [Tutorial index](index.md) | `tests/test_console_streaming.py`, `tests/test_console_error_mapping.py`, `test/project_13_cli` | Import-safe Streamlit dashboard, console API error mapping, streaming endpoint behavior, CLI smoke |
| Search engine and dataset selector | [Tutorial index](index.md) | `test/project_18_dataset_selector`, `test/20_dataset_selector_project`, `tests/test_search_engine.py` | Goal-aware dataset ranking, hard-fail explanations, evidence-partitioned search behavior |
| Documentation coverage itself | This page | `test/project_22_tutorial_coverage` | Ensures tutorial pages, linked scenario projects, and coverage rows stay connected |

## Domain Tutorial Coverage

| Tutorial | Domain | Core Qortex surfaces |
|---|---|---|
| [T01 EEG motor imagery](t01-eeg-motor-imagery.md) | EEG classification | Dataset loader, signal QC, windows, subject-safe splits, classical ML, artifact contract |
| [T02 EEG connectivity](t02-eeg-connectivity.md) | EEG graph features | Dataset loader, alpha-band windows, Pearson connectivity, graph metrics, condition comparison |
| [T03 sleep staging](t03-eeg-sleep-staging.md) | PSG classification | Sleep-EDF loader, hypnogram mapping, epoch validation, class imbalance, subject-held-out splits |
| [T04 seizure detection](t04-eeg-seizure-detection.md) | EEG event detection | CHB-MIT loader, seizure interval parsing, overlap labels, file-safe splits, false-positive rate |
| [T05 dementia baseline](t05-mri-dementia-baseline.md) | MRI research baseline | OASIS loader, clinical table join, confound report, leakage-safe subject split, research-only framing |
| [T06 age and sex QC](t06-mri-age-sex-qc.md) | MRI QC and demographics | IXI loader, demographic join, scanner/site QC, image metadata, regression/classification baselines |
| [T07 fMRI readiness](t07-fmri-design-readiness.md) | BIDS/fMRI validation | Manifest, BOLD/events pairing, timing audit, design readiness, no forced classifier target |
| [T08 tumour segmentation](t08-brain-tumour-segmentation.md) | Medical segmentation | Image-mask checks, label inventory, MONAI pipeline, Dice metrics, overlays, research-only framing |

## Project Coverage Rules

Every new public feature should land with at least one of these:

- A tutorial section when users need workflow guidance.
- A guide/API page when users need reference detail.
- A scenario project when the behavior is cross-module or end-to-end.
- A focused pytest when the behavior is local, contract-heavy, or regression-prone.

The coverage scenario checks this page for broken tutorial links and missing
project references.  It does not replace semantic review; it keeps the map
honest enough that missing tutorial/project coverage is visible during routine
verification.

## Runnable Scenario Inventory

Every directory below contains a `run.py` and is picked up by
`python test/run_all.py`.

- `test/0_import_config`
- `test/1_manifest_models`
- `test/2_selection_planning`
- `test/3_remote_preview_project`
- `test/4_download_specific_parts_project`
- `test/5_eda_events`
- `test/6_conversion_artifact`
- `test/7_readiness_report_project`
- `test/8_behavior_loader_project`
- `test/9_window_split_project`
- `test/10_local_index_validation_cache_project`
- `test/11_catalog_project`
- `test/12_cli_project`
- `test/13_dataset_facade_project`
- `test/14_live_openneuro_metadata_project`
- `test/15_decision_workflows_project`
- `test/16_catalog_ingestion_project`
- `test/17_remote_inspection_project`
- `test/18_label_landscape_project`
- `test/19_signal_budget_project`
- `test/20_dataset_selector_project`
- `test/project_01_import_and_config`
- `test/project_02_manifest_fetch`
- `test/project_03_manifest_graph`
- `test/project_04_metadata_preview`
- `test/project_05_selection_plan`
- `test/project_06_metadata_download`
- `test/project_07_local_index`
- `test/project_08_readiness_report`
- `test/project_09_eda_report`
- `test/project_10_events_to_parquet`
- `test/project_11_artifact_access`
- `test/project_12_decision_workflow`
- `test/project_13_cli`
- `test/project_14_catalog`
- `test/project_15_remote_inspection`
- `test/project_16_label_landscape`
- `test/project_17_signal_budget`
- `test/project_18_dataset_selector`
- `test/project_19_visualization`
- `test/project_20_visualization_advanced`
- `test/project_21_neuroai_runtime`
- `test/project_22_tutorial_coverage`
