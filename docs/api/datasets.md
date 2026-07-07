# Datasets API

`qortex.datasets` provides Keras-style loaders for open neuroscience datasets. Every module exposes two functions and no more: `describe()` (no download) and `load_data()` (downloads on first call, cached afterwards).

```python
from qortex.datasets import eegbci

card = eegbci.describe()         # DatasetCard — metadata only, no download
bundle = eegbci.load_data(       # EEGBundle — downloads and caches via MNE
    subjects=[1, 2, 3],
    runs=[4, 8, 12],
)
X, y = bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
features = bundle.to_feature_matrix()
print(features.shape, features.feature_names[:5])

# For local BIDS EEG datasets, delegate sidecar-aware reads to MNE-BIDS.
bids_report = bundle.read_bids_raws(root="./data/my-bids-dataset", preload=False)
print(bids_report.n_files, bids_report.to_dict()["bids_paths"][:1])
```

---

## Catalogue functions

::: qortex.datasets.list_available
    options:
      show_source: false

::: qortex.datasets.describe
    options:
      show_source: false

::: qortex.datasets.load_dataset
    options:
      show_source: false

::: qortex.datasets.summary
    options:
      show_source: false

---

## DatasetCard

::: qortex.datasets.DatasetCard
    options:
      show_source: false
      members:
        - __str__
        - to_dict

---

## EEGBundle

Returned by EEG dataset loaders (`eegbci`, `sleep_edf`, `chbmit`). In
addition to Qortex windowing, QC, and feature extraction, `EEGBundle` can read
local BIDS EEG recordings through MNE-BIDS when `qortex[eeg]` is installed.
That path preserves BIDS entities, `events.tsv` annotations, `channels.tsv`
bad-channel metadata, and inherited sidecars in the upstream MNE `Raw` object.

::: qortex.datasets._base.EEGBundle
    options:
      show_source: false
      members:
        - to_windows
        - to_feature_matrix
        - read_bids_raws
        - run_qc
        - info
        - n_channels
        - n_files

## BIDSRawReadReport

Returned by `EEGBundle.read_bids_raws()`.

::: qortex.datasets.BIDSRawReadReport
    options:
      show_source: false
      members:
        - n_files
        - to_dict

---

## MRIBundle

Returned by structural MRI loaders (`oasis1`, `ixi`).

::: qortex.datasets._base.MRIBundle
    options:
      show_source: false
      members:
        - load_images
        - run_qc
        - info
        - n_subjects

---

## FMRIBundle

Returned by task fMRI loaders (`ds000001`).

::: qortex.datasets._base.FMRIBundle
    options:
      show_source: false
      members:
        - load_events
        - run_preflight
        - info

---

## SegmentationBundle

Returned by segmentation loaders (`msd_brain`).

::: qortex.datasets._base.SegmentationBundle
    options:
      show_source: false
      members:
        - load_pair
        - info
        - n_cases

---

## DatasetRegistry

The global registry that backs `list_available()` and `describe()`. Advanced users can access it directly via `qortex.datasets._REGISTRY`.

::: qortex.datasets.DatasetRegistry
    options:
      show_source: false
      members:
        - register
        - get
        - list_all
        - list_by_modality
        - list_by_tutorial
        - summary_table

---

## Available datasets

| Name | `load_data()` returns | Tutorials |
|---|---|---|
| `eegbci` | `EEGBundle` | T01, T02 |
| `sleep_edf` | `EEGBundle` | T03 |
| `chbmit` | `EEGBundle` | T04 |
| `oasis1` | `MRIBundle` | T05 |
| `ixi` | `MRIBundle` | T06 |
| `ds000001` | `FMRIBundle` | T07 |
| `msd_brain` | `SegmentationBundle` | T08 |

---

### `qortex.datasets.eegbci`

PhysioNet EEG Motor Movement/Imagery. 109 subjects, 160 Hz, 64-channel BCI2000.

::: qortex.datasets.eegbci.describe
    options:
      show_source: false

::: qortex.datasets.eegbci.load_data
    options:
      show_source: false

**Label map constants**

| Constant | Runs | Classes |
|---|---|---|
| `LABEL_MAP_FIST_IMAGERY` | 4, 8, 12 | rest / left_fist_imagery / right_fist_imagery |
| `LABEL_MAP_FEET_IMAGERY` | 6, 10, 14 | rest / both_fists_imagery / both_feet_imagery |
| `LABEL_MAP_FIST_EXECUTION` | 3, 7, 11 | rest / left_fist / right_fist |
| `LABEL_MAP_FEET_EXECUTION` | 5, 9, 13 | rest / both_fists / both_feet |
| `LABEL_MAP_BASELINE` | 1, 2 | eyes_open / eyes_closed |

---

### `qortex.datasets.sleep_edf`

Sleep-EDF Expanded — 5-class AASM sleep staging. Cassette + temazepam subsets.

::: qortex.datasets.sleep_edf.describe
    options:
      show_source: false

::: qortex.datasets.sleep_edf.load_data
    options:
      show_source: false

**Label map** — `LABEL_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}`

---

### `qortex.datasets.chbmit`

CHB-MIT Scalp EEG Seizure Database. 23 cases, binary seizure/non-seizure labels.

::: qortex.datasets.chbmit.describe
    options:
      show_source: false

::: qortex.datasets.chbmit.load_data
    options:
      show_source: false

::: qortex.datasets.chbmit.SeizureInterval
    options:
      show_source: false
      members:
        - overlaps_window

::: qortex.datasets.chbmit.parse_seizure_summary
    options:
      show_source: false

::: qortex.datasets.chbmit.label_windows_for_file
    options:
      show_source: false

**Label map** — `LABEL_MAP = {0: "non_seizure", 1: "seizure"}`

---

### `qortex.datasets.oasis1`

OASIS-1 cross-sectional structural MRI. 416 subjects, dementia classification.

::: qortex.datasets.oasis1.describe
    options:
      show_source: false

::: qortex.datasets.oasis1.load_data
    options:
      show_source: false

::: qortex.datasets.oasis1.load_clinical_table
    options:
      show_source: false

**Label map** — `LABEL_MAP = {0: "no_dementia", 1: "dementia"}`

---

### `qortex.datasets.ixi`

IXI multimodal MRI (T1/T2/PD/MRA/DWI). 600 subjects, age regression / sex classification.

::: qortex.datasets.ixi.describe
    options:
      show_source: false

::: qortex.datasets.ixi.load_data
    options:
      show_source: false

::: qortex.datasets.ixi.load_demographics
    options:
      show_source: false

**Label map** — `LABEL_MAP_SEX = {0: "male", 1: "female"}`

---

### `qortex.datasets.ds000001`

OpenfMRI balloon analogue risk task (BART). 16 subjects, BIDS fMRI, event design.

::: qortex.datasets.ds000001.describe
    options:
      show_source: false

::: qortex.datasets.ds000001.load_data
    options:
      show_source: false

---

### `qortex.datasets.msd_brain`

MSD Brain Tumour segmentation. 4-class glioma segmentation (FLAIR, T1w, T1gd, T2w).

::: qortex.datasets.msd_brain.describe
    options:
      show_source: false

::: qortex.datasets.msd_brain.load_data
    options:
      show_source: false

**Label map** — `LABEL_MAP = {0: "background", 1: "NCR_NET", 2: "edema", 3: "enhancing_tumour"}`

**Modalities** — `MODALITIES = ["FLAIR", "T1w", "T1gd", "T2w"]`

See the [Tutorials](../tutorials/index.md) for end-to-end usage of each dataset.
