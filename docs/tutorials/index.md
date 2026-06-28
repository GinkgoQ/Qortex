# Tutorials

Practical, end-to-end workflows using real neuroscience datasets.
Each tutorial is self-contained and reproducible.  Install prerequisites once:

```bash
pip install 'qortex[tutorials]'
```

For tutorials that require MONAI (T08):

```bash
pip install 'qortex[tutorials,monai]'
```

---

## Loading datasets

Qortex provides Keras-style dataset loaders for the most widely used open
neuroscience datasets.  Every loader follows the same three-step pattern:

```python
from qortex.datasets import eegbci        # import the dataset module
card = eegbci.describe()                  # inspect — no download
bundle = eegbci.load_data(subjects=[1])   # download + load (cached)
```

| Function | What it does |
|---|---|
| `describe()` | Returns a `DatasetCard` with metadata, license, and access instructions |
| `load_data(**kwargs)` | Downloads on first call, returns a typed Bundle |
| `bundle.info()` | Prints a summary of the loaded data |
| `bundle.run_qc()` | Runs signal / image QC via `qortex.neuroclassic` |

**Bundle types by modality**

| Modality | Bundle type | Key method |
|---|---|---|
| EEG / PSG | `EEGBundle` | `.to_windows(window_s, bandpass)` → `(X, y)` |
| Structural MRI | `MRIBundle` | `.load_images()` → list of arrays |
| Task fMRI | `FMRIBundle` | `.load_events()` → event tables |
| Segmentation | `SegmentationBundle` | `.load_pair(i)` → `(image, mask)` |

**Catalogue API**

```python
import qortex.datasets as qd

qd.list_available()           # list all DatasetCards
qd.describe("eegbci")         # single DatasetCard
print(qd.summary())           # compact table
qd.load_dataset("sleep_edf", subjects=[0, 1, 2])
```

---

## Tutorial roadmap

| ID | Title | Dataset | Task | Difficulty |
|---|---|---|---|---|
| [T01](t01-eeg-motor-imagery.md) | EEG motor imagery classification | EEGBCI | Classification | Beginner |
| [T02](t02-eeg-connectivity.md) | EEG connectivity and graph features | EEGBCI | Connectivity | Beginner |
| [T03](t03-eeg-sleep-staging.md) | EEG sleep-stage classification | Sleep-EDF | Classification | Intermediate |
| [T04](t04-eeg-seizure-detection.md) | EEG seizure event detection | CHB-MIT | Binary classification | Intermediate |
| [T05](t05-mri-dementia-baseline.md) | MRI dementia research baseline | OASIS-1 | Classification + confounds | Intermediate |
| [T06](t06-mri-age-sex-qc.md) | MRI age regression / sex QC | IXI | Regression + classification | Intermediate |
| [T07](t07-fmri-design-readiness.md) | fMRI event and design readiness | ds000001 | Validation only | Beginner |
| [T08](t08-brain-tumour-segmentation.md) | Brain tumour segmentation baseline | MSD Brain | Segmentation | Advanced |

---

## Validation gates

Every tutorial enforces dataset-specific validation before any model training.
These gates prevent common mistakes — leakage, annotation mismatch, class
imbalance — and produce a machine-readable artifact contract.

**Shared gates (all tutorials)**

- File existence and loadability check
- Sampling rate / voxel size verification
- Subject-level leakage-safe split via `qortex.neuroclassic.assign_leakage_safe_splits`

**Per-tutorial gates** — see individual tutorial pages.

---

## Data licensing summary

| Dataset | License | Registration |
|---|---|---|
| EEGBCI | ODbL v1.0 (PhysioNet) | None |
| Sleep-EDF | ODbL v1.0 (PhysioNet) | None |
| CHB-MIT | ODbL v1.0 (PhysioNet) | None — research only |
| OASIS-1 | CC BY-NC-SA 3.0 | Required (sites.wustl.edu) |
| IXI | CC BY-SA 3.0 | None |
| ds000001 | PDDL 1.0 | None |
| MSD Brain | CC BY-SA 4.0 | None |

Always review the original license before redistributing derived datasets or
trained models.
