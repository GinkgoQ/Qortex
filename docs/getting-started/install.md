# Install

## Requirements

- Python 3.10 or later
- pip 23 or later

## Base install

The base package contains the core Dataset facade, all CLI commands, the download engine, and the readiness checks. It does not include optional modality-specific visualization or ML-framework integrations.

```bash
pip install qortex
```

## Optional extras

Qortex uses optional extras to avoid pulling in large dependencies unless you need them.

| Extra | What it adds | Command |
|-------|-------------|---------|
| `mri` | nibabel, nilearn — NIfTI inspection and volume visualization | `pip install "qortex[mri]"` |
| `dwi` | nibabel, dipy — DWI bval/bvec handling and gradient sphere plots | `pip install "qortex[dwi]"` |
| `eeg` | mne — EEG/MEG reading, time-frequency, topomap | `pip install "qortex[eeg]"` |
| `visual` | nibabel, plotly — interactive HTML viewers | `pip install "qortex[visual]"` |
| `visual-all` | All of the above | `pip install "qortex[visual-all]"` |
| `dicom` | pydicom — DICOM browser and metadata extraction | `pip install "qortex[dicom]"` |
| `zarr` | zarr — Zarr output format | `pip install "qortex[zarr]"` |
| `hdf5` | h5py — HDF5 output format | `pip install "qortex[hdf5]"` |
| `torch` | torch — PyTorch Dataset and DataLoader integration | `pip install "qortex[torch]"` |
| `lightning` | lightning — LightningDataModule integration | `pip install "qortex[lightning]"` |
| `tf` | tensorflow — TFRecord output format | `pip install "qortex[tf]"` |
| `hf` | datasets — HuggingFace Dataset output format | `pip install "qortex[hf]"` |
| `sklearn` | scikit-learn — sklearn arrays helper | `pip install "qortex[sklearn]"` |
| `dashboard` | panel — Local dashboard server | `pip install "qortex[dashboard]"` |
| `validation` | bids-validator — BIDS structural validation | `pip install "qortex[validation]"` |
| `all` | Everything | `pip install "qortex[all]"` |

Combine extras with commas:

```bash
pip install "qortex[mri,torch,zarr]"
```

## Verify install

```bash
qortex --version
# qortex 0.x.y
```

Check that an optional extra loaded correctly:

```python
from qortex.visualize import VolumeViewer  # requires [mri] or [visual]
```

If an extra is missing, Qortex raises an `ImportError` with a message that tells you exactly which extra to install.

## Development install

Clone the repo and install in editable mode:

```bash
git clone https://github.com/GinkgoQ/qortex.git
cd qortex
pip install -e ".[all]"
```

## Known install issues

**Conflict with existing nibabel.** If your environment already has nibabel installed from another package (e.g., nilearn), check that the version is compatible:

```bash
python -c "import nibabel; print(nibabel.__version__)"
```

Qortex requires nibabel ≥ 5.0.

**Apple Silicon (MRI extras).** Nibabel and nilearn install cleanly on ARM. No special flags required.

**Windows.** All extras work on Windows. The `validation` extra (bids-validator) requires Node.js on PATH if using the full JS validator. The Python-only validator shipped with newer bids-validator packages does not need Node.
