# DWI / Diffusion

Diffusion-weighted MRI (DWI) measures water diffusivity along gradient directions. Each DWI file is a 4D NIfTI with one volume per gradient direction, plus two required companion files: `.bval` (b-values per volume) and `.bvec` (gradient unit vectors per volume).

## BIDS files

| File | Required | Description |
|------|----------|-------------|
| `*_dwi.nii.gz` | yes | 4D DWI volume |
| `*_dwi.bval` | yes | b-values per volume (whitespace-separated) |
| `*_dwi.bvec` | yes | Gradient vectors, 3 × n_volumes matrix |
| `*_dwi.json` | yes | Scanner metadata sidecar |

Qortex downloads all three companions automatically when you download a DWI NIfTI.

## Check before download

```python
from qortex import Dataset

ds = Dataset("ds000001")
report = ds.doctor()
# NO_COMPANIONS error appears if bval/bvec are missing for any DWI file

# Inspect companion coverage
manifest = ds.manifest()
dwi_files = ds.files(datatypes=["dwi"], suffixes=["dwi"])
for f in dwi_files:
    print(f.subject, f.path)
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/dwi/sub-01_dwi.nii.gz")

info["shape"]                   # [96, 96, 60, 65]
info["n_directions"]            # 65
info["voxel_size_mm"]           # [2.0, 2.0, 2.0]
info["b_values"]                # [0, 1000, 2000]  (unique shells, rounded)
info["n_b0_volumes"]            # 7  (b < 50 s/mm²)
info["bval_available"]          # True
info["bvec_available"]          # True
info["magnetic_field_strength"] # 3.0
info["echo_time"]               # 0.085
info["total_readout_time"]      # 0.0684
```

## Visualize

```python
from qortex.visualize import dwi_summary

fig = dwi_summary(
    "data/ds000001/sub-01/dwi/sub-01_dwi.nii.gz",
    bval="data/ds000001/sub-01/dwi/sub-01_dwi.bval",
    bvec="data/ds000001/sub-01/dwi/sub-01_dwi.bvec",
)
fig.show()
```

The 4-panel summary includes: b0 reference slice, highest-b-value slice, b-value histogram, and gradient direction sphere.

CLI:

```bash
qortex dwi-qc sub-01/dwi/sub-01_dwi.nii.gz \
    --bval sub-01/dwi/sub-01_dwi.bval \
    --bvec sub-01/dwi/sub-01_dwi.bvec \
    --output sub01_dwi_qc.html
```

## Load into Python

```python
from qortex.parse.dwi import DWILoader

loader = DWILoader()
record = loader.load(file_record, local_path=Path("sub-01/dwi/sub-01_dwi.nii.gz"))

record.img          # NiBabel image (4D: x, y, z, directions)
record.shape        # (96, 96, 60, 65)
record.metadata["b_values"]       # list of floats per volume
record.metadata["bvecs"]          # (3, n_dirs) numpy array
record.metadata["n_shells"]       # number of unique b-value shells
record.metadata["b0_indices"]     # indices of b0 volumes

data = loader.to_numpy(record)    # float32 array, shape (96, 96, 60, 65)
```

## Conversion

DWI typically converts to whole-volume samples. One sample per subject per acquisition.

```python
art = ds.convert(
    data_dir="data/ds000001/",
    output_dir="artifacts/dwi/",
    format="zarr",
    datatypes=["dwi"],
    split=dict(strategy="subject"),
)
```

Each artifact sample is a `(x, y, z, n_directions)` float32 array plus b-value metadata.

## Readiness considerations

- **bval and bvec are required.** Conversion fails without them. Qortex includes them automatically in every download plan.
- **Shell count matters.** Multi-shell acquisitions (b=0, 1000, 2000) enable more diffusion models than single-shell.
- **Minimum directions.** Most tractography algorithms need ≥ 30 unique gradient directions. Warn if fewer.
- **No events.** DWI does not use events.tsv. Labels for ML typically come from participants.tsv (diagnosis, group).

## Related

- [DWI QC](../../visualization/dwi-qc.md) — QC viewer reference
- [Readiness recipes](../../readiness/recipes.md) — `dwi` recipe
