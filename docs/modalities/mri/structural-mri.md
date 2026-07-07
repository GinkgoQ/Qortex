# Structural MRI

Structural MRI — primarily T1w and T2w — provides the anatomical reference for most multi-modal neuroimaging studies. Qortex can inspect, load, visualize, and use structural MRI for overlay rendering or as a covariate in ML pipelines.

## BIDS files

| Suffix | Description |
|--------|-------------|
| `T1w` | T1-weighted (e.g. MPRAGE) — most common anatomical reference |
| `T2w` | T2-weighted — grey/white contrast complementary to T1w |
| `FLAIR` | Fluid-attenuated inversion recovery |
| `PD` | Proton density |
| `T1map`, `T2map` | Quantitative relaxometry maps |
| `mask` | Brain masks (in `anat/`) |
| `dseg` | Discrete segmentation maps |
| `probseg` | Probabilistic tissue maps |

File extension: `.nii.gz` or `.nii`. JSON sidecar carries scanner metadata.

## Inspect before download

```python
from qortex import Dataset

ds = Dataset("ds004130")
manifest = ds.manifest()

# All T1w files across all subjects
t1w_files = ds.files(datatypes=["anat"], suffixes=["T1w"])
for f in t1w_files:
    print(f.subject, f.size, f.path)
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/anat/sub-01_T1w.nii.gz")

info["shape"]                    # [256, 256, 176]
info["voxel_size_mm"]            # [1.0, 1.0, 1.0]
info["orientation"]              # "RAS"
info["magnetic_field_strength"]  # 3.0
info["manufacturer"]             # "Siemens"
info["echo_time"]                # 0.00296
info["repetition_time"]          # 2.3
```

All values come from the NIfTI header and JSON sidecar — the NIfTI data array is not loaded.

## Visualize

```python
from qortex.visualize import VolumeViewer

viewer = VolumeViewer("data/ds004130/sub-01/anat/sub-01_T1w.nii.gz")
viewer.ortho()           # three-plane orthogonal view
viewer.lightbox()        # grid of axial slices
viewer.interactive_html("sub01_T1w.html")
```

CLI:

```bash
qortex visualize data/ds004130/sub-01/anat/sub-01_T1w.nii.gz --mode interactive
```

## Load into Python

```python
from qortex.parse.mri import MRILoader
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
files = ds.files(suffixes=["T1w"])

loader = MRILoader()
record = loader.load(files[0], local_path=...)

record.img         # NiBabel NIfTI1Image — data not yet loaded
record.shape       # (256, 256, 176)
record.voxel_size  # (1.0, 1.0, 1.0)
record.affine      # 4×4 numpy array (RAS)

data = loader.to_numpy(record)  # numpy float32 array, shape (256, 256, 176)
```

`lazy_load()` returns the same `ImageRecord` but defers data loading until the array is accessed:

```python
record = loader.lazy_load(files[0], local_path=...)
# no data in memory yet
data = loader.to_numpy(record)  # reads on first access
```

## Conversion

Structural MRI converts to 3D array slices. Each subject contributes one sample per T1w file.

```python
art = ds.convert(
    data_dir="data/ds004130/",
    output_dir="artifacts/",
    format="zarr",
    suffixes=["T1w"],
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
)
```

Each sample is a `(256, 256, 176)` float32 array in the artifact. There are no windows or events for structural MRI — one file equals one sample.

## Readiness considerations

- **No events required.** Structural MRI does not need events.tsv for conversion.
- **Label source.** Labels typically come from `participants.tsv` (age, sex, diagnosis, group). These are covariate labels, not trial-type labels.
- **Size.** A T1w NIfTI at 1 mm isotropic is typically 30–80 MB compressed. Plan storage accordingly.
- **Orientation.** Qortex reorients all volumes to RAS canonical by default via nibabel. Pass `canonical=False` to skip.




## Related

- [fMRI / BOLD](fmri-bold.md) — functional volumes using structural as reference
- [Overlays](../../visualization/overlays.md) — overlay masks or stat maps on T1w
- [Visual audit](../../visualization/visual-audit.md) — T1w coverage matrix
