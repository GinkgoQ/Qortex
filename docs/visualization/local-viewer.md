# Local Viewer

`visualize()` renders figures for all supported files in a local path. It walks the directory, classifies each file by intent, and produces one figure per asset.

## Python

```python
from qortex.visualize import visualize

# Visualize all files under a subject directory
results = visualize("data/ds004130/sub-01/")

for r in results:
    print(r.asset.path, r.intent)
    r.show()
```

Or visualize a single file:

```python
from qortex.visualize import visualize

result = visualize("data/ds004130/sub-01/anat/sub-01_T1w.nii.gz")
result.show()
```

## From the Dataset API

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
results = ds.visualize(subject="01")
for r in results:
    r.to_png(f"figures/{r.asset.path.stem}.png")
```

## CLI

```bash
# Visualize one subject's data and open in browser
qortex visualize ds004130 --subject 01 --data-dir data/ds004130/

# Save all figures to a directory
qortex visualize ds004130 --subject 01 --data-dir data/ds004130/ --output figures/
```

## What each intent renders

| Intent | Figure type |
|--------|------------|
| Anatomical | 3-panel ortho (axial, coronal, sagittal) at center slices |
| BOLD | Mean EPI + temporal std + tSNR + global signal plot |
| DWI | b=0 volume + high-b volume + gradient sphere |
| PET | Ortho panels with PET colormap |
| CT | Ortho panels with brain window preset |
| Fieldmap | Ortho panels with diverging colormap |
| Mask | Overlay on center slice of the nearest anatomical |
| Labelmap | Overlay with discrete colormap |
| Stat map | Overlay with RdBu_r colormap, thresholded |
| Raw signal | Butterfly plot (EEG/MEG) |
| Unknown | Single-slice thumbnail, no colormap processing |

## Surface files (GIFTI/CIFTI)

Surface files are detected by extension (`.surf.gii`, `.func.gii`, `.dscalar.nii`, `.dtseries.nii`). Rendering falls through to a summary-only output — the intent is `INTENT_SURFACE` and the figure shows metadata only (number of vertices, data range, sampling density).

Full surface rendering requires a surface viewer not currently implemented. This is a known limitation.

## Saving output

```python
for r in results:
    r.to_html(f"figures/{r.asset.path.stem}.html")
    r.to_png(f"figures/{r.asset.path.stem}.png")
```

Interactive HTML (Plotly) is produced for BOLD and EEG; static PNG (matplotlib) for anatomical and DWI.

## Related

- [Visual pipeline](visual-pipeline.md) — how intent classification works
- [fMRI QC](fmri-qc.md) — more detailed BOLD QC panels
- [Visual audit](visual-audit.md) — coverage view across all subjects
