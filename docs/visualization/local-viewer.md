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
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
results = ds.visualize(subject="01")
for r in results:
    r.to_png(f"figures/{r.asset.path.stem}.png")
```

## CLI

```bash
# Visualize one subject's data and open in browser
qortex visualize data/ds004130/sub-01 --open

# Save all figures to a directory
qortex visualize data/ds004130/sub-01/anat/sub-01_T1w.nii.gz --output figures/sub-01_T1w.html
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

Surface files are detected by extension (`.surf.gii`, `.func.gii`, `.dscalar.nii`, `.dtseries.nii`). The intent is `INTENT_SURFACE`. GIFTI files are inspected as mesh/scalar/label assets and rendered as surface QC summaries with vertices, faces, bounds, labels, array roles, and value ranges. CIFTI files are inspected as dense matrices and rendered as axis-aware matrix summaries with sampled data ranges.

This is a QC viewer, not a replacement for Connectome Workbench. Advanced interactions such as full multi-view workbench-style surface scenes and volume-to-surface projection remain future work.

## Saving output

```python
for r in results:
    r.to_html(f"figures/{r.asset.path.stem}.html")
    r.to_png(f"figures/{r.asset.path.stem}.png")
```

Interactive HTML (Plotly) is produced for BOLD and EEG; static PNG (matplotlib) for anatomical and DWI.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-bold-axial.png" alt="Axial BOLD slice from OpenNeuro ds000001 subject 01 run 01.">
  <figcaption>Real BOLD axial slice streamed with `Dataset.stream_slice()` without downloading the full NIfTI file.</figcaption>
</figure>

```python
sl = ds.stream_slice(subject='01', modality='bold', run='01', time_index=0, axis=2)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->

## Related

- [Visual pipeline](visual-pipeline.md) — how intent classification works
- [fMRI QC](fmri-qc.md) — more detailed BOLD QC panels
- [Visual audit](visual-audit.md) — coverage view across all subjects
