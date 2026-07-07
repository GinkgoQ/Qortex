# Visualize Samples

`art.visualize_sample()` renders a single sample from an artifact as a plot.

## Python

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")

# By index
art.visualize_sample(split="train", index=0)

# By subject and label
art.visualize_sample(split="train", subject="01", label="rest")
```

## What is rendered

The plot type depends on the data shape in the artifact:

- **EEG/timeseries** (`n_channels, n_timepoints`): Butterfly plot with all channels overlaid. Channel axis is y-axis-shifted for readability.
- **fMRI** (`n_voxels,`): Bar chart of the top 50 most variable features, sorted by absolute value.
- **fMRI 4D** (`nx, ny, nz, nt`): Mean axial slice image.
- **Tabular** (`n_features,`): Horizontal bar chart of feature values.

## Batch visualization

To inspect multiple samples:

```python
for i in range(5):
    art.visualize_sample(split="train", index=i)
```

Or as a grid (if plotly is available):

```python
art.visualize_sample(split="train", indices=range(9), layout="3x3")
```

## Saving

```python
result = art.visualize_sample(split="train", index=0, show=False)
result.to_png("sample_0.png")
result.to_html("sample_0.html")
```




## Related

- [Compare splits](compare-splits.md) — distribution-level comparison
- [Artifact visualization](../visualization/artifact-visualization.md) — visual audit of the artifact
