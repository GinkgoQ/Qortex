# Electrodes and Coordinates

Electrode localization is what separates iEEG from scalp EEG for spatial analysis. The `electrodes.tsv` file maps each electrode label to a 3D coordinate. The `coordsystem.json` defines the coordinate system and optional anatomical reference image.

## electrodes.tsv

```
name   x        y       z       size   group       hemisphere
LA1    -36.2    -5.8    -18.3   1.5    amygdala    L
LA2    -40.1    -7.3    -17.1   1.5    amygdala    L
LB1    -52.4    -14.0   5.7     1.5    hippocampus L
```

Required columns: `name`, `x`, `y`, `z`.
Optional but common: `size` (mm), `group` (anatomical label), `hemisphere`.

## coordsystem.json

```json
{
  "iEEGCoordinateSystem": "ACPC",
  "iEEGCoordinateUnits": "mm",
  "iEEGCoordinateSystemDescription": "AC-PC aligned",
  "iEEGCoordinateProcessingDescription": "Electrode contacts localized using post-implant CT co-registered to pre-op MRI",
  "IntendedFor": "anat/sub-01_T1w.nii.gz"
}
```

Common coordinate systems: `ACPC`, `MNI152Lin`, `Talairach`, `CapTrak`, `PIXEL`.

## Read before download

```python
from qortex import Dataset

ds = Dataset("ds003688")

# Check that electrodes.tsv exists
manifest = ds.manifest()
elec_files = [f for f in manifest.files if "electrodes.tsv" in f.path]
print(f"{len(elec_files)} electrode files found")
```

## Read electrode positions

```python
import polars as pl

electrodes = pl.read_csv(
    "data/ds003688/sub-01/ieeg/sub-01_electrodes.tsv",
    separator="\t",
)

print(electrodes.head())
print(electrodes["group"].value_counts())  # anatomical distribution
```

Or via Qortex metadata:

```python
elec_df = ds.metadata_files(
    subject="01",
    filename="sub-01_electrodes.tsv",
)
```

## Read coordinate system

```python
coord = ds.sidecar("sub-01/ieeg/sub-01_coordsystem.json")
print(coord["iEEGCoordinateSystem"])    # "ACPC"
print(coord["iEEGCoordinateUnits"])     # "mm"
print(coord.get("IntendedFor"))         # anatomical reference path
```

## Use electrode positions in ML

Electrode coordinates can serve as spatial features or for defining channel neighborhoods:

```python
import polars as pl
import numpy as np

electrodes = pl.read_csv("sub-01_electrodes.tsv", separator="\t")
coords = electrodes.select(["x", "y", "z"]).to_numpy()
names  = electrodes["name"].to_list()

# Distance matrix for graph-based methods
from scipy.spatial.distance import cdist
dist = cdist(coords, coords)
```

## Readiness: missing electrodes

Some iEEG datasets have signal files but no `electrodes.tsv`. Qortex detects this:

```python
report = ds.doctor()
# WARNING [NO_ELECTRODES]: sub-05 has ieeg/ but no electrodes.tsv
```

Signal data can still be downloaded and converted, but spatial analysis is not possible without electrode positions.

## Related

- [iEEG files](files.md)
- [Metadata](../../dataset/metadata.md) — read sidecar files from CDN
