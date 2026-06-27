# Fieldmaps

Fieldmaps estimate the magnetic field inhomogeneity in a scanner and enable geometric distortion correction (SDC) for fMRI and DWI acquisitions. BIDS supports several fieldmap types, each with different file layouts.

## BIDS fieldmap types

| Type | Files | Description |
|------|-------|-------------|
| Phase-difference | `magnitude1.nii.gz`, `magnitude2.nii.gz`, `phasediff.nii.gz` + JSON | Two echo magnitudes + phase difference |
| Two phase images | `magnitude1.nii.gz`, `magnitude2.nii.gz`, `phase1.nii.gz`, `phase2.nii.gz` | Separate phase images per echo |
| Direct fieldmap | `fieldmap.nii.gz` + JSON | Hz/rad/s map already computed |
| EPI fieldmap | `epi.nii.gz` | Blip-up/blip-down EPI pair |

All types require a JSON sidecar. The JSON must include `IntendedFor`, pointing to the functional run(s) the fieldmap is meant to correct.

## Check fieldmap availability

```python
from qortex import Dataset

ds = Dataset("ds004130")
fmap_files = ds.files(datatypes=["fmap"])
for f in fmap_files:
    print(f.suffix, f.path, f.size)
```

## Inspect a fieldmap sidecar

```python
sidecar = ds.sidecar("sub-01/fmap/sub-01_phasediff.json")

sidecar["EchoTime1"]    # 0.00492
sidecar["EchoTime2"]    # 0.00738
sidecar["IntendedFor"]  # ["func/sub-01_task-rest_bold.nii.gz"]
sidecar["Units"]        # "rad/s"
```

## Distortion correction readiness

Qortex checks whether functional runs have a matching fieldmap via the `IntendedFor` field:

```python
report = ds.doctor()
# finding: WARNING [SDC_MISSING] — sub-03 has BOLD but no matched fieldmap
```

## Download fieldmaps with functional data

Fieldmaps are included automatically when you download BOLD files:

```python
ds.download(
    subjects=["01", "02"],
    datatypes=["func", "fmap"],
    data_dir="data/ds004130/",
)
```

## Limitations

Qortex does not perform fieldmap-based distortion correction itself. Fieldmap files are read, sidecar metadata is inspected, and `IntendedFor` relationships are resolved. Actual SDC requires fMRIPrep or a dedicated preprocessing tool.

## Related

- [Selective download](../../download/selective-download.md) — include fieldmaps in a plan
- [Metadata](../../dataset/metadata.md) — read sidecars without downloading volumes
