# PET Metadata

PET images are quantitative. Raw scanner values are in units of radioactivity (Bq/mL). To interpret them for biology or for ML, you need the metadata from the JSON sidecar.

## Required sidecar fields (BIDS PET)

| Field | Description |
|-------|-------------|
| `TracerName` | Radiotracer identifier (e.g. `18F-FDG`) |
| `TracerRadionuclide` | Radionuclide (e.g. `F18`) |
| `InjectedRadioactivity` | Activity at time of injection (MBq) |
| `InjectedRadioactivityUnits` | Usually `"MBq"` |
| `InjectionStart` | Time of injection relative to scan start (s) |
| `FrameTimesStart` | Start time of each frame (s) — list |
| `FrameDuration` | Duration of each frame (s) — list or scalar |
| `BodyWeight` | Subject body weight (kg) — required for SUV |
| `ReconMethodName` | Reconstruction algorithm |
| `DecayCorrectionFactor` | Applied decay correction |

## Read PET metadata before download

```python
from qortex import Dataset

ds = Dataset("ds001421")
sidecar = ds.sidecar("sub-01/pet/sub-01_trc-18FFDG_pet.json")

print(sidecar["TracerName"])              # "18F-FDG"
print(sidecar["InjectedRadioactivity"])   # 185.0
print(sidecar["BodyWeight"])              # 72.4
print(sidecar["FrameTimesStart"])         # [0, 15, 30, 60, 90, ...]
print(sidecar["FrameDuration"])           # [15, 15, 30, 30, 60, ...]
```

## Inspect after download

```python
info = ds.nifti_info("sub-01/pet/sub-01_trc-18FFDG_pet.nii.gz")

info["shape"]                    # [128, 128, 63, 23]
info["n_frames"]                 # 23
info["voxel_size_mm"]            # [2.09, 2.09, 2.43]
info["total_scan_time_s"]        # 3600.0
info["tracer_name"]              # "18F-FDG"
info["tracer_radionuclide"]      # "F18"
info["body_weight_kg"]           # 72.4
info["injected_activity_bq"]     # 185000000.0
info["suv_normalization_possible"]  # True — both weight and activity available
info["reconstruction_method"]    # "3D-OSEM"
```

## SUV normalization

Standardized Uptake Value (SUV) normalizes raw PET values by injected activity and body weight:

```
SUV = (PET_value [Bq/mL] × body_weight [kg]) / (injected_activity [Bq])
```

Qortex checks whether SUV normalization is possible by verifying both `BodyWeight` and `InjectedRadioactivity` are present in the sidecar. When loading PET data for ML, apply SUV normalization before training unless your model explicitly handles raw radioactivity values.

## Blood data

Some PET datasets include blood time-activity curves in `*_blood.tsv`. These are not loaded by default. Access via:

```python
blood_files = ds.files(datatypes=["pet"], suffixes=["blood"])
```

## Related

- [PET visualization](visualization.md)
- [Metadata](../../dataset/metadata.md) — read sidecars from CDN without downloading volumes
