# Modality Selection

Choosing the right modality for a machine learning project depends on the question being asked, the available hardware, and what data exists on OpenNeuro. This page helps you match a research question to a modality and then find datasets using Qortex.

## Quick guide

| Question | Best modality | Why |
|----------|-------------|-----|
| Brain region activation from task | fMRI BOLD | Best spatial resolution for cortical mapping |
| Fast neural dynamics (< 100 ms) | EEG / MEG | Millisecond temporal resolution |
| Cortical decoding in clinical patients | iEEG | Direct contact with cortex; high SNR |
| Whole-brain connectivity at rest | fMRI (resting-state) | Spatial coverage, established pipeline |
| White matter tract integrity | DWI | Only modality directly sensitive to axon microstructure |
| Haemodynamic response in low-cost settings | fNIRS | Portable, wearable; no MRI required |
| Metabolic / receptor mapping | PET | Tracer-specific quantification |
| Behavior and task performance only | Behavioral / events | No imaging required |

## Modality characteristics

| Modality | Temporal res. | Spatial res. | File size (typical) | Online processing |
|----------|--------------|-------------|---------------------|-------------------|
| Structural MRI | static | 1 mm | 30–80 MB | no |
| fMRI BOLD | 0.5–2 s (TR) | 2–3 mm | 100 MB – 2 GB | no |
| DWI | static | 2–3 mm | 200 MB – 2 GB | no |
| MEG | < 1 ms | 5–10 mm (reconstructed) | 500 MB – 5 GB | possible |
| EEG | < 1 ms | low (scalp) | 5–500 MB | yes |
| iEEG | < 1 ms | < 1 mm (contact) | 50 MB – 5 GB | yes |
| fNIRS | 0.1–1 s | 1–3 cm | 1–50 MB | yes |
| PET | frames (15 s – 30 min) | 2–5 mm | 20–200 MB | no |

## Search by modality

```python
from qortex.catalog import DatasetQuery
from qortex.client import OpenNeuroClient

# Check what's available per modality
with OpenNeuroClient() as client:
    facets = DatasetQuery().facets()
    print(facets["modalities"])
    # {"MRI": 1243, "EEG": 156, "MEG": 89, "PET": 52, ...}

# Find EEG datasets with task events, ≥ 50 subjects
results = (
    DatasetQuery()
    .modality("eeg")
    .min_subjects(50)
    .has_events()
    .fetch()
)
```

## Switching modality mid-project

Qortex's pipeline is modality-agnostic. The same `Dataset → inspect → download → convert` flow works for every modality. The output artifact format (Parquet, Zarr, etc.) is the same regardless of modality — only the per-sample shape changes.

```python
# EEG: sample shape is (n_channels, n_time_points)
# fMRI: sample shape is (n_voxels, n_time_points) or (x, y, z, t_window)
# DWI: sample shape is (x, y, z, n_directions)
# PET: sample shape is (x, y, z) per frame
```

## Related

- [Modality readiness](readiness.md) — per-modality readiness checks
- [Modality conversion](conversion.md) — per-modality conversion considerations
- [Search & filter](../dataset/search-catalog.md) — DatasetQuery reference
