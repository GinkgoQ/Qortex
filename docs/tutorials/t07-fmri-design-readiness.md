# T07 — fMRI Event and Design Readiness

**Dataset:** ds000001 — Balloon Analog Risk-taking Task (BART), 16 subjects  
**Task:** Event/design validation; no ML model required  
**First model:** Design matrix inspection + optional simple GLM diagnostic  
**Difficulty:** Beginner

---

## Prerequisites

```bash
pip install 'qortex[tutorials]'
```

**Download:** No registration required.  
Download from https://legacy.openfmri.org/dataset/ds000001/ (BIDS format, ~1.4 GB).  
Extract to `/data/ds000001/`.

---

## Step 1 — Load and inspect

```python
from pathlib import Path
from qortex.datasets import ds000001

card = ds000001.describe()
print(card)
print(card.access_instructions)

bundle = ds000001.load_data(
    local_root=Path("/data/ds000001"),
    subjects=["01", "02", "03", "04", "05"],
)
bundle.info()
# FMRIBundle: ds000001 — Balloon Analog Risk-taking Task
#   Task       : balloonanalogrisktask
#   Subjects   : 5
#   TR         : 2.0 s
#   BOLD files : 5
#   Event files: 5
```

---

## Step 2 — Validation gates

```python
from pathlib import Path

missing_bold = [p for p in bundle.bold_paths if not p.exists()]
missing_evts = [p for p in bundle.event_paths if not p.exists()]

print(f"Missing BOLD files  : {len(missing_bold)}")
print(f"Missing event files : {len(missing_evts)}")
assert not missing_bold, f"BOLD files not found: {missing_bold}"
assert not missing_evts, f"Event files not found: {missing_evts}"
print("✓ All BOLD and event files present")
```

---

## Step 3 — Parse and audit event files

```python
events = bundle.load_events()

for sub, (evt, path) in zip(bundle.subjects, zip(events, bundle.event_paths)):
    rows = evt.get("rows", [])
    n = evt.get("n_events", 0)
    print(f"sub-{sub}: {n} events in {Path(path).name}")
    if rows:
        # Show first 3 events
        for row in rows[:3]:
            print(f"  onset={row.get('onset','?')} dur={row.get('duration','?')} "
                  f"trial_type={row.get('trial_type','?')}")
```

**Expected events in ds000001 (BART):**
- `trial_type` values: `inflate`, `cashout`, `explode`, `pump`
- Check that onsets are within scan duration

---

## Step 4 — NIfTI header inspection

```python
try:
    import nibabel as nib
except ImportError:
    print("Install nibabel: pip install 'qortex[mri]'")
    raise

for sub, bold_path in zip(bundle.subjects, bundle.bold_paths):
    if not bold_path.exists():
        continue
    img = nib.load(str(bold_path))
    shape = img.shape
    pixdim = img.header.get_zooms()
    print(f"sub-{sub}: shape={shape}, voxel={pixdim[:3]}, TR={pixdim[3]:.2f}s")
    assert abs(float(pixdim[3]) - bundle.tr) < 0.1, \
        f"TR mismatch: header says {pixdim[3]:.2f}, expected {bundle.tr}"
print("✓ TR consistent with header for all subjects")
```

---

## Step 5 — Event timing audit

Check that all event onsets fall within the scan duration.

```python
for sub, (evt, bold_path) in zip(bundle.subjects, zip(events, bundle.bold_paths)):
    if not bold_path.exists():
        continue
    img = nib.load(str(bold_path))
    n_vols = img.shape[3] if len(img.shape) == 4 else 1
    scan_duration = n_vols * bundle.tr

    rows = evt.get("rows", [])
    for row in rows:
        onset = float(row.get("onset", 0))
        duration = float(row.get("duration", 0))
        if onset + duration > scan_duration:
            print(f"⚠ sub-{sub}: event at {onset:.1f}s + {duration:.1f}s "
                  f"exceeds scan duration {scan_duration:.1f}s")

    condition_counts = {}
    for row in rows:
        t = row.get("trial_type", "unknown")
        condition_counts[t] = condition_counts.get(t, 0) + 1
    print(f"sub-{sub}: {condition_counts}")
```

---

## Step 6 — Readiness report

```python
print("\n=== fMRI Design Readiness Report ===")
print(f"Dataset          : {bundle.card.full_name}")
print(f"Task             : {bundle.task}")
print(f"TR               : {bundle.tr} s")
print(f"Subjects checked : {len(bundle.subjects)}")
print(f"BOLD files       : {sum(p.exists() for p in bundle.bold_paths)} / {len(bundle.bold_paths)} present")
print(f"Event files      : {sum(p.exists() for p in bundle.event_paths)} / {len(bundle.event_paths)} present")
print(f"License          : {bundle.card.license}")
print()
print("⚠ Revision 2.0.4 corrected event timing files.")
print("  Always use the corrected revision — check dataset_description.json.")
```

---

## Step 7 — Optional: simple design matrix preview

```python
import numpy as np

# Minimal HRF convolution check (no model fitting)
try:
    from nilearn.glm.first_level import make_first_level_design_matrix  # type: ignore
    import pandas as pd

    sub = bundle.subjects[0]
    evt_data = bundle.events[0]
    rows = evt_data.get("rows", [])

    img = nib.load(str(bundle.bold_paths[0]))
    n_vols = img.shape[3]
    frame_times = np.arange(n_vols) * bundle.tr

    events_df = pd.DataFrame({
        "onset":      [float(r["onset"]) for r in rows],
        "duration":   [float(r["duration"]) for r in rows],
        "trial_type": [r.get("trial_type", "unknown") for r in rows],
    })

    dm = make_first_level_design_matrix(frame_times, events_df, hrf_model="spm")
    print(f"Design matrix shape: {dm.shape}")
    print(f"Conditions: {list(dm.columns)}")
except ImportError:
    print("nilearn not installed — skip design matrix preview")
```

---

## Validation summary

| Gate | Check |
|---|---|
| BIDS manifest | `ds000001` directory with sub-XX layout |
| BOLD file exists | Step 2 — asserted for all subjects |
| Events file exists | Step 2 — asserted for all subjects |
| Corrected revision | Check `dataset_description.json` BIDSVersion |
| Event onsets within scan | Step 5 — onset + duration ≤ scan_duration |
| TR / volume metadata | Step 4 — header TR matches bundle.tr |
