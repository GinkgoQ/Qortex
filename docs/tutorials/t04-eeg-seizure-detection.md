# T04 — EEG Seizure Event Detection

**Dataset:** CHB-MIT Scalp EEG Seizure Database (23 cases, 42.6 GB)  
**Task:** Binary window classification — seizure vs non-seizure  
**First model:** Sliding-window features + Random Forest / XGBoost  
**Later model:** 1D CNN / temporal CNN  
**Difficulty:** Intermediate  
**Research only — not for clinical use.**

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn]'
# Optionally for gradient boosting:
pip install xgboost
```

**Download:** CHB-MIT must be downloaded manually from PhysioNet.
The full dataset is ~42.6 GB.  Start with `chb01` only (~2.3 GB).

```
https://physionet.org/content/chbmit/1.0.0/
```

Download `chb01/` and place it at `/data/chbmit/chb01/`.

---

## Step 1 — Load data

```python
from pathlib import Path
from qortex.datasets import chbmit

card = chbmit.describe()
print(card.access_instructions)

bundle = chbmit.load_data(
    cases=["chb01"],
    seizure_files_only=True,    # only EDF files with seizures
    local_root=Path("/data/chbmit"),
    preload=True,
)
bundle.info()
# EEGBundle: CHB-MIT Scalp EEG Seizure Database
#   Files    : <n seizure files in chb01>
#   Channels : 23
#   Sfreq    : 256.0 Hz
```

---

## Step 2 — Validation gates

```python
assert bundle.sfreq == 256.0, f"Unexpected sfreq: {bundle.sfreq}"
assert bundle.n_files > 0, "No EDF files found in chb01"

seizure_map = bundle.metadata.get("seizure_map", {})
total_seizures = sum(len(v) for v in seizure_map.values())
print(f"✓ {bundle.n_files} seizure-containing files, {total_seizures} seizure intervals")

# Confirm seizure intervals are non-zero duration
for fname, intervals in seizure_map.items():
    for iv in intervals:
        assert iv.duration_sec > 0, f"Zero-duration seizure in {fname}"
```

---

## Step 3 — Label windows

```python
import numpy as np
from qortex.datasets.chbmit import label_windows_for_file

window_s = 5.0
step_s   = 1.0            # aggressive overlap to capture transitions

all_windows: list[np.ndarray] = []
all_labels: list[int] = []

for i, (raw, path) in enumerate(zip(bundle.raws, bundle.local_paths)):
    data = raw.get_data().astype(np.float32)   # [23, n_times]
    n_samples = data.shape[1]
    fname = path.name

    windows, labels = label_windows_for_file(
        fname,
        n_samples=n_samples,
        sfreq=bundle.sfreq,
        window_s=window_s,
        step_s=step_s,
        seizure_map=seizure_map,
    )
    for (start, end), label in zip(windows, labels):
        all_windows.append(data[:, start:end])
        all_labels.append(label)

X = np.stack(all_windows, axis=0)   # [n_windows, 23, 1280]
y = np.array(all_labels, dtype=np.int64)

n_seizure = int(y.sum())
n_total   = len(y)
print(f"Total windows   : {n_total}")
print(f"Seizure windows : {n_seizure} ({100*n_seizure/n_total:.2f}%)")
print(f"Class ratio     : 1 : {(n_total-n_seizure)//max(n_seizure,1)}")
```

**Expect severe imbalance** (~1 : 50 or worse). Handle it explicitly.

---

## Step 4 — Feature extraction

```python
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
    "hfo":   (80, 120),   # high-frequency oscillations
}

def extract_seizure_features(X, sfreq=256.0):
    features = []
    for epoch in X:
        row = []
        for ch in epoch:
            f, psd = welch(ch, sfreq, nperseg=min(512, len(ch)))
            for lo, hi in BANDS.values():
                mask = (f >= lo) & (f <= hi)
                row.append(float(psd[mask].mean()) if mask.any() else 0.0)
            row.extend([float(ch.mean()), float(ch.std()), float(np.abs(ch).max())])
        features.append(row)
    return np.array(features, dtype=np.float32)

X_feats = extract_seizure_features(X, sfreq=bundle.sfreq)
print(f"Feature matrix: {X_feats.shape}")
```

---

## Step 5 — File-level split (no same-file across train/test)

```python
from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints

# Build rows with file_id group
file_ids = []
for i, path in enumerate(bundle.local_paths):
    n = len(all_windows) // len(bundle.local_paths)  # approx equal windows per file
    file_ids.extend([path.stem] * n)
file_ids = file_ids[:len(y)]

rows = [{"id": str(i), "file": fid} for i, fid in enumerate(file_ids)]
result = assign_leakage_safe_splits(
    rows, id_column="id",
    constraints=SplitConstraints(group_columns=["file"]),
)

tr_idx = [int(k) for k, v in result.assignments.items() if v == "train"]
te_idx = [int(k) for k, v in result.assignments.items() if v == "test"]
```

---

## Step 6 — Train with class weighting

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_feats[tr_idx])
X_te = scaler.transform(X_feats[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",   # handles severe imbalance
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
print(classification_report(y_te, y_pred, target_names=["non_seizure", "seizure"]))

# False positive rate per hour (clinical metric)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
test_hours = (len(te_idx) * window_s) / 3600.0
fpr_per_hour = fp / max(test_hours, 1e-6)
print(f"False positives per hour: {fpr_per_hour:.2f}")
```

---

## Validation summary

| Gate | Check |
|---|---|
| EDF loadability | MNE reads all EDF files |
| Seizure interval parse | `chb01-summary.txt` parsed before windowing |
| No cross-file leakage | `group_columns=["file"]` in split optimizer |
| Class imbalance | Printed and `class_weight="balanced"` used |
| FPR/hour reported | Clinical-relevant metric in Step 6 |
| Research-only disclaimer | Dataset card + comment in code |
