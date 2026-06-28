# T03 — EEG Sleep-Stage Classification

**Dataset:** Sleep-EDF Expanded (197 PSG recordings)  
**Task:** 5-class sleep staging: Wake / N1 / N2 / N3 / REM  
**First model:** PSD bandpower + Random Forest  
**Later model:** 1D CNN or Braindecode sleep model  
**Difficulty:** Intermediate

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn]'
```

---

## Step 1 — Load data

```python
from qortex.datasets import sleep_edf

card = sleep_edf.describe()
print(card)

bundle = sleep_edf.load_data(
    subjects=[0, 1, 2, 3],
    recording="SC",          # Cassette recordings (sleep at home)
    crop_wake_mins=30,       # trim leading/trailing wake to reduce imbalance
    preload=True,
)
bundle.info()
```

**Label mapping (AASM-aligned)**

| Annotation | Hypnogram code | Class | Label |
|---|---|---|---|
| Wake | `W` | 0 | Wake |
| Stage 1 | `1` | 1 | N1 |
| Stage 2 | `2` | 2 | N2 |
| Stage 3 / 4 | `3` / `4` | 3 | N3 |
| REM | `R` | 4 | REM |
| Movement | `M` | — | excluded |
| Unknown | `?` | — | excluded |

---

## Step 2 — Validation gates

```python
for raw in bundle.raws:
    assert raw.info["sfreq"] == 100.0, f"Unexpected sfreq: {raw.info['sfreq']}"
    n_ann = len(raw.annotations)
    assert n_ann > 0, f"No annotations in {raw.filenames[0]}"
    # Check no excluded annotations leaked through
    bad = [a for a in raw.annotations.description if a in {"M", "?", "Sleep stage M", "Sleep stage ?"}]
    assert not bad, f"Excluded annotations present: {bad}"

print(f"✓ {len(bundle.raws)} PSG files validated")
```

---

## Step 3 — Extract 30-second epochs

Standard sleep staging uses 30-second non-overlapping windows.

```python
import numpy as np

X, y = bundle.to_windows(
    window_s=30.0,
    event_driven=False,       # sliding, no overlap
    overlap=0.0,
)
print(f"X: {X.shape}")        # (n_epochs, n_channels, 3000)

# Class distribution
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    name = bundle.label_map.get(int(u), str(u))
    print(f"  {name}: {c} ({100*c/len(y):.1f}%)")
```

---

## Step 4 — Feature extraction

```python
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),   # sleep spindles
    "beta":  (16.0, 30.0),
}

def extract_sleep_features(X, sfreq=100.0):
    """Bandpower + statistical features per epoch."""
    features = []
    for epoch in X:
        row = []
        for ch in epoch:
            f, psd = welch(ch, sfreq, nperseg=min(512, len(ch)))
            for lo, hi in BANDS.values():
                mask = (f >= lo) & (f <= hi)
                row.append(float(psd[mask].mean()) if mask.any() else 0.0)
            # Time-domain features
            row.extend([float(ch.mean()), float(ch.std()), float(np.abs(ch).max())])
        features.append(row)
    return np.array(features, dtype=np.float32)

X_feats = extract_sleep_features(X, sfreq=bundle.sfreq)
print(f"Feature matrix: {X_feats.shape}")
```

---

## Step 5 — Subject-held-out split

Sleep staging must use **subject-level** splits to prevent leakage.

```python
from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints

# Assign epoch-to-subject (approximation: epochs from file i → subject i//files_per_subject)
n_epochs = len(y)
files_per_subject = max(1, len(bundle.raws) // len(bundle.subjects))
epoch_subjects = [bundle.subjects[min(i // (n_epochs // max(len(bundle.subjects), 1)),
                                     len(bundle.subjects)-1)]
                  for i in range(n_epochs)]

rows = [{"id": str(i), "subject": str(s)} for i, s in enumerate(epoch_subjects)]
result = assign_leakage_safe_splits(
    rows,
    id_column="id",
    constraints=SplitConstraints(group_columns=["subject"]),
)
print(result.optimality_status)

tr_idx = [int(k) for k, v in result.assignments.items() if v == "train"]
te_idx = [int(k) for k, v in result.assignments.items() if v == "test"]
```

---

## Step 6 — Train and evaluate

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_feats[tr_idx])
X_te = scaler.transform(X_feats[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
target_names = [bundle.label_map[k] for k in sorted(bundle.label_map)]
print(classification_report(y_te, y_pred, target_names=target_names))
print(f"Cohen κ: {cohen_kappa_score(y_te, y_pred):.3f}")
```

---

## Validation summary

| Gate | Check |
|---|---|
| PSG + hypnogram pair | Both files exist per subject |
| Annotation coverage | `n_ann > 0` for every recording |
| 30 s alignment | `window_s=30.0, overlap=0.0` |
| Excluded annotations | M and ? filtered in `sleep_edf.load_data()` |
| Subject-held-out split | `group_columns=["subject"]` in split optimizer |
| Class distribution printed | Step 3 |
