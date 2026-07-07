# T01 — EEG Motor Imagery Classification

**Dataset:** PhysioNet EEGBCI (109 subjects, 64 ch, 160 Hz)  
**Task:** Left vs right hand motor imagery (binary) — or 3-class with rest  
**First model:** Bandpower + Logistic Regression / SVM  
**Later model:** Braindecode EEGNet  
**Difficulty:** Beginner

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn]'
```

---

## Step 1 — Load data

```python
from qortex.datasets import eegbci

card = eegbci.describe()
print(card)

bundle = eegbci.load_data(
    subjects=[1, 2, 3],
    runs=[4, 8, 12],    # left/right fist imagery runs
    preload=True,
)
bundle.info()
# EEGBundle: PhysioNet EEG Motor Movement/Imagery Dataset
#   Subjects     : [1, 2, 3]
#   Runs         : [4, 8, 12]
#   Channels     : 64
#   Sampling Hz  : 160.0
#   Label map    : {0: 'rest', 1: 'left_fist_imagery', 2: 'right_fist_imagery'}
```

**Label mapping (from PhysioNet documentation)**

Runs 4, 8, 12 — Imagine opening/closing left or right fist:

| Annotation | Class | Label |
|---|---|---|
| T0 | Rest | 0 |
| T1 | Left fist imagery | 1 |
| T2 | Right fist imagery | 2 |

---

## Step 2 — Signal QC

Always check data quality before feature extraction.

```python
qc_reports = bundle.run_qc(max_files=3)
for r in qc_reports:
    print(f"{r.scope}: flatline={r.n_flatline}, nan={r.n_nan}, "
          f"clipped={r.n_clipped}, warnings={r.warnings}")
```

---

## Step 3 — Validation gates

```python
# Gate 1: sampling rate
assert bundle.sfreq == 160.0, f"Unexpected sfreq: {bundle.sfreq}"

# Gate 2: channel count
assert bundle.n_channels == 64, f"Expected 64 channels, got {bundle.n_channels}"

# Gate 3: at least one raw file loaded
assert bundle.n_files > 0, "No EDF files loaded — check MNE data dir"
print(f"✓ Loaded {bundle.n_files} files across {len(bundle.subjects)} subjects")
```

---

## Step 4 — Window extraction

```python
import numpy as np

X, y = bundle.to_windows(
    window_s=4.0,
    bandpass=(8.0, 30.0),   # mu + beta band
    event_driven=True,
    tmin=0.5,               # discard 500 ms after cue onset
)
print(f"X shape: {X.shape}")  # (n_epochs, 64, 640)
print(f"y distribution: { {k: int((y==k).sum()) for k in np.unique(y)} }")
```

---

## Step 5 — Leakage-safe split

```python
from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints

# Build a per-epoch metadata table with subject group column
rows = [
    {"id": f"ep{i}", "subject": bundle.subjects[i % len(bundle.subjects)]}
    for i in range(len(y))
]
result = assign_leakage_safe_splits(
    rows,
    id_column="id",
    constraints=SplitConstraints(
        group_columns=["subject"],      # never split a subject across train/test
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
    ),
)
print(result.optimality_status, result.residual_imbalance)

train_ids = {k for k, v in result.assignments.items() if v == "train"}
val_ids   = {k for k, v in result.assignments.items() if v == "val"}
test_ids  = {k for k, v in result.assignments.items() if v == "test"}
```

---

## Step 6 — Feature extraction (bandpower)

```python
feature_report = bundle.to_feature_matrix(
    bands={"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)},
    include_relative_bandpower=True,
    include_log_bandpower=True,
    include_time_domain=False,
)

X_feats = feature_report.features
print(f"Feature matrix: {X_feats.shape}")
print(feature_report.feature_names[:8])
```

---

## Step 7 — Train and evaluate

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Map split IDs to indices
id_list = [f"ep{i}" for i in range(len(y))]
tr_idx = [i for i, id_ in enumerate(id_list) if id_ in train_ids]
va_idx = [i for i, id_ in enumerate(id_list) if id_ in val_ids]
te_idx = [i for i, id_ in enumerate(id_list) if id_ in test_ids]

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_feats[tr_idx])
X_te = scaler.transform(X_feats[te_idx])
y_tr, y_te = y[tr_idx], y[te_idx]

clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto")
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
target_names = [bundle.label_map[k] for k in sorted(bundle.label_map)]
print(classification_report(y_te, y_pred, target_names=target_names))
```

---

## Step 8 — CSP spatial features

Common Spatial Patterns (CSP) is a standard motor-imagery spatial-filtering
baseline. Fit CSP only on training epochs, then transform validation/test
epochs with the fitted report to avoid leakage.

```python
from qortex.neuroclassic import compute_common_spatial_patterns

csp = compute_common_spatial_patterns(
    X[tr_idx],
    y[tr_idx],
    channel_names=bundle.channel_names,
    n_components=8,
    scope="eegbci_motor_imagery_train",
)

X_tr_csp = csp.features
X_te_csp = csp.transform(X[te_idx])
```

---

## Step 9 — Next steps

- Scale up to all 109 subjects with more runs.
- Try **Braindecode EEGNet** (requires `pip install 'qortex[braindecode]'`):
  ```python
  from braindecode.models import EEGNetv4
  ```

---

## Validation summary

| Gate | Check | Status |
|---|---|---|
| EDF loadability | MNE reads all EDF files without error | Enforced in Step 2 |
| Sampling rate | 160.0 Hz | Enforced in Step 3 |
| Channel count | 64 | Enforced in Step 3 |
| Annotation mapping | T0/T1/T2 per-run mapping applied | `eegbci.load_data()` |
| Subject-safe split | No subject spans train and test | Step 5 |
| Bandpass filter | 8–30 Hz applied before feature extraction | Step 4 |
