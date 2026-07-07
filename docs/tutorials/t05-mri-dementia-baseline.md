# T05 — MRI Dementia Research Baseline

**Dataset:** OASIS-1 (416 subjects, T1 MRI + clinical CSV)  
**Task:** CDR=0 vs CDR>0 (research label; not a clinical diagnostic tool)  
**First model:** Logistic Regression / SVM on clinical + image QC features  
**Later model:** Subject-level 3D CNN (only after leakage/confound enforcement)  
**Difficulty:** Intermediate  
**Not for clinical diagnosis. Research purposes only.**

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn,mri]'
```

**Registration required:** Register at https://sites.wustl.edu/oasisbrains/home/oasis-1/  
Download the T1 archives and `oasis_cross-sectional.csv`.  
Extract to `/data/oasis1/`.

---

## Step 1 — Load data

```python
from pathlib import Path
from qortex.datasets import oasis1

card = oasis1.describe()
print(card.access_instructions)

bundle = oasis1.load_data(
    local_root=Path("/data/oasis1"),
    cdr_binary=True,              # CDR=0 → 0; CDR>0 → 1
    exclude_missing_cdr=True,
)
bundle.info()
# label_map: {0: 'no_dementia', 1: 'dementia'}
```

---

## Step 2 — Confound report

Age, sex, and MMSE are covariates — not hidden leakage.
Run the confound report to understand their association with CDR.

```python
import numpy as np
from qortex.neuroclassic import compute_statistical_diagnostics

# Build a flat row table for the statistical diagnostics module
subjects = bundle.subjects
labels = bundle.labels  # CDR binary

rows = []
for sid in subjects:
    meta = bundle.metadata.get(sid, {})
    rows.append({
        "subject": sid,
        "cdr_binary": meta.get("cdr_binary"),
        "Age": meta.get("Age"),
        "M/F": meta.get("M/F"),
        "MMSE": meta.get("MMSE"),
        "nWBV": meta.get("nWBV"),   # normalized whole brain volume
        "eTIV": meta.get("eTIV"),
    })

diag = compute_statistical_diagnostics(
    rows,
    target_column="cdr_binary",
    covariate_columns=["Age", "M/F", "MMSE", "nWBV", "eTIV"],
)
for assoc in diag.confound_associations:
    print(f"{assoc.variable:12s}: stat={assoc.statistic:.3f}, p={assoc.p_value:.4f}")
```

**Expected:** Age and nWBV will show significant association with CDR.
Document this — do not remove these variables silently.

---

## Step 3 — Subject-level split

```python
from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints

rows_split = [
    {"id": sid, "cdr": str(bundle.metadata[sid].get("cdr_binary", -1))}
    for sid in bundle.subjects
]
result = assign_leakage_safe_splits(
    rows_split,
    id_column="id",
    constraints=SplitConstraints(
        stratify_column="cdr",   # balance classes across splits
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
    ),
)
print(result.optimality_status)
print(result.class_distribution)

train_subs = {k for k, v in result.assignments.items() if v == "train"}
test_subs  = {k for k, v in result.assignments.items() if v == "test"}
```

---

## Step 4 — Image QC

```python
# Load NIfTI images for QC (first 10 training subjects)
train_paths = [p for s, p in zip(bundle.subjects, bundle.local_paths) if s in train_subs]
bundle_sub = bundle   # reuse the same bundle, just check a subset

qc_reports = bundle.run_qc(max_subjects=10)
for rpt in qc_reports:
    flag = "⚠ outlier" if rpt.is_outlier else "✓"
    print(f"{rpt.scope}: SNR={rpt.snr:.1f}, CNR={rpt.cnr:.1f} {flag}")
```

---

## Step 5 — Build tabular feature matrix

This tutorial uses clinical + image-QC-derived features only.
No voxel-level features — that requires a 3D CNN pipeline.

```python
import numpy as np

def subject_features(sid, meta, qc_map):
    """Tabular features from clinical CSV + image QC."""
    row = []
    # Clinical covariates (mean-impute missing)
    row.append(float(meta.get("Age", 70.0)))
    row.append(1.0 if str(meta.get("M/F", "M")) == "F" else 0.0)
    row.append(float(meta.get("MMSE", 27.0)))
    row.append(float(meta.get("nWBV", 0.73)))
    row.append(float(meta.get("eTIV", 1500.0)))
    # Image QC features (if available)
    qc = qc_map.get(sid)
    if qc:
        row.extend([qc.snr, qc.cnr, float(qc.is_outlier)])
    else:
        row.extend([0.0, 0.0, 0.0])
    return row

qc_map = {}  # sid → ImageQualityReport (populated by run_qc above)
for rpt in qc_reports:
    qc_map[rpt.scope] = rpt

all_subs = bundle.subjects
X_all = np.array([subject_features(s, bundle.metadata.get(s, {}), qc_map)
                  for s in all_subs], dtype=np.float32)
y_all = bundle.labels

tr_idx = [i for i, s in enumerate(all_subs) if s in train_subs]
te_idx = [i for i, s in enumerate(all_subs) if s in test_subs]
```

---

## Step 6 — Train and evaluate

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_all[tr_idx])
X_te = scaler.transform(X_all[te_idx])
y_tr, y_te = y_all[tr_idx], y_all[te_idx]

clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)[:, 1]

print(classification_report(y_te, y_pred, target_names=["no_dementia", "dementia"]))
print(f"AUROC: {roc_auc_score(y_te, y_prob):.3f}")
print()
print("⚠ These results reflect a research baseline model, not a clinical tool.")
print("  CDR is a research label; this model does not diagnose Alzheimer's disease.")
```

---

## Validation summary

| Gate | Check |
|---|---|
| T1 file found | `_find_nifti()` in `oasis1.load_data()` |
| Clinical CSV join | `oasis_cross-sectional.csv` parsed before loading |
| CDR binary label | CDR=0 → 0; CDR>0 → 1; missing CDR excluded |
| Subject-level split | `assign_leakage_safe_splits(group_columns=[...])` |
| Confound report | `compute_statistical_diagnostics()` in Step 2 |
| Research disclaimer | Printed in Step 6 and dataset card |








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-participants.png" alt="Histogram of participant ages and bar chart of sex values from ds000001 participants.tsv.">
  <figcaption>Real `participants.tsv` loaded through `Dataset.participants(prefer_api=False)`.</figcaption>
</figure>

```python
participants = ds.participants(prefer_api=False)
print(participants.shape)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->
