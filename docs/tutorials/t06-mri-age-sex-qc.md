# T06 — MRI Age Regression / Sex Classification + QC

**Dataset:** IXI (~600 healthy subjects, multimodal MRI)  
**Task:** Age regression (continuous) and sex classification (binary)  
**First model:** Ridge regression for age; Logistic Regression for sex  
**Later model:** 3D DenseNet  
**Difficulty:** Intermediate

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn,mri]'
pip install openpyxl   # for IXI.xls demographic spreadsheet
```

**Download:** No registration required.  
T1 images: https://brain-development.org/ixi-dataset/  
Demographics: download `IXI.xls` from the same page.  
Extract T1 NIfTI files to `/data/ixi/IXI-T1/` and place `IXI.xls` in `/data/ixi/`.

---

## Step 1 — Inspect and load

```python
from pathlib import Path
from qortex.datasets import ixi

card = ixi.describe()
print(card)

# Age regression
bundle_age = ixi.load_data(
    local_root=Path("/data/ixi"),
    modalities=["T1"],
    task="age_regression",
    max_subjects=100,
)
bundle_age.info()
# label_col = "AGE"

# Sex classification
bundle_sex = ixi.load_data(
    local_root=Path("/data/ixi"),
    modalities=["T1"],
    task="sex_classification",
    max_subjects=100,
)
bundle_sex.info()
# label_map = {0: 'male', 1: 'female'}
```

---

## Step 2 — Scanner / site QC

IXI data comes from 3 hospitals with different scanner hardware.
Site-related variance can inflate model performance.

```python
import numpy as np
from collections import Counter

# Site distribution
sites = [bundle_age.metadata.get(sid, {}).get("site", "unknown")
         for sid in bundle_age.subjects]
site_counts = Counter(sites)
print("Site distribution:", dict(site_counts))

# Age distribution per site
from collections import defaultdict
age_by_site = defaultdict(list)
for sid, site in zip(bundle_age.subjects, sites):
    age = bundle_age.metadata.get(sid, {}).get("AGE")
    if age:
        age_by_site[site].append(float(age))

for site, ages in age_by_site.items():
    print(f"{site}: n={len(ages)}, age={np.mean(ages):.1f}±{np.std(ages):.1f}")
```

---

## Step 3 — Image QC

```python
# Load and QC first 10 images
bundle_age.load_images(max_subjects=10)
qc_reports = bundle_age.run_qc(max_subjects=10)

for rpt in qc_reports:
    flag = "⚠" if rpt.is_outlier else "✓"
    shape = rpt.image_shape if hasattr(rpt, 'image_shape') else "?"
    vox   = rpt.voxel_sizes_mm if hasattr(rpt, 'voxel_sizes_mm') else "?"
    print(f"{flag} {rpt.scope}: shape={shape}, voxels={vox}")
```

---

## Step 4 — Tabular baseline features

Instead of loading all 100 volumes into memory, use demographic features
for a tabular baseline first.

```python
def tabular_features(bundle, include_site=True):
    """Build feature matrix from demographics only."""
    X_rows, y_vals, subjects = [], [], []
    for sid in bundle.subjects:
        meta = bundle.metadata.get(sid, {})
        age = meta.get("AGE")
        sex = meta.get("SEX_ID")
        site = meta.get("site", "unknown")
        if age is None or sex is None:
            continue
        row = [float(age), float(sex)]
        if include_site:
            row.append({"Guys": 0, "HH": 1, "IOP": 2}.get(site, -1))
        X_rows.append(row)
        label = bundle.labels[bundle.subjects.index(sid)] if bundle.labels is not None else 0
        y_vals.append(float(label))
        subjects.append(sid)
    import numpy as np
    return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=np.float32), subjects

X_age, y_age, subs_age = tabular_features(bundle_age, include_site=True)
X_sex, y_sex, subs_sex = tabular_features(bundle_sex, include_site=True)
```

---

## Step 5 — Subject-level split

```python
from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints

rows_age = [{"id": s} for s in subs_age]
result = assign_leakage_safe_splits(
    rows_age, id_column="id",
    constraints=SplitConstraints(train_fraction=0.7, val_fraction=0.1, test_fraction=0.2),
)
tr_idx = [i for i, s in enumerate(subs_age) if result.assignments.get(s) == "train"]
te_idx = [i for i, s in enumerate(subs_age) if result.assignments.get(s) == "test"]
```

---

## Step 6 — Age regression

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_age[tr_idx])
X_te = scaler.transform(X_age[te_idx])

reg = Ridge(alpha=1.0)
reg.fit(X_tr, y_age[tr_idx])
y_pred_age = reg.predict(X_te)

print(f"Age MAE  : {mean_absolute_error(y_age[te_idx], y_pred_age):.2f} years")
print(f"Age R²   : {r2_score(y_age[te_idx], y_pred_age):.3f}")
```

---

## Step 7 — Sex classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

rows_sex = [{"id": s} for s in subs_sex]
result_sex = assign_leakage_safe_splits(
    rows_sex, id_column="id",
    constraints=SplitConstraints(train_fraction=0.7, val_fraction=0.1, test_fraction=0.2),
)
tr_sex = [i for i, s in enumerate(subs_sex) if result_sex.assignments.get(s) == "train"]
te_sex = [i for i, s in enumerate(subs_sex) if result_sex.assignments.get(s) == "test"]

X_tr_s = scaler.fit_transform(X_sex[tr_sex])
X_te_s = scaler.transform(X_sex[te_sex])

clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_tr_s, y_sex[tr_sex])
y_pred_sex = clf.predict(X_te_s)
print(classification_report(y_sex[te_sex], y_pred_sex, target_names=["male", "female"]))
```

---

## Validation summary

| Gate | Check |
|---|---|
| NIfTI loadability | `bundle.load_images()` |
| Demographic join | `ixi.load_demographics()` parses IXI.xls/csv |
| Scanner/site profile | Step 2 — printed before training |
| Voxel-size QC | `run_qc()` reports voxel sizes and outlier flag |
| Subject-level split | `assign_leakage_safe_splits()` |
| Age MAE + R² | Regression metrics in Step 6 |








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
