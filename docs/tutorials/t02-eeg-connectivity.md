# T02 — EEG Connectivity and Graph Features

**Dataset:** PhysioNet EEGBCI (same data as T01)  
**Task:** Eyes-open vs eyes-closed resting state; optional motor-imagery condition comparison  
**First model:** Connectivity graph metrics + Logistic Regression / Random Forest  
**Later model:** Graph kernel or shallow MLP after graph API stabilises  
**Difficulty:** Beginner

---

## Prerequisites

```bash
pip install 'qortex[tutorials,sklearn]'
```

---

## Step 1 — Load resting-state baseline

```python
from qortex.datasets import eegbci

# Runs 1 = eyes open, 2 = eyes closed (baseline, no task events)
bundle = eegbci.load_data(
    subjects=[1, 2, 3, 4, 5],
    runs=[1, 2],
    preload=True,
)
bundle.info()
# label_map: {1: 'eyes_open', 2: 'eyes_closed'}
```

---

## Step 2 — Extract epochs (sliding windows)

```python
import numpy as np

X, y = bundle.to_windows(
    window_s=4.0,
    bandpass=(8.0, 13.0),   # alpha band
    event_driven=False,      # sliding windows for resting state
    overlap=0.5,
)
print(f"X: {X.shape}, y: {y.shape}")
```

---

## Step 3 — Compute Pearson and PLV connectivity matrices

```python
from qortex.neuroclassic import (
    compute_pearson_connectivity,
    compute_phase_locking_value_connectivity,
)

# Average connectivity across all training epochs
X_mean = X.mean(axis=0)                     # [64, 640]
ch_names = bundle.channel_names
conn = compute_pearson_connectivity(
    X_mean,
    channel_names=ch_names,
    sampling_hz=bundle.sfreq,
    time_window_s=4.0,
    frequency_band=(8.0, 13.0),
    threshold=0.3,
)
print(f"Connectivity matrix: {conn.matrix.shape}")  # (64, 64)
print(conn.spec.summary())

plv_conn = compute_phase_locking_value_connectivity(
    X_mean,
    channel_names=ch_names,
    sampling_hz=bundle.sfreq,
    time_window_s=4.0,
    frequency_band=(8.0, 13.0),
    threshold=0.5,
)
print(plv_conn.spec.summary())
```

---

## Step 4 — Compute graph metrics

```python
from qortex.neuroclassic import compute_graph_metrics

graph_report = compute_graph_metrics(conn)

print(f"Clustering coefficient : {graph_report.clustering_coefficient:.4f}")
print(f"Mean path length       : {graph_report.mean_path_length}")
print(f"Small-world σ          : {graph_report.small_world_sigma}")
print(f"Modularity Q           : {graph_report.modularity}")
print(f"Communities            : {set(graph_report.community_assignments)}")
print(f"Betweenness (top 5)    : {sorted(graph_report.betweenness_centrality, reverse=True)[:5]}")
```

---

## Step 5 — Build per-epoch feature vectors

```python
def epoch_graph_features(epoch, ch_names, sfreq):
    """Return graph metric feature vector for one epoch."""
    conn = compute_pearson_connectivity(
        epoch,
        channel_names=ch_names,
        sampling_hz=sfreq,
        time_window_s=4.0,
        frequency_band=(8.0, 13.0),
        threshold=0.3,
    )
    gr = compute_graph_metrics(conn)
    bc = gr.betweenness_centrality or [0.0] * len(ch_names)
    comm = gr.community_assignments or [0] * len(ch_names)
    feats = [
        gr.clustering_coefficient or 0.0,
        gr.mean_path_length or 0.0,
        gr.small_world_sigma or 0.0,
        gr.modularity or 0.0,
        gr.density,
        float(max(set(comm), key=comm.count)),   # dominant community
        float(max(bc)),                           # hub centrality
    ]
    return np.array(feats, dtype=np.float32)

X_graph = np.array([epoch_graph_features(ep, ch_names, bundle.sfreq) for ep in X])
print(f"Graph feature matrix: {X_graph.shape}")
```

---

## Step 6 — Classify eyes-open vs eyes-closed

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Only use epochs that have valid labels (1=eyes_open, 2=eyes_closed)
mask = y > 0
X_valid = X_graph[mask]
y_valid = y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_scaled, y_valid, cv=cv, scoring="f1_macro",
)
print(f"CV macro-F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Step 7 — Condition comparison plot (alpha power)

```python
# Compare connectivity density between conditions
density_by_condition = {}
for label, name in bundle.label_map.items():
    mask = y == label
    if mask.sum() == 0:
        continue
    X_cond = X[mask].mean(axis=0)
    c = compute_pearson_connectivity(
        X_cond,
        channel_names=ch_names,
        sampling_hz=bundle.sfreq,
        time_window_s=4.0,
        frequency_band=(8.0, 13.0),
        threshold=0.3,
    )
    g = compute_graph_metrics(c)
    density_by_condition[name] = g.density
    print(f"{name}: density={g.density:.3f}")
```

---

## Step 8 — Next steps

- Use **full sensor-level adjacency** from `graph_report.betweenness_centrality` to rank hub electrodes.
- Run on **motor imagery runs (4, 8, 12)** to compare task vs baseline graph topology.
- Feed graph features into a **graph neural network** with PyTorch Geometric.

---

## Validation summary

| Gate | Check |
|---|---|
| Channel sets match | All files have 64 channels (enforced in `eegbci.load_data`) |
| Band explicitly set | 8–13 Hz alpha passed to `to_windows()` |
| Threshold explicit | Absolute 0.3 passed to `compute_pearson_connectivity()` |
| Graph construction logged | `GraphMetricReport` carries all computed values |
| Condition labels | Run 1 = eyes_open, run 2 = eyes_closed (set in `label_map`) |
