"""Export a ConnectivityMatrix + GraphMetricReport to the JSON shape the
web/ frontend consumes (web/src/types.ts: ConnectomeData).

Run from the repo root:
    python3 web/scripts/export_connectome.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from qortex.neuroclassic.connectivity import ConnectivityMatrix, ConnectivitySpec, compute_graph_metrics


def build_demo_matrix() -> ConnectivityMatrix:
    rng = np.random.default_rng(7)
    networks = ["VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN", "CBL", "SUB"]
    n = len(networks)
    base = rng.normal(0, 0.12, size=(n, n))
    for i in range(0, n, 3):
        base[i:i + 3, i:i + 3] += 0.5
    m = (base + base.T) / 2
    np.fill_diagonal(m, 1.0)
    m = np.clip(m, -1, 1)
    m[np.abs(m) < 0.2] = 0.0

    spec = ConnectivitySpec(
        input_signal_type="fMRI_BOLD", node_definition="Schaefer-400 (17 networks)",
        parcellation_or_channel_set=networks, time_window_s=300.0,
        preprocessing_assumptions=["ICA-AROMA"], connectivity_metric="pearson",
    )
    return ConnectivityMatrix(matrix=m, node_labels=networks, spec=spec, computed_at="demo")


def to_connectome_json(matrix: ConnectivityMatrix, *, threshold: float = 0.2) -> dict:
    metrics = compute_graph_metrics(matrix, scope="web-demo")
    m = matrix.matrix
    n = m.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(m[i, j])
            if abs(w) >= threshold:
                edges.append({"source": i, "target": j, "weight": w})
    nodes = [
        {"label": label, "degree": float(metrics.degree[i]) if metrics.degree else 0.0}
        for i, label in enumerate(matrix.node_labels)
    ]
    return {
        "metric": matrix.spec.connectivity_metric,
        "node_definition": matrix.spec.node_definition,
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    matrix = build_demo_matrix()
    payload = to_connectome_json(matrix)
    out_path = Path(__file__).resolve().parents[1] / "public" / "assets" / "connectome.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(payload['nodes'])} nodes, {len(payload['edges'])} edges)")


if __name__ == "__main__":
    main()
