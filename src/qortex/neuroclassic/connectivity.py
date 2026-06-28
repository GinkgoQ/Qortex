"""Connectivity matrix construction and graph metrics.

Every connectivity feature declares its construction path explicitly.
No graph metrics are computed on ambiguous or undeclared graphs.

Install extras:
    pip install 'qortex[neuroclassic]'
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qortex.neuroclassic._base import (
    MethodConfidence,
    MetricResult,
    NeuroClassicResult,
    NeuroClassicSpec,
    _timer,
)

__version__ = "0.1.0"


@dataclass
class ConnectivitySpec:
    """Full construction-path declaration for a connectivity matrix."""
    input_signal_type: str           # e.g. "EEG", "fMRI_BOLD", "iEEG"
    node_definition: str             # e.g. "EEG_channel", "parcel_atlas_Schaefer100"
    parcellation_or_channel_set: list[str]
    time_window_s: float
    preprocessing_assumptions: list[str]
    connectivity_metric: str         # "pearson", "spearman", "PLV", "coherence", "MI"
    frequency_band: tuple[float, float] | None = None   # (low_hz, high_hz) or None
    thresholding_rule: str | None = None    # e.g. "absolute > 0.5", "proportional top 10%"
    matrix_symmetry: str = "symmetric"     # "symmetric" or "asymmetric"
    edge_weight_meaning: str = "correlation_coefficient"
    missing_node_behavior: str = "exclude"  # "exclude", "zero", "impute_mean"

    def summary(self) -> str:
        parts = [
            f"Connectivity matrix built from {len(self.parcellation_or_channel_set)} "
            f"{self.input_signal_type} {self.node_definition} nodes",
            f"using {self.connectivity_metric}",
            f"over {self.time_window_s:.1f}-second windows",
        ]
        if self.frequency_band:
            parts.append(f"in {self.frequency_band[0]}-{self.frequency_band[1]} Hz band")
        if self.thresholding_rule:
            parts.append(f"threshold: {self.thresholding_rule}")
        return "; ".join(parts) + "."


@dataclass
class ConnectivityMatrix:
    """Adjacency matrix with full construction metadata."""
    matrix: np.ndarray          # [n_nodes, n_nodes]
    node_labels: list[str]
    spec: ConnectivitySpec
    computed_at: str = ""

    @property
    def n_nodes(self) -> int:
        return self.matrix.shape[0]

    def to_dict(self, include_matrix: bool = False) -> dict:
        d = {
            "n_nodes": self.n_nodes,
            "node_labels": self.node_labels,
            "spec": {
                "metric": self.spec.connectivity_metric,
                "nodes": self.spec.node_definition,
                "time_window_s": self.spec.time_window_s,
                "frequency_band": self.spec.frequency_band,
                "threshold": self.spec.thresholding_rule,
                "summary": self.spec.summary(),
            },
            "computed_at": self.computed_at,
        }
        if include_matrix:
            d["matrix"] = self.matrix.tolist()
        return d


@dataclass
class GraphMetricReport:
    """Graph-theoretic summary computed from a ConnectivityMatrix."""
    scope: str
    n_nodes: int
    n_edges: int
    density: float
    mean_degree: float
    mean_strength: float | None
    clustering_coefficient: float | None
    global_efficiency: float | None
    modularity: float | None
    n_connected_components: int
    degree: list[float] = field(default_factory=list)
    strength: list[float] = field(default_factory=list)
    construction_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_result(self, method_version: str = __version__) -> NeuroClassicResult:
        metrics = [
            MetricResult("n_nodes", self.n_nodes),
            MetricResult("n_edges", self.n_edges),
            MetricResult("density", self.density,
                         interpretation="Fraction of possible edges present"),
            MetricResult("mean_degree", self.mean_degree),
            MetricResult("mean_strength", self.mean_strength),
            MetricResult("clustering_coefficient", self.clustering_coefficient),
            MetricResult("global_efficiency", self.global_efficiency),
            MetricResult("modularity", self.modularity),
            MetricResult("n_connected_components", self.n_connected_components),
        ]
        return NeuroClassicResult(
            method_name="connectivity_graph",
            method_version=method_version,
            modality="eeg",
            scope=self.scope,
            inputs={"n_nodes": self.n_nodes},
            parameters={"construction": self.construction_summary},
            assumptions=["Graph is undirected unless spec states otherwise.",
                         "Adjacency matrix is symmetric."],
            metrics=metrics,
            warnings=self.warnings,
            confidence=self.confidence,
            provenance={
                "method": "connectivity_graph",
                "construction": self.construction_summary,
            },
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "mean_degree": self.mean_degree,
            "mean_strength": self.mean_strength,
            "clustering_coefficient": self.clustering_coefficient,
            "global_efficiency": self.global_efficiency,
            "modularity": self.modularity,
            "n_connected_components": self.n_connected_components,
            "construction_summary": self.construction_summary,
            "warnings": self.warnings,
            "confidence": self.confidence.value,
        }


# ── Connectivity computation ──────────────────────────────────────────────────

def compute_pearson_connectivity(
    data: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    time_window_s: float = 2.0,
    sampling_hz: float = 256.0,
    frequency_band: tuple[float, float] | None = None,
    threshold: float | None = None,
    scope: str = "unknown",
) -> ConnectivityMatrix:
    """Compute Pearson correlation connectivity matrix.

    Construction path is fully declared in the returned ConnectivityMatrix.

    Parameters
    ----------
    data:
        [n_channels, n_times] float array.
    channel_names:
        Node labels.
    time_window_s:
        Window over which correlation is computed (uses full recording if n_times ≤ win_samples).
    sampling_hz:
        Sampling rate in Hz.
    frequency_band:
        Optional (low, high) Hz for bandpass filtering before correlation.
    threshold:
        Absolute correlation threshold; edges below this are zeroed.
    scope:
        Recording identifier.

    Returns
    -------
    ConnectivityMatrix
        Full adjacency matrix with construction metadata.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be [n_channels, n_times]; got {data.shape}")
    n_ch, n_t = data.shape
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_ch)]

    preprocessing = ["z-score per channel before correlation"]
    working_data = data.copy().astype(np.float64)

    # Optional bandpass
    if frequency_band is not None:
        try:
            from scipy.signal import butter, filtfilt
            lo, hi = frequency_band
            nyq = sampling_hz / 2
            b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
            for i in range(n_ch):
                if np.isfinite(working_data[i]).all():
                    working_data[i] = filtfilt(b, a, working_data[i])
            preprocessing.append(f"bandpass filtered {lo}-{hi} Hz (Butterworth order 4)")
        except ImportError:
            pass

    # Use first time_window_s of data
    win_samples = min(int(time_window_s * sampling_hz), n_t)
    segment = working_data[:, :win_samples]

    # Z-score
    mu = np.nanmean(segment, axis=1, keepdims=True)
    sigma = np.nanstd(segment, axis=1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    segment = (segment - mu) / sigma

    # Pearson correlation
    valid_mask = np.isfinite(segment).all(axis=1)
    n_valid = int(valid_mask.sum())
    corr = np.zeros((n_ch, n_ch))
    if n_valid >= 2:
        valid_seg = segment[valid_mask]
        sub_corr = np.corrcoef(valid_seg)
        idx = np.where(valid_mask)[0]
        for ii, i in enumerate(idx):
            for jj, j in enumerate(idx):
                corr[i, j] = sub_corr[ii, jj]

    # Apply threshold
    threshold_rule = None
    if threshold is not None:
        corr = np.where(np.abs(corr) >= threshold, corr, 0.0)
        threshold_rule = f"absolute |r| >= {threshold}"

    import datetime
    spec = ConnectivitySpec(
        input_signal_type="EEG",
        node_definition="EEG_channel",
        parcellation_or_channel_set=channel_names,
        time_window_s=time_window_s,
        preprocessing_assumptions=preprocessing,
        connectivity_metric="pearson",
        frequency_band=frequency_band,
        thresholding_rule=threshold_rule,
        matrix_symmetry="symmetric",
        edge_weight_meaning="pearson_correlation_coefficient",
    )
    return ConnectivityMatrix(
        matrix=corr,
        node_labels=channel_names,
        spec=spec,
        computed_at=datetime.datetime.utcnow().isoformat(),
    )


# ── Graph metrics ─────────────────────────────────────────────────────────────

def compute_graph_metrics(
    conn: ConnectivityMatrix,
    *,
    scope: str = "unknown",
    modularity_resolution: float = 1.0,
) -> GraphMetricReport:
    """Compute graph-theoretic summary from a ConnectivityMatrix.

    All metrics are computed on the absolute-value adjacency matrix
    (undirected, weighted).  The construction summary is logged explicitly
    so reports cannot omit how the graph was built.

    Valid metrics:
        degree, strength, density, clustering coefficient,
        global efficiency, modularity, connected components.
    """
    adj = np.abs(conn.matrix).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    n = adj.shape[0]

    # Binary adjacency for some metrics
    binary = (adj > 0).astype(float)
    degrees = binary.sum(axis=1)
    strengths = adj.sum(axis=1)
    n_edges = int(degrees.sum()) // 2
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0
    mean_degree = float(degrees.mean())
    mean_strength = float(strengths.mean())

    # Clustering coefficient (Watts-Strogatz, binary)
    cc_vals = []
    for i in range(n):
        neighbors = np.where(binary[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            cc_vals.append(0.0)
            continue
        sub = binary[np.ix_(neighbors, neighbors)]
        actual_edges = sub.sum() / 2
        possible_edges = k * (k - 1) / 2
        cc_vals.append(actual_edges / possible_edges if possible_edges > 0 else 0.0)
    clustering_coefficient = float(np.mean(cc_vals)) if cc_vals else None

    # Global efficiency (harmonic mean of inverse path lengths — approx via BFS)
    global_efficiency = _global_efficiency_approx(binary, n)

    # Connected components (simple BFS)
    n_components = _count_connected_components(binary, n)

    # Modularity (greedy approximation — Louvain-like, simple)
    modularity = _simple_modularity(adj, degrees, n_edges)

    warnings = []
    confidence = MethodConfidence.HIGH
    if n_components > 1:
        warnings.append(
            f"Graph has {n_components} connected components. "
            "Path-length-based metrics (efficiency) are not meaningful across components."
        )
        confidence = MethodConfidence.LOW_CONFIDENCE
    if n < 4:
        warnings.append(f"Graph has only {n} nodes; metrics may be unreliable.")
        confidence = MethodConfidence.LOW_CONFIDENCE

    return GraphMetricReport(
        scope=scope,
        n_nodes=n,
        n_edges=n_edges,
        density=density,
        mean_degree=mean_degree,
        mean_strength=mean_strength,
        clustering_coefficient=clustering_coefficient,
        global_efficiency=global_efficiency,
        modularity=modularity,
        n_connected_components=n_components,
        degree=degrees.tolist(),
        strength=strengths.tolist(),
        construction_summary=conn.spec.summary(),
        warnings=warnings,
        confidence=confidence,
    )


# ── Graph algorithm helpers ───────────────────────────────────────────────────

def _global_efficiency_approx(binary: np.ndarray, n: int) -> float | None:
    """Global efficiency via BFS (no scipy needed)."""
    if n > 200:
        return None  # too expensive without sparse solvers

    total_inv_dist = 0.0
    for i in range(n):
        # BFS from i
        dist = [-1] * n
        dist[i] = 0
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in range(n):
                if binary[u, v] > 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for j in range(n):
            if j != i and dist[j] > 0:
                total_inv_dist += 1.0 / dist[j]

    n_pairs = n * (n - 1)
    return total_inv_dist / n_pairs if n_pairs > 0 else 0.0


def _count_connected_components(binary: np.ndarray, n: int) -> int:
    visited = [False] * n
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        queue = [start]
        while queue:
            u = queue.pop()
            if visited[u]:
                continue
            visited[u] = True
            for v in range(n):
                if binary[u, v] > 0 and not visited[v]:
                    queue.append(v)
    return n_comp


def _simple_modularity(adj: np.ndarray, degrees: np.ndarray, n_edges: int) -> float | None:
    """Newman-Girvan modularity with greedy community detection (simplified)."""
    if n_edges == 0:
        return None
    n = adj.shape[0]
    m2 = 2 * n_edges

    # Initialize each node in its own community
    communities = list(range(n))
    best_q = _modularity_q(adj, communities, degrees, m2)

    # Simple greedy merge (not full Louvain — just one pass)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            original_comm = communities[i]
            for j in range(n):
                if i == j or adj[i, j] == 0:
                    continue
                neighbor_comm = communities[j]
                if neighbor_comm == original_comm:
                    continue
                communities[i] = neighbor_comm
                q = _modularity_q(adj, communities, degrees, m2)
                if q > best_q:
                    best_q = q
                    improved = True
                    break
                else:
                    communities[i] = original_comm

    return best_q


def _modularity_q(
    adj: np.ndarray,
    communities: list[int],
    degrees: np.ndarray,
    m2: int,
) -> float:
    n = adj.shape[0]
    unique_comms = set(communities)
    q = 0.0
    for c in unique_comms:
        members = [i for i in range(n) if communities[i] == c]
        for i in members:
            for j in members:
                q += adj[i, j] - degrees[i] * degrees[j] / m2
    return q / m2 if m2 > 0 else 0.0
