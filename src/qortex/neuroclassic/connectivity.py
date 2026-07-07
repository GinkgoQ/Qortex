"""Connectivity matrix construction and graph metrics.

Every connectivity feature declares its construction path explicitly.
No graph metrics are computed on ambiguous or undeclared graphs.

Graph algorithms used:
  - BFS shortest paths (Dijkstra-equivalent on unweighted graphs) for
    global efficiency (harmonic mean of inverse path lengths) and mean
    path length (arithmetic mean of shortest path lengths).
  - Brandes (2001) BFS-based algorithm for unnormalized betweenness centrality.
  - Greedy modularity maximisation (Newman-Girvan Q) for community detection.
  - Watts-Strogatz null model (Erdős–Rényi approximation) for small-world σ.
  - Phase-locking value (PLV) for phase-synchrony connectivity.

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

# Cap for BFS-based O(n²) algorithms
_BFS_NODE_LIMIT = 200


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
    """Graph-theoretic summary computed from a ConnectivityMatrix.

    All metrics are computed on the absolute-value adjacency (undirected,
    potentially weighted).  The construction_summary field ensures the
    graph provenance is always recorded alongside the metrics.

    Algorithms:
      degree/strength/density  — direct sums over adjacency matrix
      clustering_coefficient   — Watts-Strogatz local CC (binary graph)
      global_efficiency        — harmonic mean of inverse BFS distances
      mean_path_length         — arithmetic mean of BFS shortest paths
      betweenness_centrality   — Brandes (2001) BFS algorithm, normalised
      community_assignments    — greedy modularity maximisation (Newman Q)
      modularity               — Newman-Girvan Q of detected communities
      small_world_sigma        — (C/C_rand)/(L/L_rand), ER null model
    """
    scope: str
    n_nodes: int
    n_edges: int
    density: float
    mean_degree: float
    mean_strength: float | None
    clustering_coefficient: float | None
    global_efficiency: float | None
    mean_path_length: float | None
    modularity: float | None
    n_connected_components: int
    degree: list[float] = field(default_factory=list)
    strength: list[float] = field(default_factory=list)
    betweenness_centrality: list[float] = field(default_factory=list)
    community_assignments: list[int] = field(default_factory=list)
    small_world_sigma: float | None = None
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
            MetricResult("clustering_coefficient", self.clustering_coefficient,
                         interpretation="Mean local clustering coefficient (Watts-Strogatz)"),
            MetricResult("global_efficiency", self.global_efficiency,
                         interpretation="Harmonic mean of inverse shortest-path lengths"),
            MetricResult("mean_path_length", self.mean_path_length,
                         interpretation="Arithmetic mean of all pairwise BFS shortest paths"),
            MetricResult("modularity", self.modularity,
                         interpretation="Newman-Girvan Q of greedy-detected communities"),
            MetricResult("n_connected_components", self.n_connected_components),
            MetricResult("small_world_sigma", self.small_world_sigma,
                         interpretation=(
                             "σ > 1 indicates small-world topology relative to "
                             "Erdős–Rényi null model with same density"
                         )),
            MetricResult("betweenness_centrality", self.betweenness_centrality,
                         interpretation="Normalised betweenness (Brandes 2001) per node"),
            MetricResult("community_assignments", self.community_assignments,
                         interpretation="Community label per node from greedy modularity"),
        ]
        return NeuroClassicResult(
            method_name="connectivity_graph",
            method_version=method_version,
            modality="eeg",
            scope=self.scope,
            inputs={"n_nodes": self.n_nodes},
            parameters={"construction": self.construction_summary},
            assumptions=[
                "Graph is undirected (adjacency = |matrix|).",
                "Self-loops are excluded (diagonal zeroed before computation).",
                "BFS-based metrics are skipped for n > 200 (too expensive without sparse solvers).",
            ],
            metrics=metrics,
            warnings=self.warnings,
            confidence=self.confidence,
            provenance={
                "method": "connectivity_graph",
                "construction": self.construction_summary,
                "version": method_version,
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
            "mean_path_length": self.mean_path_length,
            "modularity": self.modularity,
            "n_connected_components": self.n_connected_components,
            "betweenness_centrality": self.betweenness_centrality,
            "community_assignments": self.community_assignments,
            "small_world_sigma": self.small_world_sigma,
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
        Optional (low, high) Hz for Butterworth bandpass filtering before correlation.
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

    # Optional Butterworth bandpass
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

    # Z-score per channel
    mu = np.nanmean(segment, axis=1, keepdims=True)
    sigma = np.nanstd(segment, axis=1, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    segment = (segment - mu) / sigma

    # Pearson correlation (only on fully-finite channels)
    valid_mask = np.isfinite(segment).all(axis=1)
    corr = np.zeros((n_ch, n_ch))
    if valid_mask.sum() >= 2:
        valid_seg = segment[valid_mask]
        sub_corr = np.corrcoef(valid_seg)
        idx = np.where(valid_mask)[0]
        for ii, i in enumerate(idx):
            for jj, j in enumerate(idx):
                corr[i, j] = sub_corr[ii, jj]

    # Apply absolute threshold
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


def compute_phase_locking_value_connectivity(
    data: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    time_window_s: float = 2.0,
    sampling_hz: float = 256.0,
    frequency_band: tuple[float, float] | None = None,
    threshold: float | None = None,
    scope: str = "unknown",
) -> ConnectivityMatrix:
    """Compute phase-locking value (PLV) connectivity.

    PLV is the absolute mean phase difference between analytic signals:
    ``PLV_ij = |mean(exp(1j * (phase_i - phase_j)))|``.  Values are in
    ``[0, 1]`` where 1 means perfectly phase-locked over the selected window.

    Parameters are intentionally parallel to :func:`compute_pearson_connectivity`
    so callers can swap amplitude-correlation and phase-synchrony estimates
    without changing their graph-metric pipeline.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be [n_channels, n_times]; got {data.shape}")
    if sampling_hz <= 0:
        raise ValueError(f"sampling_hz must be > 0; got {sampling_hz}")
    n_ch, n_t = data.shape
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_ch)]
    if len(channel_names) != n_ch:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_ch}")
    if n_t < 2:
        raise ValueError("PLV requires at least two time samples")

    preprocessing = ["analytic signal phase via Hilbert transform"]
    working_data = data.astype(np.float64, copy=True)

    if frequency_band is not None:
        lo, hi = frequency_band
        if not (0 < lo < hi < sampling_hz / 2):
            raise ValueError(
                "frequency_band must satisfy 0 < low < high < Nyquist; "
                f"got {frequency_band} at sampling_hz={sampling_hz}"
            )
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, [lo / (sampling_hz / 2), hi / (sampling_hz / 2)], btype="band")
            for i in range(n_ch):
                if np.isfinite(working_data[i]).all():
                    working_data[i] = filtfilt(b, a, working_data[i])
            preprocessing.append(f"bandpass filtered {lo}-{hi} Hz (Butterworth order 4)")
        except ImportError:
            preprocessing.append(
                "frequency_band requested but scipy is unavailable; PLV used unfiltered data"
            )

    win_samples = min(max(int(time_window_s * sampling_hz), 2), n_t)
    segment = working_data[:, :win_samples]
    valid_mask = np.isfinite(segment).all(axis=1)
    phases = np.zeros_like(segment, dtype=np.float64)
    if valid_mask.any():
        phases[valid_mask] = np.angle(_analytic_signal(segment[valid_mask], axis=1))

    plv = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_ch):
        if not valid_mask[i]:
            continue
        plv[i, i] = 1.0
        for j in range(i + 1, n_ch):
            if not valid_mask[j]:
                continue
            value = float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j])))))
            plv[i, j] = plv[j, i] = value

    threshold_rule = None
    if threshold is not None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1] for PLV; got {threshold}")
        plv = np.where(plv >= threshold, plv, 0.0)
        threshold_rule = f"PLV >= {threshold}"

    import datetime
    spec = ConnectivitySpec(
        input_signal_type="EEG",
        node_definition="EEG_channel",
        parcellation_or_channel_set=channel_names,
        time_window_s=time_window_s,
        preprocessing_assumptions=preprocessing,
        connectivity_metric="phase_locking_value",
        frequency_band=frequency_band,
        thresholding_rule=threshold_rule,
        matrix_symmetry="symmetric",
        edge_weight_meaning="phase_locking_value",
    )
    return ConnectivityMatrix(
        matrix=plv,
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

    All metrics operate on |adjacency| (undirected, weighted).  The
    construction summary is propagated so reports always know how the graph
    was built.

    Algorithms
    ----------
    clustering_coefficient : Watts-Strogatz local CC on binary graph, mean over nodes.
    global_efficiency      : Harmonic mean of inverse BFS path lengths (E_glob).
    mean_path_length       : Arithmetic mean of BFS shortest paths (reachable pairs).
    betweenness_centrality : Brandes (2001) normalised betweenness on binary graph.
    community_assignments  : Greedy modularity maximisation (one-pass, per-node merge).
    modularity             : Newman-Girvan Q of final community partition.
    small_world_sigma      : (C/C_rand)/(L/L_rand) vs Erdős–Rényi null model.

    BFS metrics are skipped (set to None) when n > 200 to avoid O(n²) overhead
    without sparse solvers.
    """
    adj = np.abs(conn.matrix).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    n = adj.shape[0]

    binary = (adj > 0).astype(float)
    degrees = binary.sum(axis=1)
    strengths = adj.sum(axis=1)
    n_edges = int(degrees.sum()) // 2
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0
    mean_degree = float(degrees.mean())
    mean_strength = float(strengths.mean())

    # ── Clustering coefficient (Watts-Strogatz, binary) ───────────────────────
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

    # ── BFS-based metrics (capped at n = 200) ────────────────────────────────
    global_efficiency = _global_efficiency_bfs(binary, n)
    mean_path_length = _mean_path_length_bfs(binary, n)
    betweenness_centrality = _betweenness_centrality_brandes(binary, n)

    # ── Connected components (BFS) ────────────────────────────────────────────
    n_components = _count_connected_components(binary, n)

    # ── Greedy modularity + community assignments ─────────────────────────────
    modularity, community_assignments = _modularity_and_communities(adj, degrees, n_edges)

    # ── Small-world σ (Watts-Strogatz vs ER null) ─────────────────────────────
    small_world_sigma = _small_world_sigma(
        n=n,
        density=density,
        clustering_coefficient=clustering_coefficient,
        mean_path_length=mean_path_length,
    )

    warnings: list[str] = []
    confidence = MethodConfidence.HIGH
    if n_components > 1:
        warnings.append(
            f"Graph has {n_components} connected components. "
            "Path-length-based metrics are not meaningful across disconnected components."
        )
        confidence = MethodConfidence.LOW_CONFIDENCE
    if n < 4:
        warnings.append(f"Graph has only {n} nodes; metrics may be unreliable.")
        confidence = MethodConfidence.LOW_CONFIDENCE
    if n > _BFS_NODE_LIMIT:
        warnings.append(
            f"n={n} > {_BFS_NODE_LIMIT}: BFS metrics (efficiency, path length, "
            "betweenness) were skipped. Install sparse-graph extras for large graphs."
        )

    return GraphMetricReport(
        scope=scope,
        n_nodes=n,
        n_edges=n_edges,
        density=density,
        mean_degree=mean_degree,
        mean_strength=mean_strength,
        clustering_coefficient=clustering_coefficient,
        global_efficiency=global_efficiency,
        mean_path_length=mean_path_length,
        modularity=modularity,
        n_connected_components=n_components,
        degree=degrees.tolist(),
        strength=strengths.tolist(),
        betweenness_centrality=betweenness_centrality,
        community_assignments=community_assignments,
        small_world_sigma=small_world_sigma,
        construction_summary=conn.spec.summary(),
        warnings=warnings,
        confidence=confidence,
    )


# ── Graph algorithm implementations ──────────────────────────────────────────

def _global_efficiency_bfs(binary: np.ndarray, n: int) -> float | None:
    """Global efficiency = harmonic mean of inverse BFS shortest path lengths."""
    if n > _BFS_NODE_LIMIT:
        return None
    total_inv_dist = 0.0
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            for v in range(n):
                if binary[u, v] > 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for j in range(n):
            if j != i and dist[j] > 0:
                total_inv_dist += 1.0 / dist[j]
    n_pairs = n * (n - 1)
    return total_inv_dist / n_pairs if n_pairs > 0 else 0.0


def _mean_path_length_bfs(binary: np.ndarray, n: int) -> float | None:
    """Arithmetic mean of all pairwise BFS shortest paths (reachable pairs only)."""
    if n > _BFS_NODE_LIMIT:
        return None
    total_dist = 0
    n_reachable = 0
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue = [i]
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            for v in range(n):
                if binary[u, v] > 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for j in range(n):
            if j != i and dist[j] > 0:
                total_dist += dist[j]
                n_reachable += 1
    return total_dist / n_reachable if n_reachable > 0 else None


def _betweenness_centrality_brandes(binary: np.ndarray, n: int) -> list[float]:
    """Normalised betweenness centrality via Brandes (2001) BFS algorithm.

    BC(v) = sum_{s≠v≠t} (σ_st(v) / σ_st)
    Normalised by (n-1)(n-2)/2 for undirected graphs.
    """
    if n > _BFS_NODE_LIMIT:
        return [0.0] * n

    bc = [0.0] * n
    for s in range(n):
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0

        # BFS to compute shortest-path counts and predecessors
        while head < len(queue):
            v = queue[head]; head += 1
            stack.append(v)
            for w in range(n):
                if binary[v, w] <= 0:
                    continue
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Back-propagation of dependency scores
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                ratio = sigma[v] / sigma[w] if sigma[w] > 0 else 0.0
                delta[v] += ratio * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # Normalise for undirected graph: divide by (n-1)(n-2)/2
    norm = (n - 1) * (n - 2) / 2.0 if n > 2 else 1.0
    return [v / norm for v in bc]


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


def _modularity_and_communities(
    adj: np.ndarray,
    degrees: np.ndarray,
    n_edges: int,
) -> tuple[float | None, list[int]]:
    """Greedy modularity maximisation (one-pass per-node merge).

    Returns (Q, community_labels) where community_labels[i] is the
    integer community index for node i.  Uses Newman-Girvan Q.

    Strategy: initialise each node in its own community; iterate over
    all nodes and merge each into the neighbour community that maximises Q.
    Repeat until no merge improves Q.
    """
    n = adj.shape[0]
    if n_edges == 0:
        return None, list(range(n))

    m2 = 2 * n_edges
    communities = list(range(n))
    best_q = _compute_q(adj, communities, degrees, m2)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            original_comm = communities[i]
            best_local_q = best_q
            best_comm = original_comm
            # Try merging into each neighbour's community
            for j in range(n):
                if i == j or adj[i, j] == 0:
                    continue
                nc = communities[j]
                if nc == original_comm:
                    continue
                communities[i] = nc
                q = _compute_q(adj, communities, degrees, m2)
                if q > best_local_q:
                    best_local_q = q
                    best_comm = nc
                communities[i] = original_comm
            if best_comm != original_comm:
                communities[i] = best_comm
                best_q = best_local_q
                improved = True

    # Re-label communities as consecutive integers
    label_map: dict[int, int] = {}
    next_label = 0
    labels = []
    for c in communities:
        if c not in label_map:
            label_map[c] = next_label
            next_label += 1
        labels.append(label_map[c])

    return round(best_q, 6), labels


def _compute_q(
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


def _small_world_sigma(
    n: int,
    density: float,
    clustering_coefficient: float | None,
    mean_path_length: float | None,
) -> float | None:
    """Small-world coefficient σ = (C / C_rand) / (L / L_rand).

    Null model: Erdős–Rényi random graph with the same n and edge density p.

    Approximations for ER null model:
      C_rand ≈ p = density          (CC of random graph ≈ edge probability)
      L_rand ≈ ln(n) / ln(k_mean)  (mean path length of connected ER graph)

    σ > 1 indicates small-world topology.
    σ is set to None when:
      - n < 4 or density ≤ 0 (degenerate graph)
      - mean_path_length or clustering_coefficient are None or zero
      - mean degree k < 2 (ER graph is likely disconnected)
    """
    if clustering_coefficient is None or mean_path_length is None:
        return None
    if mean_path_length == 0 or n < 4 or density <= 0:
        return None

    k_mean = density * (n - 1)  # mean degree in ER null model
    if k_mean < 2:
        return None

    c_rand = density                   # ≈ edge probability p
    try:
        l_rand = math.log(n) / math.log(k_mean)
    except (ValueError, ZeroDivisionError):
        return None

    if l_rand <= 0 or c_rand <= 0:
        return None

    gamma = clustering_coefficient / c_rand   # normalised clustering
    lam = mean_path_length / l_rand           # normalised path length
    if lam == 0:
        return None

    return round(gamma / lam, 4)


def _analytic_signal(data: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Return the analytic signal using scipy when available, otherwise FFT.

    The fallback mirrors the standard Hilbert-transform construction used by
    scipy.signal.hilbert: double positive-frequency bins, retain DC and Nyquist
    bins, and zero negative frequencies.
    """
    try:
        from scipy.signal import hilbert
        return hilbert(data, axis=axis)
    except ImportError:
        pass

    x = np.asarray(data)
    n = x.shape[axis]
    spectrum = np.fft.fft(x, axis=axis)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(spectrum * h.reshape(shape), axis=axis)
