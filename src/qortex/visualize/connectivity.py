"""Connectivity heatmap + network-graph figure for ConnectivityMatrix/GraphMetricReport.

neuroclassic.connectivity already computes real Pearson/PLV adjacency
matrices and graph-theoretic metrics (clustering, efficiency, modularity,
hub degree) but had no plotting counterpart — this module is the rendering
layer for that existing, already-correct numeric output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qortex.neuroclassic.connectivity import ConnectivityMatrix, GraphMetricReport


def _circular_layout(n: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def connectivity_figure(
    matrix: "ConnectivityMatrix",
    metrics: "GraphMetricReport | None" = None,
    *,
    threshold: float = 0.25,
    title: str | None = None,
):
    """3-panel figure: ROI-ROI heatmap, thresholded network graph, summary panel.

    Parameters
    ----------
    matrix:
        A computed ``ConnectivityMatrix`` (e.g. from
        ``compute_pearson_connectivity``). Never recomputed here — this
        function only renders values it is given.
    metrics:
        Optional ``GraphMetricReport`` (e.g. from ``compute_graph_metrics``)
        to populate the summary panel and hub-region table. When omitted,
        the summary panel shows only matrix-derived quantities.
    threshold:
        Minimum absolute edge weight to draw in the network graph.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "connectivity_figure() requires matplotlib and seaborn: "
            "pip install matplotlib seaborn"
        ) from exc

    m = np.asarray(matrix.matrix, dtype=np.float64)
    labels = matrix.node_labels
    n = m.shape[0]

    fig = plt.figure(figsize=(12.5, 5.4), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.35, figure=fig)

    # ── Panel 1: ROI-ROI heatmap ───────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[0])
    sns.heatmap(
        m, xticklabels=labels, yticklabels=labels, cmap="RdBu_r",
        vmin=-1.0, vmax=1.0, square=True, ax=ax_hm,
        cbar_kws={"label": matrix.spec.edge_weight_meaning.replace("_", " "), "shrink": 0.75},
    )
    ax_hm.set_title("ROI-ROI connectivity", fontsize=11, fontweight="bold", loc="left")
    ax_hm.tick_params(axis="both", labelsize=7, rotation=90 if n > 12 else 0)
    ax_hm.tick_params(axis="y", rotation=0)

    # ── Panel 2: thresholded network graph (circular layout) ─────────────
    ax_net = fig.add_subplot(gs[1])
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    pos = _circular_layout(n)

    off_diag = m[~np.eye(n, dtype=bool)]
    max_abs = float(np.max(np.abs(off_diag))) if off_diag.size else 1.0
    max_abs = max_abs or 1.0

    for i in range(n):
        for j in range(i + 1, n):
            w = m[i, j]
            if abs(w) < threshold:
                continue
            color = "#e03131" if w > 0 else "#1971c2"
            ax_net.plot(
                [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=color, alpha=min(1.0, 0.25 + 0.65 * abs(w) / max_abs),
                linewidth=0.5 + 2.5 * abs(w) / max_abs, zorder=1,
            )

    degree = np.sum(np.abs(m) >= threshold, axis=1) - (np.abs(np.diag(m)) >= threshold)
    node_size = 220 + 60 * degree
    ax_net.scatter(pos[:, 0], pos[:, 1], s=node_size, c="#495057", zorder=2, edgecolors="white", linewidth=1.2)
    for i, label in enumerate(labels):
        r = 1.18
        ax_net.text(pos[i, 0] * r, pos[i, 1] * r, label, fontsize=8, ha="center", va="center", fontweight="bold")
    ax_net.set_xlim(-1.5, 1.5)
    ax_net.set_ylim(-1.5, 1.5)
    ax_net.set_title(f"Network connectome (|edge| ≥ {threshold:.2f})", fontsize=11, fontweight="bold", loc="left")

    # ── Panel 3: summary + hub table ───────────────────────────────────────
    ax_sum = fig.add_subplot(gs[2])
    ax_sum.axis("off")

    mean_conn = float(np.mean(off_diag)) if off_diag.size else 0.0
    pct_pos = float(np.mean(off_diag > 0) * 100) if off_diag.size else 0.0

    lines: list[tuple[str, str]] = [
        ("Mean connectivity", f"{mean_conn:.2f}"),
        ("Positive edges", f"{pct_pos:.0f}%"),
    ]
    if metrics is not None:
        lines += [
            ("Network modularity (Q)", f"{metrics.modularity:.2f}" if metrics.modularity is not None else "n/a"),
            ("Mean path length", f"{metrics.mean_path_length:.2f}" if metrics.mean_path_length is not None else "n/a"),
            ("Global efficiency", f"{metrics.global_efficiency:.2f}" if metrics.global_efficiency is not None else "n/a"),
            ("Small-worldness (σ)", f"{metrics.small_world_sigma:.2f}" if metrics.small_world_sigma is not None else "n/a"),
        ]

    y = 0.97
    ax_sum.text(0.0, y, "Global summary", fontsize=10, fontweight="bold", transform=ax_sum.transAxes)
    y -= 0.09
    for label, value in lines:
        ax_sum.text(0.0, y, label, fontsize=8.5, color="#495057", transform=ax_sum.transAxes)
        ax_sum.text(1.0, y, value, fontsize=8.5, fontweight="bold", ha="right", transform=ax_sum.transAxes)
        y -= 0.075

    y -= 0.05
    ax_sum.text(0.0, y, "Hub regions (degree)", fontsize=10, fontweight="bold", transform=ax_sum.transAxes)
    y -= 0.09
    hub_degree = metrics.degree if metrics is not None and metrics.degree else degree.tolist()
    ranked = sorted(zip(labels, hub_degree), key=lambda kv: kv[1], reverse=True)[:5]
    for rank, (label, deg) in enumerate(ranked, start=1):
        ax_sum.text(0.0, y, f"{rank}. {label}", fontsize=8.5, color="#495057", transform=ax_sum.transAxes)
        ax_sum.text(1.0, y, f"{deg:.2f}", fontsize=8.5, fontweight="bold", ha="right", transform=ax_sum.transAxes)
        y -= 0.075

    header = title or f"Connectivity analysis — {matrix.spec.connectivity_metric} ({matrix.spec.node_definition})"
    fig.suptitle(header, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(top=0.86, left=0.06, right=0.97, bottom=0.08)
    return fig


__all__ = ["connectivity_figure"]
