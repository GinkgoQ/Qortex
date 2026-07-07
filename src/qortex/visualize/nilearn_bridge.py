"""Nilearn-backed glass-brain, connectome, and stat-map figures.

Qortex's own ``connectivity.py`` renders a heatmap + schematic circular
graph — useful when nodes have no anatomical coordinates. When real MNI
coordinates are available, this module hands the rendering to Nilearn's
domain-standard plotting instead of re-implementing glass-brain projection
or MNI-space statistical overlays by hand.

Node coordinates are never invented: callers must supply real MNI mm
coordinates (e.g. from an atlas' published centroids). No coordinate
fabrication happens here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qortex.neuroclassic.connectivity import ConnectivityMatrix


def _require_nilearn():
    try:
        import nilearn  # noqa: F401
        return nilearn
    except ImportError as exc:
        raise ImportError(
            "This figure requires nilearn: pip install nilearn"
        ) from exc


def glass_brain_connectome_figure(
    matrix: "ConnectivityMatrix",
    node_coords: np.ndarray,
    *,
    edge_threshold: str | float = "80%",
    title: str | None = None,
    node_color: str = "#4f46e5",
):
    """Glass-brain connectome using Nilearn's ``plot_connectome``.

    Parameters
    ----------
    matrix:
        A computed ``ConnectivityMatrix`` (e.g. from
        ``compute_pearson_connectivity``) — rendered as given, not recomputed.
    node_coords:
        Real MNI mm coordinates, shape ``(n_nodes, 3)``, one row per
        ``matrix.node_labels`` entry in the same order. Never invented here
        — raises if missing or mismatched.
    edge_threshold:
        Forwarded to Nilearn: a percentile string (``"80%"``) or an absolute
        edge-weight cutoff.
    """
    _require_nilearn()
    from nilearn import plotting

    adjacency = np.asarray(matrix.matrix, dtype=np.float64)
    coords = np.asarray(node_coords, dtype=np.float64)
    if coords.shape != (adjacency.shape[0], 3):
        raise ValueError(
            f"node_coords must have shape {(adjacency.shape[0], 3)} to match "
            f"{adjacency.shape[0]} nodes in the connectivity matrix, got {coords.shape}"
        )

    header = title or f"Glass-brain connectome — {matrix.spec.connectivity_metric} ({matrix.spec.node_definition})"
    display = plotting.plot_connectome(
        adjacency, coords, edge_threshold=edge_threshold, node_color=node_color,
        title=header, colorbar=True, display_mode="ortho",
    )
    return display


def stat_map_figure(
    stat_img: Any,
    *,
    bg_img: Any = None,
    threshold: float = 2.3,
    title: str | None = None,
    display_mode: str = "ortho",
    cut_coords: Any = None,
):
    """Thresholded statistical map overlay using Nilearn's ``plot_stat_map``.

    Parameters
    ----------
    stat_img:
        Path or nibabel image with the statistic values (e.g. a z-map).
        Never synthesized here — the caller provides a real image.
    bg_img:
        Optional anatomical background (path or nibabel image). Defaults to
        Nilearn's bundled MNI152 template when omitted.
    threshold:
        Minimum |value| to display.
    """
    _require_nilearn()
    from nilearn import plotting

    header = title or "Statistical map"
    display = plotting.plot_stat_map(
        stat_img, bg_img=bg_img, threshold=threshold, title=header,
        display_mode=display_mode, cut_coords=cut_coords, colorbar=True,
    )
    return display


def glass_brain_stat_figure(
    stat_img: Any,
    *,
    threshold: float = 2.3,
    title: str | None = None,
):
    """Glass-brain projection of a statistical map via ``plot_glass_brain``."""
    _require_nilearn()
    from nilearn import plotting

    header = title or "Glass brain"
    display = plotting.plot_glass_brain(stat_img, threshold=threshold, title=header, colorbar=True)
    return display


__all__ = ["glass_brain_connectome_figure", "stat_map_figure", "glass_brain_stat_figure"]
