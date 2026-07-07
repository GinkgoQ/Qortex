"""Matplotlib rcParams setup for the Qortex report design system."""

from __future__ import annotations

from qortex.visualize.design.palettes import BORDER, GRID, INK, PANEL_BG, SUBINK
from qortex.visualize.design.typography import FONT_STACK, SIZE_BODY, SIZE_SECTION


def apply_theme() -> None:
    """Configure matplotlib rcParams. Idempotent — safe to call from every figure builder."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": SIZE_BODY,
        "text.color": INK,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": SIZE_SECTION,
        "axes.labelsize": SIZE_BODY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": SUBINK,
        "ytick.color": SUBINK,
        "xtick.labelsize": SIZE_BODY - 1,
        "ytick.labelsize": SIZE_BODY - 1,
        "figure.facecolor": PANEL_BG,
        "savefig.facecolor": PANEL_BG,
        "savefig.dpi": 200,
        "legend.frameon": False,
        "legend.fontsize": SIZE_BODY - 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


__all__ = ["apply_theme"]
