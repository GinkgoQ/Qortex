"""Shared matplotlib design system for Qortex report figures.

One font stack, one color language, one set of card/table/badge components —
reused by every matplotlib-based figure builder (dataset_readiness,
participants, connectivity, reproducibility, psd, and the NeuroAI showcase
boards) so they read as one product instead of each hand-rolling its own
colors and falling back to default DejaVu Sans.
"""

from __future__ import annotations

from typing import Any

# Prefer humanist sans-serifs actually present on this system over the
# matplotlib default (DejaVu Sans) — checked via font_manager, not assumed.
FONT_STACK = ["Lato", "Roboto", "Open Sans", "Noto Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

INK = "#111827"      # primary text (gray-900)
SUBINK = "#6b7280"   # secondary text (gray-500)
FAINT = "#9ca3af"    # tertiary / captions (gray-400)
BORDER = "#e5e7eb"   # gray-200
CARD_BG = "#f9fafb"  # gray-50
PANEL_BG = "#ffffff"
GRID = "#eef1f4"

STATUS = {
    "success": "#0f9960",
    "warning": "#d97706",
    "danger": "#dc2626",
    "neutral": "#6b7280",
    "info": "#2563eb",
}

# Curated categorical palette — distinct hues at matched lightness/chroma,
# not the seaborn/matplotlib tab10 default.
CATEGORICAL = ["#4f46e5", "#0891b2", "#d97706", "#dc2626", "#7c3aed", "#059669", "#db2777", "#64748b"]

_THEME_APPLIED = False


def apply_theme() -> None:
    """Configure matplotlib rcParams. Idempotent — safe to call from every figure builder."""
    global _THEME_APPLIED
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 9,
        "text.color": INK,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": SUBINK,
        "ytick.color": SUBINK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": PANEL_BG,
        "savefig.facecolor": PANEL_BG,
        "savefig.dpi": 200,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    _THEME_APPLIED = True


def figure_title(fig: Any, text: str, *, subtitle: str | None = None, x: float = 0.025) -> None:
    """Two-tier figure title: bold headline + muted subtitle, reserved vertical space so they never collide."""
    fig.text(x, 0.975, text, fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(x, 0.975 - 0.035, subtitle, fontsize=9.5, color=SUBINK, ha="left", va="top")


def section_title(ax: Any, text: str, *, y: float = 1.04) -> None:
    """Panel-level section heading, sitting just above the axes (not eating into plot area)."""
    ax.text(0.0, y, text, fontsize=10.5, fontweight="bold", color=INK, transform=ax.transAxes, va="bottom")


def metric_card(ax: Any, *, value: str, label: str, color: str = INK, accent: str | None = None) -> None:
    """One metric card: big value + small label, optional left accent bar, rounded card background."""
    from matplotlib.patches import FancyBboxPatch

    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0.0,rounding_size=0.06",
        transform=ax.transAxes, facecolor=CARD_BG, edgecolor=BORDER, linewidth=1.1, clip_on=False,
    ))
    if accent:
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.08), 0.03, 0.84, boxstyle="round,pad=0.0,rounding_size=0.015",
            transform=ax.transAxes, facecolor=accent, edgecolor="none", clip_on=False,
        ))
    ax.text(0.13, 0.58, value, fontsize=19, fontweight="bold", color=color, transform=ax.transAxes, va="center")
    ax.text(0.13, 0.26, label, fontsize=8.5, color=SUBINK, transform=ax.transAxes, va="center")


def status_badge(ax: Any, *, text: str, color: str, x: float = 0.0, y: float = 0.5, fontsize: float = 10.0) -> None:
    """Pill-style status badge (soft fill + matching border), not a plain colored rectangle."""
    ax.text(
        x, y, text, fontsize=fontsize, fontweight="bold", color=color,
        transform=ax.transAxes, va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.1),
    )


def style_table(tbl: Any, *, header_bg: str = CARD_BG, zebra: bool = True, fontsize: float = 8.5) -> None:
    """Clean data-table style: no harsh vertical rules, subtle zebra striping, bold header row."""
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.6)
        cell.PAD = 0.045
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.set_text_props(color=INK)
            cell.set_facecolor("#fbfbfd" if (zebra and r % 2 == 0) else PANEL_BG)


__all__ = [
    "FONT_STACK", "INK", "SUBINK", "FAINT", "BORDER", "CARD_BG", "PANEL_BG", "GRID",
    "STATUS", "CATEGORICAL",
    "apply_theme", "figure_title", "section_title", "metric_card", "status_badge", "style_table",
]
