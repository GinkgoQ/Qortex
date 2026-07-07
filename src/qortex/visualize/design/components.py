"""Reusable chart components: figure titles, section headers, metric cards,
status badges, and table styling — shared across every Qortex report figure.
"""

from __future__ import annotations

from typing import Any

from qortex.visualize.design.palettes import BORDER, CARD_BG, INK, PANEL_BG, SUBINK
from qortex.visualize.design.typography import SIZE_METRIC_LABEL, SIZE_METRIC_VALUE, SIZE_SECTION, SIZE_SUBTITLE, SIZE_TITLE


def figure_title(fig: Any, text: str, *, subtitle: str | None = None, x: float = 0.025) -> None:
    """Two-tier figure title: bold headline + muted subtitle, reserved vertical space so they never collide."""
    fig.text(x, 0.975, text, fontsize=SIZE_TITLE, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(x, 0.975 - 0.035, subtitle, fontsize=SIZE_SUBTITLE, color=SUBINK, ha="left", va="top")


def section_title(ax: Any, text: str, *, y: float = 1.04) -> None:
    """Panel-level section heading, sitting just above the axes (not eating into plot area)."""
    ax.text(0.0, y, text, fontsize=SIZE_SECTION, fontweight="bold", color=INK, transform=ax.transAxes, va="bottom")


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
    ax.text(0.13, 0.58, value, fontsize=SIZE_METRIC_VALUE, fontweight="bold", color=color,
            transform=ax.transAxes, va="center")
    ax.text(0.13, 0.26, label, fontsize=SIZE_METRIC_LABEL, color=SUBINK, transform=ax.transAxes, va="center")


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


__all__ = ["figure_title", "section_title", "metric_card", "status_badge", "style_table"]
