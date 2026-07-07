"""Qortex report design system.

One font stack, one color language, one set of chart components, used by
every matplotlib-based figure builder in this package (dataset_readiness,
participants, connectivity, reproducibility, psd, and the NeuroAI showcase
boards) so they read as one product.

    from qortex.visualize.design import apply_theme, figure_title, metric_card, STATUS

Layout
------
theme.py       — matplotlib rcParams (call ``apply_theme()`` once per figure)
palettes.py    — INK/SUBINK/BORDER/CARD_BG grays, STATUS, CATEGORICAL
typography.py  — FONT_STACK + named type scale (SIZE_TITLE, SIZE_SECTION, ...)
components.py  — figure_title, section_title, metric_card, status_badge, style_table
"""

from __future__ import annotations

from qortex.visualize.design.components import (
    figure_title,
    metric_card,
    section_title,
    status_badge,
    style_table,
)
from qortex.visualize.design.palettes import (
    BORDER,
    CARD_BG,
    CATEGORICAL,
    FAINT,
    GRID,
    INK,
    NEUROAI_CLASS_COLORS,
    PANEL_BG,
    STATUS,
    SUBINK,
)
from qortex.visualize.design.theme import apply_theme
from qortex.visualize.design.typography import (
    FONT_STACK,
    SIZE_BODY,
    SIZE_METRIC_LABEL,
    SIZE_METRIC_VALUE,
    SIZE_SECTION,
    SIZE_SMALL,
    SIZE_SUBTITLE,
    SIZE_TITLE,
)

__all__ = [
    "apply_theme",
    "figure_title", "section_title", "metric_card", "status_badge", "style_table",
    "INK", "SUBINK", "FAINT", "BORDER", "CARD_BG", "PANEL_BG", "GRID",
    "STATUS", "CATEGORICAL", "NEUROAI_CLASS_COLORS",
    "FONT_STACK", "SIZE_TITLE", "SIZE_SUBTITLE", "SIZE_SECTION", "SIZE_BODY", "SIZE_SMALL",
    "SIZE_METRIC_VALUE", "SIZE_METRIC_LABEL",
]
