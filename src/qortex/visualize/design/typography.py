"""Font stack and type scale shared by every Qortex report figure.

One named scale (not ad hoc `fontsize=` literals scattered per figure) so
title/section/body/caption sizes stay consistent across dataset-readiness,
participant-metadata, connectivity, reproducibility, PSD, and NeuroAI
showcase boards.
"""

from __future__ import annotations

# Preference order checked against installed system fonts (font_manager),
# not assumed — falls back to matplotlib's DejaVu Sans only if none exist.
FONT_STACK = ["Lato", "Roboto", "Open Sans", "Noto Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

# Named type scale (pt). Figures should reference these, not literals.
SIZE_TITLE = 16.0        # figure-level headline
SIZE_SUBTITLE = 9.5      # figure-level subtitle/caption
SIZE_SECTION = 10.5      # panel section heading
SIZE_BODY = 9.0          # tick labels, axis labels, table body
SIZE_SMALL = 8.5         # legend, secondary annotations
SIZE_METRIC_VALUE = 19.0  # metric-card big number
SIZE_METRIC_LABEL = 8.5   # metric-card caption

__all__ = [
    "FONT_STACK",
    "SIZE_TITLE", "SIZE_SUBTITLE", "SIZE_SECTION", "SIZE_BODY", "SIZE_SMALL",
    "SIZE_METRIC_VALUE", "SIZE_METRIC_LABEL",
]
