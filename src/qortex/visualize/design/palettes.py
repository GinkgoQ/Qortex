"""Color tokens shared by every Qortex report figure.

Semantic tokens (status colors, ink/border/card grays) plus one curated
categorical palette — not the seaborn/matplotlib tab10 default, and not a
different ad hoc palette per figure file.
"""

from __future__ import annotations

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

# Distinct hues at matched lightness/chroma for multi-series plots.
CATEGORICAL = ["#4f46e5", "#0891b2", "#d97706", "#dc2626", "#7c3aed", "#059669", "#db2777", "#64748b"]

# NeuroAI class-color convention (detection/segmentation legends) — reuses
# the same categorical scale so a class keeps one color across figure kinds.
NEUROAI_CLASS_COLORS = CATEGORICAL

__all__ = [
    "INK", "SUBINK", "FAINT", "BORDER", "CARD_BG", "PANEL_BG", "GRID",
    "STATUS", "CATEGORICAL", "NEUROAI_CLASS_COLORS",
]
