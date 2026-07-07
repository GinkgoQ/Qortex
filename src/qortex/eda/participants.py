"""participants.tsv parsing and metadata visualization.

Parses the BIDS ``participants.tsv`` table (+ optional ``participants.json``
sidecar) and detects dirty categorical values (extra whitespace, trailing
punctuation, casing) before any statistic is computed from them — a value
like ``"M,"`` is never silently plotted as a valid group.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_MISSING_TOKENS = {"", "n/a", "na", "nan", "unknown"}


def _clean_token(raw: str) -> str:
    return raw.strip().strip(",;|").strip()


def _is_missing(raw: str | None) -> bool:
    return raw is None or raw.strip().lower() in _MISSING_TOKENS


def _float_or_none(raw: Any) -> float | None:
    if raw is None or _is_missing(str(raw)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class ParticipantRecord:
    participant_id: str
    values: dict[str, str]


@dataclass
class ParticipantsTable:
    columns: list[str]
    records: list[ParticipantRecord]
    sidecar: dict[str, Any] = field(default_factory=dict)

    @property
    def n_participants(self) -> int:
        return len(self.records)


def parse_participants_tsv(
    path: Path | str,
    sidecar_path: Path | str | None = None,
) -> ParticipantsTable:
    """Parse participants.tsv and its participants.json sidecar, if present."""
    tsv_path = Path(path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv_path}")

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = list(reader.fieldnames or [])
        records = [
            ParticipantRecord(
                participant_id=row.get("participant_id", ""),
                values={k: v for k, v in row.items() if k != "participant_id"},
            )
            for row in reader
        ]

    sc_path = Path(sidecar_path) if sidecar_path is not None else tsv_path.with_suffix(".json")
    sidecar: dict[str, Any] = {}
    if sc_path.exists():
        try:
            sidecar = json.loads(sc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sidecar = {}

    return ParticipantsTable(columns=columns, records=records, sidecar=sidecar)


@dataclass
class CategoricalSummary:
    column: str
    valid_counts: dict[str, int]
    invalid_values: dict[str, list[int]]  # raw value -> row indices
    n_missing: int

    @property
    def n_valid(self) -> int:
        return sum(self.valid_counts.values())

    @property
    def n_invalid(self) -> int:
        return sum(len(rows) for rows in self.invalid_values.values())


def summarize_categorical(table: ParticipantsTable, column: str) -> CategoricalSummary:
    """Detect valid categories vs. dirty/unrecognized values in one column.

    Ground truth for "valid" comes from the participants.json sidecar's
    ``Levels`` mapping when available (BIDS convention); otherwise it is
    inferred as the set of values that need no cleaning (already trimmed,
    no trailing punctuation) — anything that only matches after cleaning,
    or matches nothing, is reported as invalid rather than folded in.
    """
    col_meta = table.sidecar.get(column) if isinstance(table.sidecar, dict) else None
    sidecar_levels: set[str] | None = None
    if isinstance(col_meta, dict) and isinstance(col_meta.get("Levels"), dict):
        sidecar_levels = set(col_meta["Levels"].keys())

    raw_values = [rec.values.get(column, "") for rec in table.records]
    n_missing = sum(1 for v in raw_values if _is_missing(v))

    if sidecar_levels is not None:
        known = sidecar_levels
    else:
        known = {
            v for v in raw_values
            if not _is_missing(v) and v == _clean_token(v)
        }

    valid_counts: dict[str, int] = {}
    invalid_values: dict[str, list[int]] = {}
    for i, raw in enumerate(raw_values):
        if _is_missing(raw):
            continue
        # A value only counts as valid if it is *already* clean and matches a
        # known category exactly — cleaning a dirty value (e.g. "M,") and
        # merging it into the matching group would silently hide the defect
        # this check exists to surface.
        if raw == _clean_token(raw) and raw in known:
            valid_counts[raw] = valid_counts.get(raw, 0) + 1
        else:
            invalid_values.setdefault(raw, []).append(i)

    return CategoricalSummary(
        column=column, valid_counts=valid_counts,
        invalid_values=invalid_values, n_missing=n_missing,
    )


@dataclass
class GroupStats:
    group: str
    n: int
    median: float
    q1: float
    q3: float
    vmin: float
    vmax: float


def numeric_by_group(
    table: ParticipantsTable,
    value_col: str,
    group_summary: CategoricalSummary,
) -> list[GroupStats]:
    """Per-group median/IQR/range for a numeric column, including an
    "Invalid" pseudo-group for rows whose categorical value was rejected."""
    by_group: dict[str, list[float]] = {g: [] for g in group_summary.valid_counts}
    invalid_rows = {i for rows in group_summary.invalid_values.values() for i in rows}
    invalid_values: list[float] = []

    for i, rec in enumerate(table.records):
        val = _float_or_none(rec.values.get(value_col))
        if val is None:
            continue
        raw_group = rec.values.get(group_summary.column, "")
        clean_group = _clean_token(raw_group)
        if i in invalid_rows:
            invalid_values.append(val)
        elif clean_group in by_group:
            by_group[clean_group].append(val)

    stats = []
    for g, vals in by_group.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        stats.append(GroupStats(
            group=g, n=len(arr),
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)), q3=float(np.percentile(arr, 75)),
            vmin=float(arr.min()), vmax=float(arr.max()),
        ))
    if invalid_values:
        arr = np.asarray(invalid_values, dtype=np.float64)
        stats.append(GroupStats(
            group="Invalid", n=len(arr),
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)), q3=float(np.percentile(arr, 75)),
            vmin=float(arr.min()), vmax=float(arr.max()),
        ))
    return stats


def participants_metadata_figure(
    table: ParticipantsTable,
    *,
    group_col: str = "sex",
    value_col: str = "age",
    dataset_id: str = "",
    title: str | None = None,
):
    """Card-style age/sex distribution figure with dirty-value detection.

    Layout: metric cards (total / valid / invalid / missing) → violin + box
    + strip plot per valid group, with invalid-group rows shown separately
    as an "Invalid" scatter column → per-group summary table → an explicit
    invalid-value warning banner (never silently absorbed into a group).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "participants_metadata_figure() requires matplotlib and seaborn: "
            "pip install matplotlib seaborn"
        ) from exc

    group_summary = summarize_categorical(table, group_col)
    stats = numeric_by_group(table, value_col, group_summary)
    valid_order = [s.group for s in stats if s.group != "Invalid"]
    has_invalid = any(s.group == "Invalid" for s in stats)

    groups_valid: list[str] = []
    values_valid: list[float] = []
    invalid_values: list[float] = []
    invalid_rows = {i for rows in group_summary.invalid_values.values() for i in rows}
    for i, rec in enumerate(table.records):
        val = _float_or_none(rec.values.get(value_col))
        if val is None:
            continue
        if i in invalid_rows:
            invalid_values.append(val)
        else:
            clean_group = _clean_token(rec.values.get(group_col, ""))
            if clean_group in group_summary.valid_counts:
                groups_valid.append(clean_group)
                values_valid.append(val)

    n_total = table.n_participants
    n_valid = group_summary.n_valid
    n_invalid = group_summary.n_invalid
    n_missing = group_summary.n_missing

    sns.set_theme(style="whitegrid", font_scale=0.85)
    fig = plt.figure(figsize=(11.5, 6.4), dpi=150)
    gs = gridspec.GridSpec(
        3, 4, height_ratios=[0.5, 2.6, 0.55], hspace=0.65, wspace=0.4, figure=fig,
    )

    def _pct(n: int) -> str:
        return f"{n / n_total * 100:.0f}%" if n_total else "0%"

    cards = [
        ("Total participants", str(n_total), "#343a40"),
        ("Valid entries", f"{n_valid} ({_pct(n_valid)})", "#2f9e44"),
        ("Invalid entries", f"{n_invalid} ({_pct(n_invalid)})", "#e8590c" if n_invalid else "#868e96"),
        ("Missing", str(n_missing), "#868e96"),
    ]
    for i, (label, value, color) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle(
            (0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
            facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=1, zorder=0,
        ))
        ax.text(0.12, 0.62, value, fontsize=16, fontweight="bold", color=color,
                transform=ax.transAxes, va="center")
        ax.text(0.12, 0.24, label, fontsize=8, color="#495057", transform=ax.transAxes, va="center")

    ax_main = fig.add_subplot(gs[1, :3])
    palette = sns.color_palette("muted", n_colors=max(1, len(valid_order)))
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(valid_order)}

    if values_valid:
        sns.violinplot(
            x=groups_valid, y=values_valid, order=valid_order,
            hue=groups_valid, hue_order=valid_order, legend=False,
            palette=color_map, inner=None, cut=0, ax=ax_main, linewidth=1,
        )
        sns.boxplot(
            x=groups_valid, y=values_valid, order=valid_order,
            width=0.16, showcaps=True, ax=ax_main,
            boxprops={"facecolor": "white", "zorder": 3},
            whiskerprops={"zorder": 3}, medianprops={"zorder": 3},
        )
        sns.stripplot(
            x=groups_valid, y=values_valid, order=valid_order,
            color="black", alpha=0.35, size=3, jitter=0.12, ax=ax_main,
        )

    x_labels = list(valid_order)
    if has_invalid and invalid_values:
        x_invalid = len(valid_order)
        ax_main.scatter(
            [x_invalid] * len(invalid_values), invalid_values,
            color="#e8590c", marker="x", s=45, linewidth=1.5, zorder=4, label="Invalid",
        )
        x_labels = x_labels + ["Invalid"]
        ax_main.set_xlim(-0.6, x_invalid + 0.6)
        ax_main.set_xticks(range(len(x_labels)))
        ax_main.set_xticklabels(x_labels)

    ax_main.set_xlabel("")
    ax_main.set_ylabel(value_col.replace("_", " ").title())
    ax_main.set_title(
        f"{value_col.replace('_', ' ').title()} distribution by {group_col}",
        fontsize=11, fontweight="bold", loc="left",
    )
    sns.despine(ax=ax_main)

    ax_table = fig.add_subplot(gs[1, 3])
    ax_table.axis("off")
    rows = [
        [s.group, str(s.n), f"{s.median:.0f}", f"[{s.q1:.0f}-{s.q3:.0f}]", f"{s.vmin:.0f}-{s.vmax:.0f}"]
        for s in stats
    ]
    if rows:
        tbl = ax_table.table(
            cellText=rows, colLabels=["Group", "n", "Median", "IQR", "Range"],
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dee2e6")
            if r == 0:
                cell.set_facecolor("#f1f3f5")
                cell.set_text_props(fontweight="bold")

    ax_warn = fig.add_subplot(gs[2, :])
    ax_warn.axis("off")
    if group_summary.invalid_values:
        examples = ", ".join(repr(v) for v in list(group_summary.invalid_values)[:4])
        msg = (
            f"{n_invalid} invalid '{group_col}' value(s) detected and excluded from "
            f"group statistics — shown as a separate 'Invalid' group: {examples}"
        )
        ax_warn.add_patch(plt.Rectangle(
            (0.0, 0.1), 1.0, 0.8, transform=ax_warn.transAxes,
            facecolor="#fff3bf", edgecolor="#f08c00", linewidth=1,
        ))
        ax_warn.text(0.015, 0.5, "⚠ " + msg, fontsize=8.5, color="#5c3c00",
                     va="center", transform=ax_warn.transAxes)
    else:
        ax_warn.text(0.015, 0.5, f"No invalid '{group_col}' values detected.",
                     fontsize=8.5, color="#495057", va="center", transform=ax_warn.transAxes)

    header = title or (f"Participant metadata — {dataset_id}" if dataset_id else "Participant metadata")
    fig.suptitle(header, fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", va="top")
    fig.text(
        0.02, 0.945,
        f"{value_col.title()}s extracted from participants.tsv (BIDS)",
        fontsize=8.5, color="#868e96", va="top",
    )
    fig.subplots_adjust(top=0.84, left=0.045, right=0.98, bottom=0.06)
    return fig


__all__ = [
    "ParticipantRecord",
    "ParticipantsTable",
    "CategoricalSummary",
    "GroupStats",
    "parse_participants_tsv",
    "summarize_categorical",
    "numeric_by_group",
    "participants_metadata_figure",
]
