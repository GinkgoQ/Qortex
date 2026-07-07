"""Dataset-readiness report figure.

Renders the score/counts/findings already computed by
``qortex.check.readiness.compute_readiness`` as a report-card figure —
metric cards, a trainability bar chart, and explicit blocker/next-step
panels — instead of a bare score number or a raw ``ReadinessReport.summary()``
text dump.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qortex.core.entities import ReadinessReport

_STATUS_COLORS = {
    "ready": "#2f9e44",
    "uncertain": "#e8590c",
    "blocked": "#c92a2a",
}


def _status_for(report: "ReadinessReport") -> str:
    if report.n_recordings == 0 or report.n_label_ready == 0:
        return "blocked"
    if report.score >= 80:
        return "ready"
    if report.score >= 50:
        return "uncertain"
    return "blocked"


def dataset_readiness_figure(
    report: "ReadinessReport",
    *,
    target: str | None = None,
    estimated_download_mb: float | None = None,
    title: str | None = None,
):
    """Report-card figure for a ``ReadinessReport``.

    Parameters
    ----------
    report:
        Output of ``qortex.check.readiness.compute_readiness``.
    target:
        Label-policy target column name, if one was used, shown verbatim.
    estimated_download_mb:
        Download size to display, if known (``report`` does not carry it).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        raise ImportError(
            "dataset_readiness_figure() requires matplotlib: pip install matplotlib"
        ) from exc

    status = _status_for(report)
    status_color = _STATUS_COLORS[status]

    errors = [f for f in report.findings if f.severity == "error"]
    warnings_ = [f for f in report.findings if f.severity == "warning"]
    recommendations = []
    for f in report.findings:
        if f.recommendation and f.recommendation not in recommendations:
            recommendations.append(f.recommendation)

    fig = plt.figure(figsize=(11.5, 6.8), dpi=150)
    gs = gridspec.GridSpec(4, 4, height_ratios=[0.75, 0.5, 1.1, 1.0], hspace=0.55, wspace=0.4, figure=fig)

    # ── Header ──────────────────────────────────────────────────────────────
    ax_head = fig.add_subplot(gs[0, :2])
    ax_head.axis("off")
    ax_head.text(0.0, 0.82, report.dataset_id, fontsize=17, fontweight="bold", color="#20242c", transform=ax_head.transAxes)
    ax_head.text(0.0, 0.5, f"snapshot: {report.snapshot}", fontsize=9, color="#868e96", transform=ax_head.transAxes)
    ax_head.add_patch(plt.Rectangle((0.0, 0.05), 0.42, 0.24, transform=ax_head.transAxes,
                                     facecolor=status_color, alpha=0.15, edgecolor=status_color))
    ax_head.text(0.02, 0.17, f"status: {status}", fontsize=10, fontweight="bold", color=status_color, transform=ax_head.transAxes)

    ax_info = fig.add_subplot(gs[0, 2:])
    ax_info.axis("off")
    lines = [f"Target: {target}" if target else "Target: not specified"]
    if estimated_download_mb is not None:
        lines.append(f"Required download: {estimated_download_mb:.1f} MB")
    ax_info.text(0.0, 0.7, "\n".join(lines), fontsize=9.5, color="#495057", transform=ax_info.transAxes, va="top")

    # ── Metric cards ────────────────────────────────────────────────────────
    cards = [
        ("Recordings", str(report.n_recordings), "#343a40"),
        ("Loadable", str(report.n_loadable), "#2f9e44" if report.n_loadable else "#868e96"),
        ("Event-complete", str(report.n_event_complete), "#2f9e44" if report.n_event_complete else "#868e96"),
        ("Label-ready", str(report.n_label_ready), "#2f9e44" if report.n_label_ready else "#c92a2a"),
    ]
    for i, (label, value, color) in enumerate(cards):
        ax = fig.add_subplot(gs[1, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                                    facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=1))
        ax.text(0.12, 0.58, value, fontsize=15, fontweight="bold", color=color, transform=ax.transAxes, va="center")
        ax.text(0.12, 0.22, label, fontsize=8, color="#495057", transform=ax.transAxes, va="center")

    # ── Trainability bars ───────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[2, :])
    n = max(report.n_recordings, 1)
    metrics = [
        ("label-ready", report.n_label_ready),
        ("event-complete", report.n_event_complete),
        ("loadable", report.n_loadable),
        ("recordings", report.n_recordings),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    y_pos = range(len(labels))
    colors = ["#c92a2a" if v == 0 else "#4c6ef5" for v in values]
    ax_bar.barh(y_pos, values, color=colors)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("count")
    ax_bar.set_title("Trainability assessment", fontsize=11, fontweight="bold", loc="left")
    ax_bar.axvline(n * 0.5, color="#adb5bd", linestyle="--", linewidth=1)
    ax_bar.text(n * 0.5, len(labels) - 0.4, "minimum recommended", fontsize=7.5, color="#868e96", ha="left")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    # ── Blockers panel ──────────────────────────────────────────────────────
    ax_block = fig.add_subplot(gs[3, :2])
    ax_block.axis("off")
    if report.n_label_ready == 0:
        header = "Why this dataset cannot be used for training"
        body = (
            f"No label-ready recordings were found"
            + (f" for target {target!r}." if target else ".")
            + "\nCheck annotations or select a different target."
        )
    elif errors:
        header = "Blockers"
        body = "\n".join(f"• {e.message}" for e in errors[:4])
    elif warnings_:
        header = "Warnings"
        body = "\n".join(f"• {w.message}" for w in warnings_[:4])
    else:
        header = "Blockers"
        body = "None — no error-level findings."
    ax_block.text(0.0, 0.95, header, fontsize=10.5, fontweight="bold", color="#20242c", va="top", transform=ax_block.transAxes)
    ax_block.text(0.0, 0.72, body, fontsize=9, color="#495057", va="top", transform=ax_block.transAxes)

    # ── Next steps panel ─────────────────────────────────────────────────────
    ax_next = fig.add_subplot(gs[3, 2:])
    ax_next.axis("off")
    ax_next.text(0.0, 0.95, "Recommended next steps", fontsize=10.5, fontweight="bold", color="#20242c",
                 va="top", transform=ax_next.transAxes)
    wrapped_steps = [
        textwrap.fill(f"• {r}", width=52, subsequent_indent="  ")
        for r in recommendations[:5]
    ] or ["• No further action recommended."]
    ax_next.text(0.0, 0.72, "\n".join(wrapped_steps), fontsize=9, color="#495057", va="top", transform=ax_next.transAxes)

    header_title = title or "Dataset readiness summary"
    fig.suptitle(header_title, fontsize=13, fontweight="bold", x=0.02, y=0.985, ha="left", va="top")
    fig.subplots_adjust(top=0.9, left=0.11, right=0.98, bottom=0.06)
    return fig


__all__ = ["dataset_readiness_figure"]
