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

_STATUS_KEY = {"ready": "success", "uncertain": "warning", "blocked": "danger"}


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

    from qortex.visualize.design import (
        BORDER, CARD_BG, INK, SUBINK, STATUS,
        apply_theme, figure_title, metric_card, section_title, status_badge,
    )

    apply_theme()

    status = _status_for(report)
    status_color = STATUS[_STATUS_KEY[status]]

    errors = [f for f in report.findings if f.severity == "error"]
    warnings_ = [f for f in report.findings if f.severity == "warning"]
    recommendations = []
    for f in report.findings:
        if f.recommendation and f.recommendation not in recommendations:
            recommendations.append(f.recommendation)

    fig = plt.figure(figsize=(12.0, 7.4))
    gs = gridspec.GridSpec(
        4, 4, height_ratios=[0.7, 0.62, 1.15, 1.05], hspace=0.68, wspace=0.4, figure=fig,
        top=0.86, bottom=0.06, left=0.13, right=0.97,
    )

    figure_title(fig, "Dataset readiness summary", subtitle=f"{report.dataset_id}  ·  snapshot {report.snapshot}")

    # ── Header: status badge + target/download info ──────────────────────────
    ax_head = fig.add_subplot(gs[0, :2])
    ax_head.axis("off")
    status_badge(ax_head, text=f"status: {status}", color=status_color, x=0.0, y=0.55, fontsize=12)
    ax_head.text(0.0, 0.1, f"readiness score  {report.score:.0f}/100", fontsize=9.5, color=SUBINK,
                 transform=ax_head.transAxes)

    ax_info = fig.add_subplot(gs[0, 2:])
    ax_info.axis("off")
    lines = [f"Target: {target}" if target else "Target: not specified"]
    if estimated_download_mb is not None:
        lines.append(f"Required download: {estimated_download_mb:.1f} MB")
    ax_info.text(0.0, 0.6, "\n".join(lines), fontsize=9.5, color=SUBINK, transform=ax_info.transAxes, va="top")

    # ── Metric cards ────────────────────────────────────────────────────────
    cards = [
        ("Recordings", str(report.n_recordings), INK, None),
        ("Loadable", str(report.n_loadable), STATUS["success"] if report.n_loadable else SUBINK,
         STATUS["success"] if report.n_loadable else BORDER),
        ("Event-complete", str(report.n_event_complete), STATUS["success"] if report.n_event_complete else SUBINK,
         STATUS["success"] if report.n_event_complete else BORDER),
        ("Label-ready", str(report.n_label_ready), STATUS["success"] if report.n_label_ready else STATUS["danger"],
         STATUS["success"] if report.n_label_ready else STATUS["danger"]),
    ]
    for i, (label, value, color, accent) in enumerate(cards):
        ax = fig.add_subplot(gs[1, i])
        metric_card(ax, value=value, label=label, color=color, accent=accent)

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
    colors = [STATUS["danger"] if v == 0 else "#4f46e5" for v in values]
    bars = ax_bar.barh(y_pos, values, color=colors, height=0.6)
    for bar, v in zip(bars, values):
        ax_bar.text(bar.get_width() + n * 0.012, bar.get_y() + bar.get_height() / 2, f"{v:,}",
                    va="center", fontsize=8.5, color=INK)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("count", fontsize=9, color=SUBINK)
    ax_bar.set_xlim(0, n * 1.12)
    section_title(ax_bar, "Trainability assessment", y=1.12)
    ax_bar.axvline(n * 0.5, color="#c7cbd1", linestyle="--", linewidth=1.1)
    ax_bar.text(n * 0.5 + n * 0.012, len(labels) - 0.55, "minimum recommended", fontsize=7.5, color=SUBINK, ha="left")
    ax_bar.grid(axis="y", visible=False)

    # ── Blockers panel ──────────────────────────────────────────────────────
    ax_block = fig.add_subplot(gs[3, :2])
    ax_block.axis("off")
    if report.n_label_ready == 0:
        header = "Why this dataset cannot be used for training"
        body = (
            "No label-ready recordings were found"
            + (f" for target {target!r}." if target else ".")
            + "\nCheck annotations or select a different target."
        )
    elif errors:
        header = "Blockers"
        body = "\n".join(textwrap.fill(f"• {e.message}", width=52) for e in errors[:4])
    elif warnings_:
        header = "Warnings"
        body = "\n".join(textwrap.fill(f"• {w.message}", width=52) for w in warnings_[:4])
    else:
        header = "Blockers"
        body = "None — no error-level findings."
    section_title(ax_block, header, y=1.0)
    ax_block.text(0.0, 0.86, body, fontsize=9, color=SUBINK, va="top", transform=ax_block.transAxes)

    # ── Next steps panel ─────────────────────────────────────────────────────
    ax_next = fig.add_subplot(gs[3, 2:])
    ax_next.axis("off")
    section_title(ax_next, "Recommended next steps", y=1.0)
    wrapped_steps = [
        textwrap.fill(f"• {r}", width=52, subsequent_indent="  ")
        for r in recommendations[:5]
    ] or ["• No further action recommended."]
    ax_next.text(0.0, 0.86, "\n".join(wrapped_steps), fontsize=9, color=SUBINK, va="top", transform=ax_next.transAxes)

    return fig


__all__ = ["dataset_readiness_figure"]
