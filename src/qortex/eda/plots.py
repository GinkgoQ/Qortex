"""EDA visualisation helpers — returns plotly figures (optional dependency)."""

from __future__ import annotations

from typing import Any


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError(
            "Visualisation requires plotly. Install with: pip install plotly"
        )


def modality_bar(modality_counts: dict[str, int]) -> Any:
    """Horizontal bar chart of file count per modality."""
    go = _require_plotly()
    labels = list(modality_counts.keys())
    values = list(modality_counts.values())
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#5c85d6",
    ))
    fig.update_layout(
        title="Files per Modality",
        xaxis_title="File Count",
        height=max(300, len(labels) * 40),
        margin=dict(l=120, r=20, t=40, b=40),
    )
    return fig


def subject_coverage_heatmap(coverage_df: Any) -> Any:
    """Heatmap of subject × modality coverage."""
    import polars as pl

    go = _require_plotly()
    if coverage_df.is_empty():
        return go.Figure()

    subjects = sorted(coverage_df["subject"].drop_nulls().unique().to_list())
    modalities = sorted(coverage_df["modality"].drop_nulls().unique().to_list())

    z = []
    for sub in subjects:
        row = []
        for mod in modalities:
            count = len(coverage_df.filter(
                (pl.col("subject") == sub) & (pl.col("modality") == mod)
            ))
            row.append(count)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=modalities,
        y=subjects,
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title="Subject × Modality Coverage",
        xaxis_title="Modality",
        yaxis_title="Subject",
        height=max(400, len(subjects) * 20 + 80),
    )
    return fig


def size_distribution(file_df: Any) -> Any:
    """Histogram of file sizes in MB."""
    import polars as pl

    go = _require_plotly()
    sizes = file_df.filter(pl.col("size_mb").is_not_null())["size_mb"].to_list()
    fig = go.Figure(go.Histogram(x=sizes, nbinsx=50, marker_color="#5c85d6"))
    fig.update_layout(
        title="File Size Distribution",
        xaxis_title="Size (MB)",
        yaxis_title="Count",
    )
    return fig


def task_event_coverage(
    task_counts: dict[str, int],
    events_counts: dict[str, int],
) -> Any:
    """Bar chart comparing signal files vs events files per task."""
    go = _require_plotly()
    tasks = sorted(set(list(task_counts) + list(events_counts)))
    signal_vals = [task_counts.get(t, 0) for t in tasks]
    events_vals = [events_counts.get(t, 0) for t in tasks]

    fig = go.Figure([
        go.Bar(name="Signal files", x=tasks, y=signal_vals, marker_color="#5c85d6"),
        go.Bar(name="Events files", x=tasks, y=events_vals, marker_color="#85d65c"),
    ])
    fig.update_layout(
        barmode="group",
        title="Signal vs Events File Coverage per Task",
        xaxis_title="Task",
        yaxis_title="Count",
    )
    return fig
