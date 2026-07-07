"""Exploratory data analysis public API.

EDA helpers have optional dataframe/plotting dependencies. Keep package import
lightweight and load concrete helpers on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EDAEngine",
    "build_dataset_summary",
    "build_modality_summaries",
    "compute_quality_metrics",
    "summarize_events",
    "file_table",
    "coverage_matrix",
    "modality_bar",
    "subject_coverage_heatmap",
    "size_distribution",
    "task_event_coverage",
    "parse_participants_tsv",
    "summarize_categorical",
    "numeric_by_group",
    "participants_metadata_figure",
    "dataset_readiness_figure",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EDAEngine": ("qortex.eda.report", "EDAEngine"),
    "build_dataset_summary": ("qortex.eda.summary", "build_dataset_summary"),
    "build_modality_summaries": ("qortex.eda.summary", "build_modality_summaries"),
    "compute_quality_metrics": ("qortex.eda.quality", "compute_quality_metrics"),
    "summarize_events": ("qortex.eda.events", "summarize_events"),
    "file_table": ("qortex.eda.summary", "file_table"),
    "coverage_matrix": ("qortex.eda.summary", "coverage_matrix"),
    "modality_bar": ("qortex.eda.plots", "modality_bar"),
    "subject_coverage_heatmap": ("qortex.eda.plots", "subject_coverage_heatmap"),
    "size_distribution": ("qortex.eda.plots", "size_distribution"),
    "task_event_coverage": ("qortex.eda.plots", "task_event_coverage"),
    "parse_participants_tsv": ("qortex.eda.participants", "parse_participants_tsv"),
    "summarize_categorical": ("qortex.eda.participants", "summarize_categorical"),
    "numeric_by_group": ("qortex.eda.participants", "numeric_by_group"),
    "participants_metadata_figure": ("qortex.eda.participants", "participants_metadata_figure"),
    "dataset_readiness_figure": ("qortex.eda.dataset_readiness", "dataset_readiness_figure"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
