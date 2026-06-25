"""Local events.tsv summaries for label-readiness and EDA."""

from __future__ import annotations

import csv
from pathlib import Path

from qortex.core.entities import EventLabelSummary, Manifest
LABEL_COLUMN_PREFERENCE = (
    "trial_type",
    "event_type",
    "condition",
    "category",
    "label",
    "stim_type",
)


def summarize_events(
    manifest: Manifest,
    local_path: str | Path,
    *,
    max_files: int | None = None,
) -> list[EventLabelSummary]:
    """Summarize local BIDS events files with conservative label detection."""
    root = Path(local_path).expanduser().resolve()
    events = [
        file for file in manifest.files
        if file.suffix == "events" and file.extension == ".tsv" and not file.is_dir
    ]
    if max_files is not None:
        events = events[:max_files]

    summaries: list[EventLabelSummary] = []
    for file in events:
        path = root / file.path
        if not path.exists():
            continue
        summary = _summarize_one(path, file.path)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _summarize_one(path: Path, relative_path: str) -> EventLabelSummary | None:
    try:
        import polars as pl

        df = pl.read_csv(
            path,
            separator="\t",
            null_values=["n/a", "N/A", "NA", "NaN", "nan", ""],
            ignore_errors=True,
        )
    except Exception:
        return _summarize_one_stdlib(path, relative_path)

    n_events = df.height
    label_column = _choose_label_column(df.columns)
    if label_column is None:
        return EventLabelSummary(path=relative_path, n_events=n_events)

    counts_df = (
        df
        .filter(pl.col(label_column).is_not_null())
        .group_by(label_column)
        .len()
        .sort("len", descending=True)
    )
    label_counts = {
        str(row[label_column]): int(row["len"])
        for row in counts_df.iter_rows(named=True)
    }
    n_missing = int(df.select(pl.col(label_column).is_null().sum()).item())
    return EventLabelSummary(
        path=relative_path,
        n_events=n_events,
        label_column=label_column,
        label_counts=label_counts,
        n_missing_labels=n_missing,
    )


def _summarize_one_stdlib(path: Path, relative_path: str) -> EventLabelSummary | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            columns = reader.fieldnames or []
            label_column = _choose_label_column(columns)
            n_events = 0
            counts: dict[str, int] = {}
            n_missing = 0
            for row in reader:
                n_events += 1
                if label_column is None:
                    continue
                value = _clean_label(row.get(label_column))
                if value is None:
                    n_missing += 1
                    continue
                counts[value] = counts.get(value, 0) + 1
    except Exception:
        return None
    return EventLabelSummary(
        path=relative_path,
        n_events=n_events,
        label_column=label_column,
        label_counts=counts,
        n_missing_labels=n_missing,
    )


def _choose_label_column(columns: list[str]) -> str | None:
    for candidate in LABEL_COLUMN_PREFERENCE:
        if candidate in columns:
            return candidate
    return None


def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"n/a", "N/A", "NA", "NaN", "nan"}:
        return None
    return text
