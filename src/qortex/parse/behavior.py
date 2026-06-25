"""Behavioral / events / phenotype loader — Polars with full BIDS TSV semantics."""

from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Any, Iterator

from qortex.core.entities import EventsRecord, FileRecord, SampleRecord
from qortex.core.exceptions import LoadError

log = logging.getLogger(__name__)

# All BIDS column names that carry event onset timing
_ONSET_COLUMNS = ("onset", "onset_time", "onset_s")
# Candidate label columns in priority order
_LABEL_COLUMNS = (
    "trial_type", "event_type", "stim_type",
    "condition", "category", "label",
)
# Candidate reaction-time columns
_RT_COLUMNS = ("response_time", "reaction_time", "rt", "RT")

_BIDS_NULL_VALUES = ["n/a", "N/A", "NA", "NaN", "nan", "", "undefined"]

_BEHAVIORAL_SUFFIXES = frozenset({
    "events", "channels", "electrodes", "physio", "stim",
    "scans", "sessions", "participants", "beh",
})


class BehaviorLoader:
    modality = "behavior"
    supported_extensions = frozenset({".tsv", ".csv"})

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.extension in self.supported_extensions
            and (
                file.suffix in _BEHAVIORAL_SUFFIXES
                or file.datatype == "beh"
                or file.filename in {"participants.tsv", "scans.tsv"}
            )
            and not file.is_dir
        )

    # ── inspect ───────────────────────────────────────────────────────────

    def inspect(self, file: FileRecord, local_path: Path) -> dict[str, Any]:
        try:
            sep = "\t" if file.extension == ".tsv" else ","
            # Count rows without loading full file
            n_rows = _count_rows(local_path, sep)
            columns, dtypes, label_col, label_preview = _inspect_table_head(
                local_path,
                sep,
            )
            onset_col = next((c for c in _ONSET_COLUMNS if c in columns), None)

            return {
                "n_rows": n_rows,
                "columns": columns,
                "dtypes": dtypes,
                "label_column": label_col,
                "label_values_preview": label_preview,
                "onset_column": onset_col,
                "has_duration": "duration" in columns,
                "has_response_time": any(c in columns for c in _RT_COLUMNS),
                "size_bytes": local_path.stat().st_size,
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect behavioral file {local_path}: {exc}",
            ) from exc

    # ── load ──────────────────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> EventsRecord:
        try:
            import polars as pl

            sep = "\t" if file.extension == ".tsv" else ","
            df = pl.read_csv(
                str(local_path),
                separator=sep,
                null_values=_BIDS_NULL_VALUES,
                infer_schema_length=10_000,
                ignore_errors=False,
                **{k: v for k, v in kwargs.items() if k not in {"separator", "null_values"}},
            )
        except Exception as exc:
            raise LoadError(
                f"Cannot load behavioral file {local_path}: {exc}",
            ) from exc

        label_col = _detect_label_col(df)

        # Encode label column as integer category
        label_values: list = []
        label_encoder: dict[str, int] = {}
        if label_col and df[label_col].dtype == pl.Utf8:
            unique = sorted(df[label_col].drop_nulls().unique().to_list())
            label_values = unique
            label_encoder = {v: i for i, v in enumerate(unique)}
            df = df.with_columns(
                pl.col(label_col)
                .map_elements(lambda x: label_encoder.get(x), return_dtype=pl.Int64)
                .alias(f"{label_col}_encoded")
            )

        # Onset column normalisation: ensure float seconds
        onset_col = next((c for c in _ONSET_COLUMNS if c in df.columns), None)
        if onset_col and onset_col != "onset":
            df = df.rename({onset_col: "onset"})

        return EventsRecord(
            file=file,
            data=df,
            columns=df.columns,
            n_events=len(df),
            label_column=label_col,
            label_values=label_values,
        )

    # ── lazy_load ─────────────────────────────────────────────────────────

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> EventsRecord:
        """Polars scan_csv — metadata read immediately, data materialised on demand.

        This is the true lazy path: returns an EventsRecord whose .data is a
        Polars LazyFrame rather than a DataFrame.  Downstream code that calls
        .data.collect() will trigger actual I/O.
        """
        try:
            import polars as pl

            sep = "\t" if file.extension == ".tsv" else ","
            lf = pl.scan_csv(
                str(local_path),
                separator=sep,
                null_values=_BIDS_NULL_VALUES,
                infer_schema_length=10_000,
                ignore_errors=True,
            )
            # Materialise schema only (no data rows read)
            schema = lf.schema
            n_rows = _count_rows(local_path, sep)
            cols = list(schema.keys())
            df_head = lf.limit(5).collect()
            label_col = _detect_label_col(df_head)
            label_values = (
                df_head[label_col].drop_nulls().unique().to_list()
                if label_col else []
            )
        except Exception as exc:
            raise LoadError(
                f"Cannot lazy-load behavioral file {local_path}: {exc}",
            ) from exc

        return EventsRecord(
            file=file,
            data=lf,           # LazyFrame — not yet materialised
            columns=cols,
            n_events=n_rows,
            label_column=label_col,
            label_values=label_values,
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: EventsRecord, **kwargs) -> Any:
        """Convert the events table to a numpy structured array."""
        df = record.data
        if hasattr(df, "collect"):
            df = df.collect()
        return df.to_numpy(structured=True)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: EventsRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord per row — each row is a behavioural event.

        The raw row dict is stored in SampleRecord.data.
        Numeric label (from encoded column) is in SampleRecord.label.
        String label (original) is in SampleRecord.label_name.
        """
        df = record.data
        if hasattr(df, "collect"):
            df = df.collect()

        label_col = record.label_column
        encoded_col = f"{label_col}_encoded" if label_col else None
        ents = record.file.entities
        rt_col = next((c for c in _RT_COLUMNS if c in df.columns), None)

        for row in df.iter_rows(named=True):
            label_int = row.get(encoded_col) if encoded_col else None
            label_str = row.get(label_col) if label_col else None
            onset = _safe_float(row.get("onset"))
            duration = _safe_float(row.get("duration"))
            rt = _safe_float(row.get(rt_col)) if rt_col else None

            yield SampleRecord(
                data=row,
                label=label_int,
                label_name=str(label_str) if label_str is not None else None,
                subject=ents.subject,
                session=ents.session,
                task=ents.task,
                run=ents.run,
                modality=self.modality,
                onset=onset,
                duration=duration,
                provenance={
                    "source": record.file.path,
                    "response_time": rt,
                },
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_label_col(df: Any) -> str | None:
    for candidate in _LABEL_COLUMNS:
        if candidate in df.columns:
            return candidate
    return None


def _inspect_table_head(path: Path, sep: str) -> tuple[list[str], dict[str, str], str | None, list]:
    try:
        import polars as pl

        df_head = pl.read_csv(
            str(path),
            separator=sep,
            null_values=_BIDS_NULL_VALUES,
            n_rows=10,
            infer_schema_length=1000,
            ignore_errors=True,
        )
        label_col = _detect_label_col(df_head)
        label_preview: list = []
        if label_col:
            label_preview = df_head[label_col].drop_nulls().unique().to_list()
        return (
            df_head.columns,
            {c: str(t) for c, t in zip(df_head.columns, df_head.dtypes)},
            label_col,
            label_preview,
        )
    except ImportError:
        return _inspect_table_head_stdlib(path, sep)


def _inspect_table_head_stdlib(path: Path, sep: str) -> tuple[list[str], dict[str, str], str | None, list]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        columns = reader.fieldnames or []
        label_col = next((c for c in _LABEL_COLUMNS if c in columns), None)
        seen: list[str] = []
        for idx, row in enumerate(reader):
            if label_col:
                value = row.get(label_col)
                if value and value not in _BIDS_NULL_VALUES and value not in seen:
                    seen.append(value)
            if idx >= 9:
                break
    return columns, {column: "str" for column in columns}, label_col, seen


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _count_rows(path: Path, sep: str) -> int:
    """Count data rows without loading the full file into memory."""
    try:
        n = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            next(fh)  # skip header
            for _ in fh:
                n += 1
        return n
    except Exception:
        return -1
