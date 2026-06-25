"""Signal windowing — event-aligned and fixed-stride windows from SampleRecords."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from qortex.core.entities import EventsRecord, SampleRecord


@dataclass(frozen=True)
class WindowSpec:
    """Parameters for a sliding or event-aligned window."""

    duration_s: float
    overlap: float = 0.0  # fraction of window that overlaps with previous
    tmin: float = 0.0     # for event-aligned: seconds before event onset


def fixed_windows(
    sample: SampleRecord,
    spec: WindowSpec,
) -> Iterator[SampleRecord]:
    """Yield fixed-stride windows from a SampleRecord.

    Signal in sample.data is expected to be shaped (n_channels, n_timepoints).
    """
    if sample.data is None or sample.sfreq is None:
        return

    sfreq = sample.sfreq
    sig: np.ndarray = np.asarray(sample.data)
    if sig.ndim != 2:
        return

    win_samples = int(round(spec.duration_s * sfreq))
    step_samples = max(1, int(round(win_samples * (1.0 - spec.overlap))))
    n_ch, n_times = sig.shape

    win_idx = 0
    start = 0
    while start + win_samples <= n_times:
        window = sig[:, start : start + win_samples]
        onset_s = start / sfreq
        prov = dict(sample.provenance)
        prov["window_index"] = win_idx
        prov["window_onset_s"] = onset_s

        yield SampleRecord(
            data=window,
            label=sample.label,
            label_name=sample.label_name,
            subject=sample.subject,
            session=sample.session,
            task=sample.task,
            run=sample.run,
            modality=sample.modality,
            onset=onset_s,
            duration=spec.duration_s,
            sfreq=sfreq,
            split=sample.split,
            provenance=prov,
        )
        start += step_samples
        win_idx += 1


def event_aligned_windows(
    sample: SampleRecord,
    events: EventsRecord,
    spec: WindowSpec,
) -> Iterator[SampleRecord]:
    """Yield one window per event onset, aligned to the event.

    Events.data is a Polars DataFrame with at minimum an 'onset' column
    (seconds) and optionally 'trial_type' for labels.
    """
    if sample.data is None or sample.sfreq is None:
        return

    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required for event-aligned windowing")

    sfreq = sample.sfreq
    sig: np.ndarray = np.asarray(sample.data)
    if sig.ndim != 2:
        return

    n_ch, n_times = sig.shape
    win_samples = int(round(spec.duration_s * sfreq))

    df: pl.DataFrame = events.data
    has_trial_type = "trial_type" in df.columns
    has_onset = "onset" in df.columns
    if not has_onset:
        return

    unique_types = (
        sorted(df["trial_type"].drop_nulls().unique().to_list())
        if has_trial_type else []
    )
    label_map: dict[str, int] = {tt: i for i, tt in enumerate(unique_types)}

    for i, row in enumerate(df.iter_rows(named=True)):
        onset_s = float(row.get("onset", 0)) + spec.tmin
        start_sample = int(round(onset_s * sfreq))
        end_sample = start_sample + win_samples

        if start_sample < 0 or end_sample > n_times:
            continue

        window = sig[:, start_sample:end_sample]
        trial_type = row.get("trial_type", "unknown") if has_trial_type else None
        label = label_map.get(trial_type, -1) if trial_type else None

        prov = dict(sample.provenance)
        prov["event_index"] = i
        prov["event_onset_s"] = float(row.get("onset", 0))

        yield SampleRecord(
            data=window,
            label=label,
            label_name=trial_type,
            subject=sample.subject,
            session=sample.session,
            task=sample.task,
            run=sample.run,
            modality=sample.modality,
            onset=onset_s,
            duration=spec.duration_s,
            sfreq=sfreq,
            split=sample.split,
            provenance=prov,
        )
