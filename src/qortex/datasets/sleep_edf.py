"""qortex.datasets.sleep_edf — Sleep-EDF Expanded Dataset (PhysioNet).

Usage::

    from qortex.datasets import sleep_edf

    card = sleep_edf.describe()
    bundle = sleep_edf.load_data(subjects=[0, 1, 2])
    X, y = bundle.to_windows(window_s=30.0, event_driven=False)
    # y: {0=Wake, 1=N1, 2=N2, 3=N3, 4=REM}

Dataset facts
-------------
- Sleep-EDF Expanded: 197 whole-night PSG recordings.
- 20 Cassette recordings (SC) + 22 Telemetry recordings (ST).
- EEG (Fpz-Cz, Pz-Oz), EOG (horizontal), chin EMG, event markers.
- Annotation channel: W=Wake, R=REM, 1=N1, 2=N2, 3=N3, 4=N3 (AASM mapped), M=Movement, ?=Unknown.
- Label mapping: {W→0, 1→1, 2→2, 3→3, 4→3, R→4}; M and ? excluded.
- License: Open Data Commons Attribution License v1.0.
- Source: https://physionet.org/content/sleep-edfx/1.0.0/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, EEGBundle, _REGISTRY

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="sleep_edf",
    full_name="Sleep-EDF Expanded Dataset",
    version="1.0.0",
    source_url="https://physionet.org/content/sleep-edfx/1.0.0/",
    license="Open Data Commons Attribution License v1.0",
    citation="Goldberger AL et al. PhysioBank, PhysioToolkit, PhysioNet. Circulation. 2000.",
    modality="eeg",
    n_subjects=82,
    n_channels=7,
    sampling_hz=100.0,
    description=(
        "197 whole-night polysomnographic (PSG) recordings with hypnogram annotations.\n"
        "Channels: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental, event markers.\n"
        "Target: 5-class sleep staging (Wake / N1 / N2 / N3 / REM).\n"
        "Downloaded on demand via MNE (EDF format)."
    ),
    tasks=["sleep_stage_classification"],
    tutorial_ids=["T03"],
    size_gb_approx=2.3,
    requires_registration=False,
    access_instructions=None,
)
_REGISTRY.register(_CARD)


# ── Label map ─────────────────────────────────────────────────────────────────

# 5-class AASM-aligned mapping for tutorial T03
LABEL_MAP = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

# Raw hypnogram annotation → integer class
_ANNOTATION_TO_CLASS = {
    "Sleep stage W": 0,
    "W": 0,
    "Sleep stage 1": 1,
    "1": 1,
    "Sleep stage 2": 2,
    "2": 2,
    "Sleep stage 3": 3,
    "3": 3,
    "Sleep stage 4": 3,  # AASM N3 = 3+4
    "4": 3,
    "Sleep stage R": 4,
    "R": 4,
    # M and ? are excluded (not in this map → will be skipped)
}

_EXCLUDED_ANNOTATIONS = {"Sleep stage M", "M", "Sleep stage ?", "?"}


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    subjects: list[int] | None = None,
    recording: str = "SC",
    preload: bool = True,
    crop_wake_mins: int = 30,
    verbose: bool = False,
) -> EEGBundle:
    """Download (first call) and load Sleep-EDF data.

    Parameters
    ----------
    subjects        : Subject indices (0-based). Defaults to [0, 1, 2].
    recording       : "SC" (Cassette), "ST" (Telemetry), or "SC_ST" for both.
    preload         : Load raw data into memory.
    crop_wake_mins  : Crop leading/trailing wake to N minutes to reduce
                      class imbalance. Set to None to disable.
    verbose         : MNE verbosity.

    Returns
    -------
    EEGBundle with label_map = LABEL_MAP (5-class AASM staging).

    Examples
    --------
    >>> bundle = sleep_edf.load_data(subjects=[0, 1, 2, 3])
    >>> X, y = bundle.to_windows(window_s=30.0, event_driven=False)
    """
    try:
        import mne  # type: ignore[import]
        from mne.datasets.sleep_physionet.age import fetch_data  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "Sleep-EDF loading requires MNE. Install with: pip install 'qortex[eeg]'"
        ) from None

    if subjects is None:
        subjects = [0, 1, 2]

    local_paths: list[Path] = []
    raws = []
    channel_names: list[str] | None = None
    sfreq: float | None = None
    metadata: dict[str, Any] = {}

    if recording not in ("SC", "ST", "SC_ST"):
        raise ValueError(f"recording must be 'SC', 'ST', or 'SC_ST', got {recording!r}")

    for subject in subjects:
        files: list[tuple[str, str]] = []
        try:
            # age.fetch_data's `recording` argument selects night numbers (1, 2)
            # for the Sleep Cassette (SC) study — it is not "SC"/"ST", those are
            # two separate PhysioNet studies served by two different MNE fetchers.
            if recording in ("SC", "SC_ST"):
                files.extend(fetch_data(
                    subjects=[subject], recording=[1, 2], on_missing="warn", verbose=verbose,
                ))
            if recording in ("ST", "SC_ST"):
                from mne.datasets.sleep_physionet.temazepam import fetch_data as _fetch_temazepam
                files.extend(_fetch_temazepam(subjects=[subject], verbose=verbose))
        except Exception as exc:
            import warnings
            warnings.warn(f"Could not fetch sleep-EDF subject {subject}: {exc}", RuntimeWarning, stacklevel=2)
            continue

        for psg_path, ann_path in files:
            psg = Path(psg_path)
            ann = Path(ann_path)
            local_paths.append(psg)
            local_paths.append(ann)

            try:
                raw = mne.io.read_raw_edf(str(psg), preload=preload, verbose=verbose)
                annotations = mne.read_annotations(str(ann))

                # Filter out movement and unknown annotations
                keep_mask = [
                    ann_desc not in _EXCLUDED_ANNOTATIONS
                    for ann_desc in annotations.description
                ]
                annotations_clean = annotations[keep_mask]
                raw.set_annotations(annotations_clean, verbose=verbose)

                if crop_wake_mins is not None:
                    _crop_flanking_wake(raw, crop_wake_mins, verbose=verbose)

                raws.append(raw)
                if channel_names is None:
                    channel_names = list(raw.info.ch_names)
                    sfreq = raw.info["sfreq"]
            except Exception as exc:
                import warnings
                warnings.warn(f"Could not read {psg}: {exc}", RuntimeWarning, stacklevel=2)

            metadata[f"subject_{subject}"] = {
                "subject": subject,
                "psg": str(psg),
                "hypnogram": str(ann),
            }

    if sfreq is None:
        sfreq = 100.0
    if channel_names is None:
        channel_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental", "Event marker"]

    return EEGBundle(
        card=_CARD,
        subjects=subjects,
        runs=[],  # Sleep EDF has no run numbers
        sfreq=sfreq,
        channel_names=channel_names,
        label_map=LABEL_MAP,
        local_paths=[p for p in local_paths if p.suffix == ".edf"],
        raws=raws,
        metadata=metadata,
        annotation_to_class=_ANNOTATION_TO_CLASS,
    )


def _crop_flanking_wake(raw: Any, max_wake_mins: int, verbose: bool) -> None:
    """Crop flanking wake periods to reduce severe class imbalance."""
    try:
        import mne
        import numpy as np
        annotations = raw.annotations
        tmax = raw.times[-1]

        # Find first non-wake annotation
        first_sleep_onset = tmax
        last_sleep_offset = 0.0
        for ann in annotations:
            if ann["description"] not in {"Sleep stage W", "W"}:
                onset = float(ann["onset"])
                offset = onset + float(ann["duration"])
                if onset < first_sleep_onset:
                    first_sleep_onset = onset
                if offset > last_sleep_offset:
                    last_sleep_offset = offset

        # Crop: keep max_wake_mins of wake before first sleep
        tmin_crop = max(0.0, first_sleep_onset - max_wake_mins * 60.0)
        tmax_crop = min(tmax, last_sleep_offset + max_wake_mins * 60.0)
        if tmax_crop > tmin_crop + 30.0:
            raw.crop(tmin=tmin_crop, tmax=tmax_crop, verbose=verbose)
    except Exception:
        pass  # Non-fatal: continue without cropping
