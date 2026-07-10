"""qortex.datasets.eegbci — PhysioNet EEG Motor Movement/Imagery Dataset.

Usage::

    from qortex.datasets import eegbci

    card = eegbci.describe()
    print(card)

    bundle = eegbci.load_data(subjects=[1, 2, 3], runs=[4, 8, 12])
    X, y = bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
    # X: [n_epochs, 64, 640]   y: {0=rest, 1=left_fist, 2=right_fist}

    # Connectivity workflow (T02)
    bundle2 = eegbci.load_data(subjects=[1, 2, 3, 4, 5], runs=[1, 2])
    # label_map: {1: 'eyes_open', 2: 'eyes_closed'}

Dataset facts
-------------
- 109 subjects, 14 runs per subject (some missing).
- 64 EEG channels, 160 Hz.
- Runs 3–14 have motor task; runs 1–2 are resting-state baselines.
- Annotations: T0=rest, T1/T2 meanings depend on run number.
- License: Open Data Commons Attribution License v1.0 (PhysioNet).
- Source: https://physionet.org/content/eegmmidb/1.0.0/

Run → task mapping (from PhysioNet documentation)
--------------------------------------------------
Run 1,2   : Baseline eyes-open / eyes-closed (no task events)
Run 3,7,11: Open/close left or right FIST (execution)
Run 4,8,12: Imagine opening/closing left or right FIST (imagery)
Run 5,9,13: Open/close both FISTS or both FEET (execution)
Run 6,10,14: Imagine opening/closing both FISTS or both FEET (imagery)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, EEGBundle, _REGISTRY

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="eegbci",
    full_name="PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)",
    version="1.0.0",
    source_url="https://physionet.org/content/eegmmidb/1.0.0/",
    license="Open Data Commons Attribution License v1.0",
    citation="Goldberger AL, et al. (2000) PhysioBank, PhysioToolkit, PhysioNet. Circulation. 101(23):e215.",
    modality="eeg",
    n_subjects=109,
    n_channels=64,
    sampling_hz=160.0,
    description=(
        "109 subjects performing motor/imagery tasks.\n"
        "14 runs per subject: 2 baseline (eyes-open/closed) + 12 task runs.\n"
        "Task runs alternate between fist movements and fist/feet imagery.\n"
        "Downloaded on demand via MNE (EDF+ format, ~3.4 GB uncompressed)."
    ),
    tasks=["motor_imagery_classification", "eeg_connectivity"],
    tutorial_ids=["T01", "T02"],
    size_gb_approx=3.4,
    requires_registration=False,
    access_instructions=None,
)
_REGISTRY.register(_CARD)


# ── Label maps ────────────────────────────────────────────────────────────────

# For runs 4, 8, 12 (imagine left/right fist):
LABEL_MAP_FIST_IMAGERY = {0: "rest", 1: "left_fist_imagery", 2: "right_fist_imagery"}

# For runs 6, 10, 14 (imagine both fists / both feet):
LABEL_MAP_FEET_IMAGERY = {0: "rest", 1: "both_fists_imagery", 2: "both_feet_imagery"}

# For runs 3, 7, 11 (execute left/right fist):
LABEL_MAP_FIST_EXECUTION = {0: "rest", 1: "left_fist", 2: "right_fist"}

# For runs 5, 9, 13 (execute both fists / both feet):
LABEL_MAP_FEET_EXECUTION = {0: "rest", 1: "both_fists", 2: "both_feet"}

# For baseline runs 1, 2:
LABEL_MAP_BASELINE = {1: "eyes_open", 2: "eyes_closed"}


def _label_map_for_runs(runs: list[int]) -> dict[int, str]:
    """Return a merged label map for the given run list.

    When a mix of run types is used, the T1/T2 meanings differ.
    We include the run number in the label string only when ambiguous.
    """
    sets = set(runs)
    if sets <= {1, 2}:
        return LABEL_MAP_BASELINE
    if sets <= {4, 8, 12}:
        return LABEL_MAP_FIST_IMAGERY
    if sets <= {6, 10, 14}:
        return LABEL_MAP_FEET_IMAGERY
    if sets <= {3, 7, 11}:
        return LABEL_MAP_FIST_EXECUTION
    if sets <= {5, 9, 13}:
        return LABEL_MAP_FEET_EXECUTION
    # Mixed: annotate with run context
    return {
        0: "rest",
        1: "imagery_or_execution_T1",
        2: "imagery_or_execution_T2",
        3: "eyes_open",
        4: "eyes_closed",
    }


def _annotation_to_class_for_runs(runs: list[int]) -> dict[str, int] | None:
    sets = set(runs)
    if sets <= {1, 2}:
        return None
    return {"T0": 0, "T1": 1, "T2": 2}


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    subjects: list[int] | None = None,
    runs: list[int] | None = None,
    preload: bool = True,
    verbose: bool = False,
) -> EEGBundle:
    """Download (first call) and load EEGBCI data.

    Parameters
    ----------
    subjects  : Subject numbers 1–109. Defaults to [1, 2, 3].
    runs      : Run numbers 1–14. Defaults to [4, 8, 12] (left/right fist imagery).
    preload   : If True, load raw data into memory (required for to_windows()).
    verbose   : Forward MNE verbosity.

    Returns
    -------
    EEGBundle with label_map, raws, local_paths, sfreq, channel_names.

    Examples
    --------
    >>> bundle = eegbci.load_data(subjects=[1, 2, 3], runs=[4, 8, 12])
    >>> X, y = bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
    """
    try:
        import mne  # type: ignore[import]
        from mne.datasets import eegbci as mne_eegbci  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "EEGBCI loading requires MNE. Install with: pip install 'qortex[eeg]'"
        ) from None

    if subjects is None:
        subjects = [1, 2, 3]
    if runs is None:
        runs = [4, 8, 12]

    label_map = _label_map_for_runs(runs)
    annotation_to_class = _annotation_to_class_for_runs(runs)
    local_paths: list[Path] = []
    raws = []
    channel_names: list[str] | None = None
    sfreq: float | None = None
    metadata: dict[str, Any] = {}
    failures: list[str] = []

    for subject in subjects:
        try:
            fnames = mne_eegbci.load_data(
                subjects=[subject],
                runs=runs,
                update_path=True,
                verbose=verbose,
            )
        except Exception as exc:
            import warnings
            failures.append(f"subject {subject}: {exc}")
            warnings.warn(f"Could not load subject {subject}: {exc}", RuntimeWarning, stacklevel=2)
            continue

        for fname in fnames:
            p = Path(fname)
            local_paths.append(p)
            try:
                raw = mne.io.read_raw_edf(str(p), preload=preload, verbose=verbose)
                mne.datasets.eegbci.standardize(raw)
                raw.set_montage("standard_1005", verbose=verbose)
                raws.append(raw)
                if channel_names is None:
                    channel_names = list(raw.info.ch_names)
                    sfreq = raw.info["sfreq"]
            except Exception as exc:
                import warnings
                failures.append(f"{p}: {exc}")
                warnings.warn(f"Could not read {p}: {exc}", RuntimeWarning, stacklevel=2)

        metadata[str(subject)] = {"subject": subject, "runs": runs, "n_files": len(fnames)}

    if not raws:
        detail = "; ".join(failures[:5]) or "no files were returned"
        raise RuntimeError(
            "No EEGBCI raw files were loaded. "
            f"Requested subjects={subjects}, runs={runs}. "
            f"Failures: {detail}"
        )

    if sfreq is None:
        sfreq = 160.0
    if channel_names is None:
        channel_names = [f"EEG{i:03d}" for i in range(64)]

    return EEGBundle(
        card=_CARD,
        subjects=subjects,
        runs=runs,
        sfreq=sfreq,
        channel_names=channel_names,
        label_map=label_map,
        local_paths=local_paths,
        raws=raws,
        metadata=metadata,
        annotation_to_class=annotation_to_class,
    )
