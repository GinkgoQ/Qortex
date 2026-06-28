"""qortex.datasets.chbmit — CHB-MIT Scalp EEG Seizure Database (PhysioNet).

Usage::

    from qortex.datasets import chbmit

    card = chbmit.describe()
    bundle = chbmit.load_data(cases=["chb01"], seizure_files_only=True)
    X, y = bundle.to_windows(window_s=5.0, event_driven=False)
    # y: {0=non_seizure, 1=seizure}

Dataset facts
-------------
- 23 cases from 22 pediatric subjects (chb01–chb24, chb21 reused).
- 664 EDF files; most: 23 EEG channels at 256 Hz; some files vary.
- Seizure intervals: parsed from chbNN-summary.txt.
- Binary label: seizure=1 if window overlaps any seizure interval, else 0.
- Severe class imbalance: seizure windows are a small minority.
- License: Open Data Commons Attribution License v1.0.
- Source: https://physionet.org/content/chbmit/1.0.0/
- Research only — not for clinical use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, EEGBundle, _REGISTRY
from qortex.datasets._cache import dataset_cache_dir

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="chbmit",
    full_name="CHB-MIT Scalp EEG Seizure Database",
    version="1.0.0",
    source_url="https://physionet.org/content/chbmit/1.0.0/",
    license="Open Data Commons Attribution License v1.0",
    citation="Shoeb AH. Application of Machine Learning to Epileptic Seizure Onset Detection. PhD Thesis, MIT, 2009.",
    modality="eeg",
    n_subjects=23,
    n_channels=23,
    sampling_hz=256.0,
    description=(
        "23 cases of pediatric scalp EEG with manual seizure annotations.\n"
        "664 EDF files (~42.6 GB full dataset); most cases have 23 channels at 256 Hz.\n"
        "Seizure intervals parsed from chbNN-summary.txt files.\n"
        "For research purposes only — not validated for clinical seizure detection."
    ),
    tasks=["seizure_detection", "seizure_onset_segmentation"],
    tutorial_ids=["T04"],
    size_gb_approx=42.6,
    requires_registration=False,
    access_instructions=(
        "Large dataset (~42.6 GB). Load individual cases only.\n"
        "Use seizure_files_only=True to download only seizure-containing files."
    ),
)
_REGISTRY.register(_CARD)


# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "non_seizure", 1: "seizure"}


# ── Seizure interval parser ───────────────────────────────────────────────────

@dataclass
class SeizureInterval:
    """A single seizure interval within an EDF file."""
    file_name: str
    start_sec: float
    end_sec: float
    duration_sec: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_sec = self.end_sec - self.start_sec

    def overlaps_window(self, window_start: float, window_end: float) -> bool:
        """True if the seizure interval overlaps [window_start, window_end)."""
        return self.start_sec < window_end and self.end_sec > window_start


def parse_seizure_summary(summary_path: Path) -> dict[str, list[SeizureInterval]]:
    """Parse a chbNN-summary.txt file into a dict of {filename: [SeizureInterval]}.

    Summary file format (from PhysioNet CHB-MIT documentation):
        File Name: chb01_03.edf
        File Start Time: 14:43:00
        File End Time: 15:43:00
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    """
    result: dict[str, list[SeizureInterval]] = {}
    if not summary_path.exists():
        return result

    text = summary_path.read_text(errors="replace")
    # Split into file blocks
    blocks = re.split(r"\n(?=File Name:)", text.strip())

    for block in blocks:
        file_match = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not file_match:
            continue
        fname = file_match.group(1).strip()

        n_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n_seizures = int(n_match.group(1)) if n_match else 0

        if n_seizures == 0:
            result[fname] = []
            continue

        intervals: list[SeizureInterval] = []
        # Multiple seizures per file use numbered keys
        starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second", block, re.IGNORECASE)
        ends = re.findall(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second", block, re.IGNORECASE)

        for s, e in zip(starts, ends):
            intervals.append(SeizureInterval(fname, float(s), float(e)))
        result[fname] = intervals

    return result


# ── Window labeling ───────────────────────────────────────────────────────────

def label_windows_for_file(
    fname: str,
    n_samples: int,
    sfreq: float,
    window_s: float,
    step_s: float,
    seizure_map: dict[str, list[SeizureInterval]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute window boundaries and binary labels for one EDF file.

    Returns
    -------
    (windows, labels)
        windows : list of (start_sample, end_sample) tuples
        labels  : list of int (0=non_seizure, 1=seizure)
    """
    intervals = seizure_map.get(fname, [])
    win_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    windows: list[tuple[int, int]] = []
    labels: list[int] = []

    start = 0
    while start + win_samples <= n_samples:
        t_start = start / sfreq
        t_end = (start + win_samples) / sfreq
        label = int(any(iv.overlaps_window(t_start, t_end) for iv in intervals))
        windows.append((start, start + win_samples))
        labels.append(label)
        start += step_samples

    return windows, labels


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    cases: list[str] | None = None,
    seizure_files_only: bool = True,
    local_root: Path | str | None = None,
    preload: bool = True,
    verbose: bool = False,
) -> EEGBundle:
    """Load CHB-MIT data from a local directory.

    CHB-MIT must be downloaded separately from PhysioNet (~42.6 GB).
    Provide the root directory containing case subdirectories (chb01/, etc.).

    Parameters
    ----------
    cases           : Case names, e.g. ["chb01", "chb02"]. Defaults to ["chb01"].
    seizure_files_only : If True, only load EDF files that contain seizures
                       (as listed in summary.txt). Reduces download footprint.
    local_root      : Path to the downloaded dataset root.
                      Falls back to QORTEX_DATA_DIR/chbmit or ~/.cache/qortex/datasets/chbmit.
    preload         : Load raw data into memory.
    verbose         : MNE verbosity.

    Returns
    -------
    EEGBundle with label_map = {0: 'non_seizure', 1: 'seizure'}.
    The bundle also carries `.metadata['seizure_map']` for window-level labeling.

    Examples
    --------
    >>> bundle = chbmit.load_data(cases=["chb01"], seizure_files_only=True,
    ...                           local_root="/data/chbmit")
    >>> windows, labels = chbmit.label_windows_for_file(
    ...     "chb01_03.edf", n_samples=..., sfreq=256.0,
    ...     window_s=5.0, step_s=1.0, seizure_map=bundle.metadata["seizure_map"])
    """
    try:
        import mne  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "CHB-MIT loading requires MNE. Install with: pip install 'qortex[eeg]'"
        ) from None

    if cases is None:
        cases = ["chb01"]

    if local_root is None:
        local_root = dataset_cache_dir("chbmit")
    local_root = Path(local_root)

    local_paths: list[Path] = []
    raws = []
    channel_names: list[str] | None = None
    sfreq: float | None = None
    metadata: dict[str, Any] = {}
    all_seizure_maps: dict[str, list[SeizureInterval]] = {}

    for case in cases:
        case_dir = local_root / case
        if not case_dir.exists():
            import warnings
            warnings.warn(
                f"Case directory not found: {case_dir}. "
                f"Download CHB-MIT from https://physionet.org/content/chbmit/1.0.0/ "
                f"and point local_root to the dataset root.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        summary_path = case_dir / f"{case}-summary.txt"
        seizure_map = parse_seizure_summary(summary_path)
        all_seizure_maps.update(seizure_map)

        # Determine which EDF files to load
        edf_files = sorted(case_dir.glob("*.edf"))
        if seizure_files_only:
            edf_files = [f for f in edf_files if f.name in seizure_map and len(seizure_map[f.name]) > 0]

        for edf_path in edf_files:
            local_paths.append(edf_path)
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose=verbose)
                raws.append(raw)
                if channel_names is None:
                    channel_names = list(raw.info.ch_names)
                    sfreq = raw.info["sfreq"]
            except Exception as exc:
                import warnings
                warnings.warn(f"Could not read {edf_path}: {exc}", RuntimeWarning, stacklevel=2)

        metadata[case] = {
            "case": case,
            "n_seizure_files": sum(1 for v in seizure_map.values() if v),
            "total_files": len(edf_files),
            "total_seizures": sum(len(v) for v in seizure_map.values()),
        }

    metadata["seizure_map"] = all_seizure_maps  # type: ignore[assignment]

    if sfreq is None:
        sfreq = 256.0
    if channel_names is None:
        channel_names = [f"EEG{i+1}" for i in range(23)]

    return EEGBundle(
        card=_CARD,
        subjects=list(range(len(cases))),
        runs=[],
        sfreq=sfreq,
        channel_names=channel_names,
        label_map=LABEL_MAP,
        local_paths=local_paths,
        raws=raws,
        metadata=metadata,
    )
