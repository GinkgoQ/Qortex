"""Epoch-level signal Dataset — synchronized BIDS events + continuous recordings.

Provides ``BIDSEpochDataset``: a PyTorch Dataset that reads a continuous BIDS
electrophysiology recording and its paired ``*_events.tsv`` file, then yields
discrete time-locked epochs of shape ``(n_channels, n_times)``.

Also provides ``TorchEEGBridge``: generates datasets compatible with the
TorchEEG library's ``BaseDataset`` convention (``{eeg: tensor, label: int}``).

Design principles
-----------------
* All epoch slicing is done on already-loaded MNE ``Raw`` objects; data is read
  once per file and reused across all epochs in that file.
* Label encoding mirrors ``BIDSSignalDataset``: string trial_types → contiguous
  ints; numeric values are returned as-is.
* Baseline correction, band-pass filtering, and channel selection can be
  composed via the ``transform`` callable.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class _EventMarker:
    onset_s: float
    duration_s: float
    trial_type: str
    subject: str
    run: str | None


def _parse_events_tsv(tsv_path: Path, trial_type_col: str = "trial_type") -> list[_EventMarker]:
    """Parse a BIDS events.tsv and return a list of EventMarkers."""
    markers: list[_EventMarker] = []
    if not tsv_path.is_file():
        return markers
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for row in rows:
        onset_raw = row.get("onset", "")
        dur_raw = row.get("duration", "0")
        tt = row.get(trial_type_col, "").strip()
        try:
            onset = float(onset_raw)
        except (ValueError, TypeError):
            continue
        try:
            dur = float(dur_raw)
        except (ValueError, TypeError):
            dur = 0.0
        markers.append(_EventMarker(
            onset_s=onset,
            duration_s=dur,
            trial_type=tt,
            subject="",
            run=None,
        ))
    return markers


def _encode_trial_types(markers: list[_EventMarker]) -> dict[str, int]:
    unique = sorted({m.trial_type for m in markers if m.trial_type})
    return {tt: i for i, tt in enumerate(unique)}


class BIDSEpochDataset:
    """Event-locked epoch Dataset from BIDS electrophysiology files.

    Reads each subject's signal file once using MNE, then slices out all
    valid epochs based on the paired ``*_events.tsv``.

    Parameters
    ----------
    bids_root:
        Root of the downloaded BIDS dataset.
    modality:
        ``"eeg"`` | ``"meg"`` | ``"ieeg"``
    epoch_duration_s:
        Length of each epoch in seconds.
    tmin:
        Seconds before event onset to include (negative = pre-stimulus baseline).
    event_id:
        Optional trial_type filter; if given, only epochs matching this
        trial_type are included.  ``None`` = all trial_types.
    trial_type_col:
        Column in events.tsv to use as label (default ``"trial_type"``).
    baseline:
        Optional ``(tmin_baseline, tmax_baseline)`` in seconds for baseline
        correction.  ``None`` = no baseline.
    resample_hz:
        Resample all recordings to a common rate before epoch extraction.
    picks:
        MNE channel selection string or list (e.g. ``"eeg"`` or ``["Cz", "Pz"]``).
    transform:
        Optional callable applied to each epoch dict.
    subjects:
        Optional list of subject IDs to include.

    Returns (``__getitem__``)
    -------------------------
    dict with keys:
        ``"eeg"`` or ``"meg"``   : float32 ndarray shape (n_channels, n_times)
        ``"label"``              : int
        ``"trial_type"``         : str
        ``"onset_s"``            : float
        ``"subject"``            : str
    """

    def __init__(
        self,
        bids_root: Path,
        modality: str = "eeg",
        *,
        epoch_duration_s: float = 1.0,
        tmin: float = 0.0,
        event_id: str | list[str] | None = None,
        trial_type_col: str = "trial_type",
        baseline: tuple[float, float] | None = None,
        resample_hz: float | None = None,
        picks: str | list[str] = "all",
        transform: Callable | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        self.modality = modality
        self.epoch_duration_s = epoch_duration_s
        self.tmin = tmin
        self.event_ids = {event_id} if isinstance(event_id, str) else set(event_id or [])
        self.trial_type_col = trial_type_col
        self.baseline = baseline
        self.resample_hz = resample_hz
        self.picks = picks
        self.transform = transform

        _EXTS: dict[str, list[str]] = {
            "eeg":  [".edf", ".bdf", ".set", ".vhdr", ".fif"],
            "meg":  [".fif", ".ds", ".sqd", ".con"],
            "ieeg": [".edf", ".bdf"],
        }
        signal_exts = _EXTS.get(modality, [".edf"])

        # Discover all (signal_path, events_path) pairs
        all_subs = sorted({
            p.name for p in self.bids_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })
        if subjects:
            sub_set = {s if s.startswith("sub-") else f"sub-{s}" for s in subjects}
            all_subs = [s for s in all_subs if s in sub_set]

        # Build epoch index
        raw_pairs: list[tuple[Path, Path]] = []
        for sub in all_subs:
            sub_dir = self.bids_root / sub
            for search_root in [sub_dir, *sorted(sub_dir.glob("ses-*"))]:
                mod_dir = search_root / modality
                if not mod_dir.is_dir():
                    continue
                for ext in signal_exts:
                    for sig_path in sorted(mod_dir.glob(f"*{ext}")):
                        # Find paired events TSV
                        stem = sig_path.name
                        for chk_ext in signal_exts:
                            stem = stem.removesuffix(chk_ext)
                        # Strip suffix entity to find events file
                        parts = stem.rsplit("_", 1)
                        base = parts[0] if len(parts) > 1 else stem
                        events_path = mod_dir / f"{base}_events.tsv"
                        if events_path.is_file():
                            raw_pairs.append((sig_path, events_path))
                            break
                        # Broader fallback
                        fallbacks = sorted(mod_dir.glob("*_events.tsv"))
                        if fallbacks:
                            raw_pairs.append((sig_path, fallbacks[0]))

        # Load all events to build the epoch index
        self._epochs: list[dict[str, Any]] = []
        all_trial_types: list[str] = []

        for sig_path, evt_path in raw_pairs:
            markers = _parse_events_tsv(evt_path, trial_type_col)
            sub = next(
                (part for part in sig_path.parts if part.startswith("sub-")), ""
            )
            for m in markers:
                if self.event_ids and m.trial_type not in self.event_ids:
                    continue
                all_trial_types.append(m.trial_type)
                self._epochs.append({
                    "signal_path": sig_path,
                    "subject": sub,
                    "onset_s": m.onset_s,
                    "trial_type": m.trial_type,
                    "label": -1,  # filled after encoding
                })

        # Encode labels
        self._trial_type_map: dict[str, int] = _encode_trial_types([
            _EventMarker(onset_s=0, duration_s=0, trial_type=tt, subject="", run=None)
            for tt in all_trial_types
        ])
        for ep in self._epochs:
            ep["label"] = self._trial_type_map.get(ep["trial_type"], -1)

        # Pre-load MNE Raw objects (one per unique signal file) to avoid reloading
        self._raw_cache: dict[str, Any] = {}
        self._load_all_raws(sig_paths={e["signal_path"] for e in self._epochs})

        log.info(
            "BIDSEpochDataset: %d epochs from %d files, %d classes",
            len(self._epochs), len(raw_pairs),
            len(self._trial_type_map),
        )

    def __len__(self) -> int:
        return len(self._epochs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep = self._epochs[idx]
        sig_path = str(ep["signal_path"])
        raw = self._raw_cache.get(sig_path)
        if raw is None:
            return {
                self.modality: np.zeros((1, 1), dtype=np.float32),
                "label": -1,
                "trial_type": ep["trial_type"],
                "onset_s": ep["onset_s"],
                "subject": ep["subject"],
                "error": "raw not loaded",
            }

        tmin_abs = ep["onset_s"] + self.tmin
        tmax_abs = tmin_abs + self.epoch_duration_s

        # Guard against out-of-bounds
        rec_end = raw.times[-1]
        if tmin_abs < 0 or tmax_abs > rec_end + 0.001:
            return {
                self.modality: np.zeros((1, 1), dtype=np.float32),
                "label": -1,
                "trial_type": ep["trial_type"],
                "onset_s": ep["onset_s"],
                "subject": ep["subject"],
                "error": "epoch out of bounds",
            }

        try:
            import mne
            data, _ = raw.get_data(
                picks=self.picks,
                tmin=tmin_abs,
                tmax=tmax_abs,
                return_times=True,
            )
            data = data.astype(np.float32)
        except Exception as exc:
            log.debug("Epoch extraction failed for %s: %s", sig_path, exc)
            return {
                self.modality: np.zeros((1, 1), dtype=np.float32),
                "label": -1,
                "trial_type": ep["trial_type"],
                "onset_s": ep["onset_s"],
                "subject": ep["subject"],
                "error": str(exc),
            }

        if self.baseline is not None:
            b_tmin, b_tmax = self.baseline
            b_start = int((b_tmin - self.tmin) * raw.info["sfreq"])
            b_end   = int((b_tmax - self.tmin) * raw.info["sfreq"])
            if 0 <= b_start < b_end <= data.shape[1]:
                baseline_mean = data[:, b_start:b_end].mean(axis=1, keepdims=True)
                data -= baseline_mean

        sample: dict[str, Any] = {
            self.modality: data,
            "label": ep["label"],
            "trial_type": ep["trial_type"],
            "onset_s": ep["onset_s"],
            "subject": ep["subject"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def trial_type_classes(self) -> dict[str, int]:
        return dict(self._trial_type_map)

    def _load_all_raws(self, sig_paths: set[Path]) -> None:
        try:
            import mne
        except ImportError:
            log.error("MNE is required for BIDSEpochDataset: pip install 'qortex[signal]'")
            return

        for path in sorted(sig_paths):
            try:
                ext = path.suffix.lower()
                if ext == ".fif":
                    raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
                elif ext in (".edf", ".bdf"):
                    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
                elif ext == ".set":
                    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
                elif ext == ".vhdr":
                    raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
                else:
                    raw = mne.io.read_raw(str(path), preload=True, verbose=False)

                if self.resample_hz and raw.info["sfreq"] != self.resample_hz:
                    raw.resample(self.resample_hz, verbose=False)

                self._raw_cache[str(path)] = raw
            except Exception as exc:
                log.warning("Cannot load %s: %s", path.name, exc)


class TorchEEGBridge:
    """Adapter that converts a ``BIDSEpochDataset`` into TorchEEG's expected format.

    TorchEEG datasets return ``{"eeg": Tensor(shape), "label": int}`` dicts.
    This bridge repackages ``BIDSEpochDataset`` output to match that interface.

    Parameters
    ----------
    epoch_dataset:
        A configured ``BIDSEpochDataset`` instance.
    stack_to_grid:
        When True, reshape ``(n_channels, n_times)`` to a 2D electrode grid
        using ``grid_size`` (default ``(9, 9)`` — standard 10-20 layout).
        Only meaningful for EEG datasets with standard channel layouts.
    grid_size:
        Target ``(rows, cols)`` for electrode grid reshaping.

    Returns (``__getitem__``)
    -------------------------
    dict: ``{"eeg": FloatTensor, "label": int}``
    """

    def __init__(
        self,
        epoch_dataset: BIDSEpochDataset,
        *,
        stack_to_grid: bool = False,
        grid_size: tuple[int, int] = (9, 9),
    ) -> None:
        self._ds = epoch_dataset
        self.stack_to_grid = stack_to_grid
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._ds[idx]
        modality = self._ds.modality
        signal = sample.get(modality, np.zeros((1, 1), dtype=np.float32))

        if self.stack_to_grid and signal.ndim == 2:
            r, c = self.grid_size
            n_ch, n_t = signal.shape
            grid = np.zeros((r * c, n_t), dtype=np.float32)
            grid[:min(n_ch, r * c), :] = signal[:min(n_ch, r * c), :]
            signal = grid.reshape(r, c, n_t)

        try:
            import torch
            signal_t = torch.from_numpy(signal)
        except ImportError:
            signal_t = signal

        return {"eeg": signal_t, "label": sample["label"]}

    @property
    def trial_type_classes(self) -> dict[str, int]:
        return self._ds.trial_type_classes
