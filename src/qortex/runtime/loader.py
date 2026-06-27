"""Native BIDS-to-PyTorch/MONAI dataset implementations.

Provides map-style ``torch.utils.data.Dataset`` subclasses that load NIfTI
volumes and signal files directly from a local BIDS tree — no intermediate
Parquet conversion required.

Design choices
--------------
* **Nibabel-backed**: volumes are loaded via nibabel's ``ArrayProxy`` (lazy
  memory mapping) so the Dataset constructor is O(1).  Actual data is read
  during ``__getitem__``.
* **Label injection**: demographics and classification targets are read once
  from ``participants.tsv`` on construction and stored in a lookup dict.
* **Transform-composable**: accepts any callable ``transform(sample: dict) → dict``
  compatible with MONAI's ``Compose`` or TorchIO's ``Compose``.
* **Multimodal**: supports multiple NIfTI suffixes per sample (T1w + T2w),
  returning a stacked tensor or a named dict of separate tensors.
* **MONAI-compatible output**: ``MONAIDictBuilder`` converts a BIDS root into
  the ``[{"image": path, "label": val}, ...]`` datalist expected by
  ``monai.data.CacheDataset``.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_participants(bids_root: Path, label_column: str | None) -> dict[str, Any]:
    """Parse participants.tsv and return {sub_id: label_value} (or {sub_id: None})."""
    tsv = bids_root / "participants.tsv"
    if not tsv.is_file():
        return {}
    with open(tsv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    result: dict[str, Any] = {}
    for row in rows:
        pid = row.get("participant_id", "").strip()
        if not pid:
            continue
        if not pid.startswith("sub-"):
            pid = f"sub-{pid}"
        val = row.get(label_column or "__none__", None) if label_column else None
        if val is not None:
            val = val.strip()
            if val.lower() in ("n/a", "na", "nan", ""):
                val = None
        result[pid] = val
    return result


def _find_nifti(
    sub_dir: Path,
    datatype: str,
    suffix: str,
    extension: str,
) -> Path | None:
    """Find the first NIfTI file matching (datatype, suffix) under a subject dir."""
    for root in [sub_dir, *sorted(sub_dir.glob("ses-*"))]:
        dt = root / datatype
        if dt.is_dir():
            candidates = sorted(dt.glob(f"*_{suffix}{extension}"))
            if candidates:
                return candidates[0]
    return None


def _encode_labels(labels: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Build {sub_id: int} + {str_val: int} encoding for string labels."""
    str_vals = sorted({
        str(v) for v in labels.values()
        if v is not None and not _is_numeric(v)
    })
    if not str_vals:
        return labels, {}
    encoding = {v: i for i, v in enumerate(str_vals)}
    encoded = {
        k: encoding.get(str(v), -1) if not _is_numeric(v) else _safe_num(v)
        for k, v in labels.items()
    }
    return encoded, encoding


def _is_numeric(v: Any) -> bool:
    if v is None:
        return False
    try:
        float(str(v))
        return True
    except (ValueError, TypeError):
        return False


def _safe_num(v: Any) -> float | None:
    try:
        return float(str(v))
    except (ValueError, TypeError):
        return None


# ── BIDSImageDataset ──────────────────────────────────────────────────────────

class BIDSImageDataset:
    """Map-style PyTorch Dataset for volumetric NIfTI images in a local BIDS tree.

    Parameters
    ----------
    bids_root:
        Root directory of the downloaded BIDS dataset.
    suffix:
        BIDS suffix to load, e.g. ``"T1w"``, or a list of suffixes for
        multi-channel inputs: ``["T1w", "T2w"]``.
    datatype:
        BIDS datatype folder, e.g. ``"anat"`` (default) or ``"func"``.
    extension:
        File extension, default ``".nii.gz"``.
    label_column:
        Column in ``participants.tsv`` to use as classification target.
        Categorical values are encoded to integers.
    label_map:
        Optional explicit string → int mapping.  Overrides auto-encoding.
    transform:
        Optional callable applied to the sample dict in ``__getitem__``.
        Receives ``{"image": np.ndarray, "label": int, "subject": str, ...}``.
        Compatible with MONAI's ``Compose`` and TorchIO's ``Compose``.
    require_all_suffixes:
        When multiple suffixes are given: if True (default), skip subjects
        missing any suffix.  If False, fill missing channels with zeros.
    canonical:
        Reorient loaded volumes to RAS orientation via nibabel (default True).
    include_metadata:
        Include sidecar JSON fields in the returned sample dict (default True).
    subjects:
        Optional explicit list of subject IDs to include.

    Returns (in ``__getitem__``)
    ----------------------------
    dict with keys:
        ``"image"`` : float32 np.ndarray of shape (C, X, Y, Z) — C=1 for single suffix
        ``"label"`` : int or float, -1 if missing
        ``"subject"``: BIDS sub-ID string
        ``"path"``  : absolute path to the primary image
        + any sidecar metadata keys if ``include_metadata=True``
    """

    def __init__(
        self,
        bids_root: Path,
        suffix: str | list[str] = "T1w",
        *,
        datatype: str = "anat",
        extension: str = ".nii.gz",
        label_column: str | None = None,
        label_map: dict[str, int] | None = None,
        transform: Callable | None = None,
        require_all_suffixes: bool = True,
        canonical: bool = True,
        include_metadata: bool = True,
        subjects: list[str] | None = None,
        derivatives_pipeline: str | None = None,
    ) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        self.suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
        self.datatype = datatype
        self.extension = extension
        self.label_column = label_column
        self.transform = transform
        self.canonical = canonical
        self.include_metadata = include_metadata
        self.derivatives_pipeline = derivatives_pipeline

        search_root = (
            self.bids_root / "derivatives" / derivatives_pipeline
            if derivatives_pipeline else self.bids_root
        )

        # Discover subjects
        all_subs = sorted({
            p.name for p in search_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })
        if subjects:
            sub_set = {s if s.startswith("sub-") else f"sub-{s}" for s in subjects}
            all_subs = [s for s in all_subs if s in sub_set]

        # Load labels from participants.tsv
        raw_labels = _load_participants(self.bids_root, label_column)
        if label_map:
            self._label_map = label_map
            self._labels = {
                k: label_map.get(str(v), -1) if v is not None else -1
                for k, v in raw_labels.items()
            }
        else:
            self._labels, self._label_map = _encode_labels(raw_labels)

        # Build index of valid (subject, paths) tuples
        self._index: list[dict[str, Any]] = []
        missing = 0
        for sub in all_subs:
            sub_dir = search_root / sub
            paths: list[Path] = []
            all_found = True
            for sfx in self.suffixes:
                p = _find_nifti(sub_dir, datatype, sfx, extension)
                if p is None:
                    if require_all_suffixes:
                        all_found = False
                        break
                    paths.append(None)
                else:
                    paths.append(p)
            if not all_found:
                missing += 1
                continue
            self._index.append({
                "subject": sub,
                "paths": paths,
                "label": self._labels.get(sub, -1),
            })

        if missing:
            log.warning(
                "%d/%d subjects excluded (missing required suffix %s)",
                missing, len(all_subs), self.suffixes,
            )
        log.info(
            "BIDSImageDataset: %d subjects, %d channels, label=%r",
            len(self._index), len(self.suffixes), label_column,
        )

    # ── torch.utils.data.Dataset interface ───────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._index[idx]
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "BIDSImageDataset requires nibabel: pip install 'qortex[mri]'"
            )

        volumes: list[np.ndarray] = []
        primary_path = entry["paths"][0]
        meta: dict[str, Any] = {}

        for path in entry["paths"]:
            if path is None:
                # Missing channel — fill with zeros using first volume's shape
                if volumes:
                    volumes.append(np.zeros_like(volumes[0]))
                continue
            img = nib.load(str(path))
            if self.canonical:
                img = nib.as_closest_canonical(img)
            arr = img.get_fdata(dtype=np.float32)
            if arr.ndim == 4:
                arr = arr[..., 0]  # take first volume of 4D (fMRI → T1w slot)
            volumes.append(arr)

            if self.include_metadata:
                from qortex.parse._mne_utils import load_json_sidecar
                try:
                    sc = load_json_sidecar(path)
                    meta.update(sc)
                except Exception:
                    pass

        # Stack channels along axis 0: (C, X, Y, Z)
        if len(volumes) == 1:
            image = volumes[0][np.newaxis, ...]
        else:
            image = np.stack(volumes, axis=0)

        sample: dict[str, Any] = {
            "image": image,
            "label": entry["label"] if entry["label"] is not None else -1,
            "subject": entry["subject"],
            "path": str(primary_path),
            **meta,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def label_classes(self) -> dict[str, int]:
        return dict(self._label_map)

    @property
    def subjects(self) -> list[str]:
        return [e["subject"] for e in self._index]

    def to_dataloader(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Return a ``torch.utils.data.DataLoader`` wrapping this dataset."""
        try:
            import torch.utils.data as td
        except ImportError:
            raise ImportError("to_dataloader() requires PyTorch: pip install torch")
        return td.DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, **kwargs)


# ── BIDSSignalDataset ─────────────────────────────────────────────────────────

class BIDSSignalDataset:
    """Map-style PyTorch Dataset for electrophysiology recordings in a BIDS tree.

    Loads MNE-compatible signal files (EEG, MEG, iEEG) and returns per-recording
    signal arrays paired with labels.  For epoch-level windowing, see
    ``BIDSEpochDataset`` in ``qortex.runtime.epochs``.

    Parameters
    ----------
    bids_root:
        Root of the downloaded BIDS dataset.
    modality:
        ``"eeg"`` | ``"meg"`` | ``"ieeg"`` | ``"fnirs"``
    label_column:
        Column in ``participants.tsv`` (classification target) or
        ``events.tsv`` column for trial-level labels.
    label_source:
        ``"participants"`` (subject-level, default) or ``"events"``
        (trial-level — returns the most common trial_type per recording).
    max_duration_s:
        Truncate recordings to this duration.  None = no truncation.
    resample_hz:
        Resample all recordings to a common sampling rate.  None = native rate.
    transform:
        Callable applied to each sample dict.
    """

    _MODALITY_EXTENSIONS: dict[str, list[str]] = {
        "eeg":   [".edf", ".bdf", ".set", ".vhdr", ".fif"],
        "meg":   [".fif", ".ds", ".sqd", ".con", ".4d", ".mef"],
        "ieeg":  [".edf", ".bdf", ".nwb", ".eeg"],
        "fnirs": [".snirf"],
    }

    def __init__(
        self,
        bids_root: Path,
        modality: str = "eeg",
        *,
        label_column: str | None = None,
        label_source: str = "participants",
        max_duration_s: float | None = None,
        resample_hz: float | None = None,
        transform: Callable | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        self.modality = modality
        self.label_column = label_column
        self.label_source = label_source
        self.max_duration_s = max_duration_s
        self.resample_hz = resample_hz
        self.transform = transform

        extensions = self._MODALITY_EXTENSIONS.get(modality, [".edf"])

        # Load labels
        raw_labels: dict[str, Any] = {}
        if label_source == "participants" and label_column:
            raw_labels = _load_participants(self.bids_root, label_column)
        self._labels, self._label_map = _encode_labels(raw_labels)

        # Discover signal files
        all_subs = sorted({
            p.name for p in self.bids_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })
        if subjects:
            sub_set = {s if s.startswith("sub-") else f"sub-{s}" for s in subjects}
            all_subs = [s for s in all_subs if s in sub_set]

        self._index: list[dict[str, Any]] = []
        for sub in all_subs:
            sub_dir = self.bids_root / sub
            paths = self._find_signal_files(sub_dir, modality, extensions)
            for path in paths:
                self._index.append({
                    "subject": sub,
                    "path": path,
                    "label": self._labels.get(sub, -1),
                })

        log.info(
            "BIDSSignalDataset: %d recordings for modality=%r",
            len(self._index), modality,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._index[idx]
        try:
            import mne
        except ImportError:
            raise ImportError(
                "BIDSSignalDataset requires MNE: pip install 'qortex[signal]'"
            )

        path = entry["path"]
        ext = path.suffix.lower()
        raw: Any = None
        try:
            if ext == ".fif":
                raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
            elif ext in (".edf", ".bdf"):
                raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
            elif ext == ".set":
                raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
            elif ext == ".vhdr":
                raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
            elif ext == ".snirf":
                raw = mne.io.read_raw_snirf(str(path), preload=True, verbose=False)
            else:
                raw = mne.io.read_raw(str(path), preload=True, verbose=False)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return {"signal": np.zeros((1, 1), dtype=np.float32), "label": -1, "subject": entry["subject"], "path": str(path), "error": str(exc)}

        if self.resample_hz and raw.info["sfreq"] != self.resample_hz:
            raw.resample(self.resample_hz, verbose=False)

        if self.max_duration_s:
            tmax = min(self.max_duration_s, raw.times[-1])
            raw.crop(tmax=tmax)

        data = raw.get_data().astype(np.float32)
        sample: dict[str, Any] = {
            "signal": data,
            "label": entry["label"] if entry["label"] is not None else -1,
            "subject": entry["subject"],
            "path": str(path),
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def label_classes(self) -> dict[str, int]:
        return dict(self._label_map)

    def to_dataloader(self, batch_size: int = 8, num_workers: int = 0, **kwargs: Any) -> Any:
        try:
            import torch.utils.data as td
        except ImportError:
            raise ImportError("Requires PyTorch: pip install torch")
        return td.DataLoader(self, batch_size=batch_size, num_workers=num_workers, **kwargs)

    def _find_signal_files(
        self,
        sub_dir: Path,
        modality: str,
        extensions: list[str],
    ) -> list[Path]:
        found: list[Path] = []
        search_roots = [sub_dir] + sorted(sub_dir.glob("ses-*"))
        for root in search_roots:
            mod_dir = root / modality
            if not mod_dir.is_dir():
                continue
            for ext in extensions:
                found.extend(sorted(mod_dir.glob(f"*{ext}")))
        return found


# ── MONAIDictBuilder ─────────────────────────────────────────────────────────

class MONAIDictBuilder:
    """Convert a BIDS dataset into MONAI's list-of-dicts datalist format.

    MONAI requires::

        [
            {"image": "/path/T1w.nii.gz", "label": 0},
            {"image": ["/path/T1w.nii.gz", "/path/T2w.nii.gz"], "label": 1},
            ...
        ]

    This builder handles multi-modal inputs, optional segmentation masks,
    split assignment, and JSON sidecar metadata injection.

    Parameters
    ----------
    bids_root:
        Root of the BIDS dataset.
    image_keys:
        ``{"suffix": "image_key"}`` mapping, e.g.
        ``{"T1w": "image", "T2w": "image2"}``.
        When all values map to ``"image"``, images are stacked into a list
        (multi-channel MONAI convention).

    Examples
    --------
    >>> builder = MONAIDictBuilder(bids_root, {"T1w": "image", "T2w": "image2"})
    >>> datalist = builder.build(label_column="diagnosis")
    >>> ds = monai.data.CacheDataset(data=datalist, transform=transforms)
    """

    def __init__(
        self,
        bids_root: Path,
        image_keys: dict[str, str] | None = None,
        datatype: str = "anat",
        extension: str = ".nii.gz",
    ) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        self.image_keys = image_keys or {"T1w": "image"}
        self.datatype = datatype
        self.extension = extension

    def build(
        self,
        *,
        label_column: str | None = None,
        label_map: dict[str, int] | None = None,
        seg_suffix: str | None = None,
        seg_key: str = "label",
        include_metadata: bool = False,
        use_absolute_paths: bool = True,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build the MONAI datalist.

        Returns
        -------
        dict
            Keys: ``"training"``, ``"validation"``, ``"test"`` each mapping
            to a list of sample dicts.
        """
        raw_labels = _load_participants(self.bids_root, label_column)
        encoded_labels, encoding_map = _encode_labels(raw_labels)
        if label_map:
            encoding_map = label_map
            encoded_labels = {
                k: label_map.get(str(v), -1) if v is not None else -1
                for k, v in raw_labels.items()
            }

        all_subs = sorted({
            p.name for p in self.bids_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })

        samples: list[dict[str, Any]] = []
        for sub in all_subs:
            sub_dir = self.bids_root / sub
            entry: dict[str, Any] = {}
            missing = False

            # Collect images per key
            key_paths: dict[str, list[Path]] = {}
            for suffix, key in self.image_keys.items():
                p = _find_nifti(sub_dir, self.datatype, suffix, self.extension)
                if p is None:
                    missing = True
                    break
                pstr = str(p) if use_absolute_paths else str(p.relative_to(self.bids_root))
                key_paths.setdefault(key, []).append(pstr)

            if missing:
                continue

            for key, paths in key_paths.items():
                entry[key] = paths[0] if len(paths) == 1 else paths

            # Segmentation mask
            if seg_suffix:
                seg_p = _find_nifti(sub_dir, self.datatype, seg_suffix, self.extension)
                if seg_p:
                    entry[seg_key] = str(seg_p) if use_absolute_paths else str(
                        seg_p.relative_to(self.bids_root)
                    )

            # Classification label
            if label_column and seg_suffix is None and "label" not in entry:
                entry["label"] = encoded_labels.get(sub, -1)

            entry["subject_id"] = sub
            samples.append(entry)

        # Split
        import hashlib
        n = len(samples)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        shuffled = sorted(
            samples,
            key=lambda s: hashlib.sha256(f"{seed}:{s['subject_id']}".encode()).hexdigest(),
        )
        return {
            "training":   shuffled[:n_train],
            "validation": shuffled[n_train: n_train + n_val],
            "test":       shuffled[n_train + n_val:],
            "label_classes": {str(v): k for k, v in encoding_map.items()},
        }
