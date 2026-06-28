"""Base types for the qortex.datasets module.

Every dataset module exposes two public functions:
  describe() → DatasetCard   (no download, no heavy deps)
  load_data(**kwargs) → *Bundle  (downloads on first call, cached afterwards)

Design contract
---------------
- DatasetCard is frozen and always importable (no optional deps at module level).
- Bundle types hold references to local files and pre-loaded arrays.
- Integrity checks (via qortex.neuroclassic) are available as bundle methods,
  not run automatically — the user can call .run_qc() when needed.
- Label maps are embedded in the bundle so downstream code never needs to
  re-derive them from raw annotation strings.
- All paths are absolute and verified to exist before the bundle is returned.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Dataset card ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetCard:
    """Immutable metadata record for a Qortex dataset.

    Describes the dataset without downloading it.  Suitable for offline
    display, catalogue listings, and tutorial documentation.
    """
    name: str                         # short id used in qortex.datasets.name
    full_name: str                    # human-readable title
    version: str                      # dataset version string
    source_url: str                   # canonical URL (PhysioNet, etc.)
    license: str                      # SPDX identifier or description
    citation: str                     # primary reference (DOI or BibTeX key)
    modality: str                     # "eeg", "mri", "fmri", "eeg_mri"
    n_subjects: int                   # total cohort size
    description: str                  # multi-line description
    tasks: list[str]                  # downstream tasks this dataset suits
    tutorial_ids: list[str]           # which tutorial IDs this enables
    size_gb_approx: float             # approximate full-dataset download size
    requires_registration: bool       # True = user must register first
    access_instructions: str | None   # guidance for registration-gated data
    # Channel / spatial metadata
    n_channels: int | None = None
    sampling_hz: float | None = None
    # Image-specific
    image_shape: tuple | None = None  # typical voxel shape
    n_classes: int | None = None      # for classification/segmentation tasks

    def __str__(self) -> str:
        lines = [
            f"Dataset     : {self.full_name} ({self.name})",
            f"Version     : {self.version}",
            f"Modality    : {self.modality}",
            f"Subjects    : {self.n_subjects}",
            f"Tasks       : {', '.join(self.tasks)}",
            f"Tutorials   : {', '.join(self.tutorial_ids)}",
            f"License     : {self.license}",
            f"Size (~GB)  : {self.size_gb_approx:.1f}",
            f"Source      : {self.source_url}",
        ]
        if self.requires_registration:
            lines.append(f"⚠ Registration required — see access_instructions")
        if self.n_channels:
            lines.append(f"Channels    : {self.n_channels}")
        if self.sampling_hz:
            lines.append(f"Sampling    : {self.sampling_hz} Hz")
        lines.append("")
        lines.append(self.description)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "version": self.version,
            "modality": self.modality,
            "n_subjects": self.n_subjects,
            "tasks": self.tasks,
            "tutorial_ids": self.tutorial_ids,
            "license": self.license,
            "size_gb_approx": self.size_gb_approx,
            "source_url": self.source_url,
            "requires_registration": self.requires_registration,
            "n_channels": self.n_channels,
            "sampling_hz": self.sampling_hz,
        }


# ── EEG bundle ───────────────────────────────────────────────────────────────

@dataclass
class EEGBundle:
    """Loaded EEG dataset.

    Attributes
    ----------
    card         : Dataset metadata.
    subjects     : Subject IDs that were loaded.
    runs         : Run numbers that were loaded (for EEGBCI-style datasets).
    sfreq        : Sampling frequency in Hz.
    channel_names: Channel names (same across all files unless flagged).
    label_map    : {int → str} mapping of integer class codes to semantic names.
    local_paths  : Absolute paths to local EDF/FIF/BDF files.
    raws         : List of mne.io.BaseRaw objects (populated when preload=True).
    epochs       : [n_epochs, n_channels, n_times] float32 array after windowing.
    labels       : [n_epochs] int array after windowing.
    metadata     : Per-subject/session metadata dict.
    qc_report    : Signal QC report from qortex.neuroclassic (None until .run_qc()).
    """
    card: DatasetCard
    subjects: list[int]
    runs: list[int]
    sfreq: float
    channel_names: list[str]
    label_map: dict[int, str]
    local_paths: list[Path]
    raws: list[Any] = field(default_factory=list)
    epochs: "Any | None" = None          # np.ndarray after .to_windows()
    labels: "Any | None" = None          # np.ndarray after .to_windows()
    metadata: dict[str, Any] = field(default_factory=dict)
    qc_report: Any = None

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def n_files(self) -> int:
        return len(self.local_paths)

    def to_windows(
        self,
        window_s: float = 4.0,
        overlap: float = 0.5,
        bandpass: tuple[float, float] | None = None,
        event_driven: bool = True,
        tmin: float = 0.0,
    ) -> "tuple[Any, Any]":
        """Extract fixed-length windows and return (X, y) arrays.

        Parameters
        ----------
        window_s     : Window length in seconds.
        overlap      : Fraction of overlap between successive windows
                       (ignored when event_driven=True).
        bandpass     : Optional (low_hz, high_hz) to filter before windowing.
        event_driven : If True, extract one window per event (for task EEG).
                       If False, use sliding windows with given overlap.
        tmin         : Epoch start relative to event onset (seconds).

        Returns
        -------
        (X, y)
            X : np.ndarray, shape [n_epochs, n_channels, n_samples]
            y : np.ndarray, shape [n_epochs], dtype int
        """
        import numpy as np

        if not self.raws:
            raise RuntimeError(
                "No raw data loaded. Call load_data(preload=True) first, "
                "or use the dataset module's load_data() function."
            )

        try:
            import mne
        except ImportError:
            raise ImportError(
                "to_windows() requires MNE. Install with: pip install 'qortex[eeg]'"
            ) from None

        all_epochs: list[np.ndarray] = []
        all_labels: list[int] = []
        win_samples = int(window_s * self.sfreq)

        for raw in self.raws:
            if bandpass is not None:
                raw = raw.copy().filter(
                    l_freq=bandpass[0], h_freq=bandpass[1], verbose=False
                )

            data = raw.get_data()  # [n_ch, n_times]
            n_ch, n_t = data.shape

            if event_driven and hasattr(raw, "_annotations") and len(raw.annotations) > 0:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                for ev in events:
                    onset_sample = ev[0]
                    label_code = ev[2]
                    if label_code not in self.label_map:
                        continue
                    start = int(onset_sample + tmin * self.sfreq)
                    end = start + win_samples
                    if start < 0 or end > n_t:
                        continue
                    epoch = data[:, start:end].astype(np.float32)
                    all_epochs.append(epoch)
                    # Map raw event code to our label code
                    # label_map keys are sequential 0,1,2,...
                    # find which key maps to this event_id value
                    matched_key = None
                    for k, v in self.label_map.items():
                        if v == self.label_map.get(label_code):
                            matched_key = k
                            break
                    all_labels.append(label_code)
            else:
                # Sliding window
                step = max(1, int(win_samples * (1.0 - overlap)))
                start = 0
                while start + win_samples <= n_t:
                    epoch = data[:, start:start + win_samples].astype(np.float32)
                    all_epochs.append(epoch)
                    all_labels.append(0)  # unlabeled sliding windows
                    start += step

        if not all_epochs:
            import numpy as np
            return np.empty((0, self.n_channels, win_samples), dtype=np.float32), np.empty(0, dtype=np.int64)

        import numpy as np
        X = np.stack(all_epochs, axis=0)
        y = np.array(all_labels, dtype=np.int64)
        self.epochs = X
        self.labels = y
        return X, y

    def run_qc(self, max_files: int = 5) -> Any:
        """Run signal QC on the first max_files files.

        Returns a SignalQualityReport for each file.
        Requires qortex[neuroclassic] extras.
        """
        import numpy as np
        from qortex.neuroclassic import compute_signal_qc

        reports = []
        for i, raw in enumerate(self.raws[:max_files]):
            data = raw.get_data().astype(np.float32)
            rpt = compute_signal_qc(
                data,
                sampling_frequency_hz=self.sfreq,
                channel_names=list(raw.info.ch_names),
                scope=str(self.local_paths[i] if i < len(self.local_paths) else f"file_{i}"),
            )
            reports.append(rpt)
        self.qc_report = reports
        return reports

    def info(self) -> None:
        """Print a summary of the loaded bundle."""
        lines = [
            f"EEGBundle: {self.card.full_name}",
            f"  Subjects     : {self.subjects}",
            f"  Runs         : {self.runs}",
            f"  Files        : {self.n_files}",
            f"  Channels     : {self.n_channels}",
            f"  Sampling Hz  : {self.sfreq}",
            f"  Label map    : {self.label_map}",
        ]
        if self.epochs is not None:
            lines.append(f"  Epochs shape : {self.epochs.shape}")
        if self.labels is not None:
            import numpy as np
            unique, counts = np.unique(self.labels, return_counts=True)
            dist = {self.label_map.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}
            lines.append(f"  Label dist.  : {dist}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        return (
            f"EEGBundle({self.card.name!r}, subjects={self.subjects}, "
            f"runs={self.runs}, n_files={self.n_files}, "
            f"sfreq={self.sfreq} Hz, n_channels={self.n_channels})"
        )


# ── MRI bundle ────────────────────────────────────────────────────────────────

@dataclass
class MRIBundle:
    """Loaded structural MRI dataset (T1, T2, etc.).

    Attributes
    ----------
    card         : Dataset metadata.
    subjects     : Subject identifiers.
    modality     : "T1", "T2", "PD", "DWI", etc.
    local_paths  : Absolute paths to NIfTI files (one per subject).
    metadata_df  : Demographic / clinical table (None until loaded).
    labels       : [n_subjects] array for the primary task (e.g. age or class).
    label_col    : Column name from the metadata table used as label.
    label_map    : {int → str} for classification tasks; None for regression.
    affines      : List of 4×4 affine matrices (None until load_images()).
    images       : List of [x, y, z] arrays (None until load_images()).
    qc_report    : Image QC report (None until .run_qc()).
    """
    card: DatasetCard
    subjects: list[str]
    modality: str
    local_paths: list[Path]
    metadata: dict[str, Any]           # {subject_id → {col → val}}
    labels: "Any | None" = None        # np.ndarray
    label_col: str | None = None
    label_map: dict[int, str] | None = None
    affines: "list[Any] | None" = None
    images: "list[Any] | None" = None
    qc_report: Any = None

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    def load_images(self, max_subjects: int | None = None) -> "list[Any]":
        """Load NIfTI images into memory as numpy arrays.

        Requires nibabel: pip install 'qortex[mri]'

        Returns list of [x, y, z] or [x, y, z, t] float32 arrays.
        """
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            raise ImportError(
                "load_images() requires nibabel. Install with: pip install 'qortex[mri]'"
            ) from None

        paths = self.local_paths if max_subjects is None else self.local_paths[:max_subjects]
        images = []
        affines = []
        for p in paths:
            img = nib.load(str(p))
            img = nib.as_closest_canonical(img)
            data = img.get_fdata(dtype=np.float32)
            images.append(data)
            affines.append(img.affine)
        self.images = images
        self.affines = affines
        return images

    def run_qc(self, max_subjects: int = 5) -> Any:
        """Run image QC on the first max_subjects files.

        Requires nibabel + qortex[neuroclassic].
        """
        import numpy as np
        from qortex.neuroclassic import compute_image_qc

        if self.images is None:
            self.load_images(max_subjects=max_subjects)

        reports = []
        images_to_check = (self.images or [])[:max_subjects]
        affines = (self.affines or [None] * len(images_to_check))[:max_subjects]
        for i, (img, aff) in enumerate(zip(images_to_check, affines)):
            vox = None
            if aff is not None:
                import numpy as np
                vox = tuple(float(abs(aff[j, j])) for j in range(3))
            rpt = compute_image_qc(
                img,
                voxel_sizes_mm=vox,
                affine=aff,
                scope=str(self.local_paths[i] if i < len(self.local_paths) else f"subject_{i}"),
            )
            reports.append(rpt)
        self.qc_report = reports
        return reports

    def info(self) -> None:
        """Print bundle summary."""
        print(f"MRIBundle: {self.card.full_name}")
        print(f"  Modality   : {self.modality}")
        print(f"  Subjects   : {self.n_subjects}")
        print(f"  Label col  : {self.label_col}")
        print(f"  Label map  : {self.label_map}")
        if self.images is not None:
            shapes = [str(im.shape) for im in self.images[:3]]
            print(f"  Image shapes (first 3): {shapes}")

    def __repr__(self) -> str:
        return (
            f"MRIBundle({self.card.name!r}, modality={self.modality!r}, "
            f"n_subjects={self.n_subjects}, label_col={self.label_col!r})"
        )


# ── fMRI bundle ───────────────────────────────────────────────────────────────

@dataclass
class FMRIBundle:
    """Loaded task fMRI dataset in BIDS layout.

    Attributes
    ----------
    card        : Dataset metadata.
    subjects    : Subject IDs.
    task        : BIDS task name.
    tr          : Repetition time in seconds.
    bold_paths  : Per-subject BOLD NIfTI paths.
    event_paths : Per-subject events.tsv paths.
    events      : Parsed event tables (list of dicts-of-lists or DataFrames).
    n_volumes   : Expected volume count per run.
    preflight   : Preflight readiness report (None until .run_preflight()).
    """
    card: DatasetCard
    subjects: list[str]
    task: str
    tr: float
    bold_paths: list[Path]
    event_paths: list[Path]
    events: list[Any] = field(default_factory=list)
    n_volumes: int | None = None
    preflight: Any = None

    def run_preflight(self, dataset_path: Path) -> Any:
        """Run qortex preflight check for fMRI design readiness."""
        from qortex.checks import run_preflight
        report = run_preflight(dataset_path, goal="visualize", modality="fmri")
        self.preflight = report
        return report

    def load_events(self) -> list[dict]:
        """Parse all events.tsv files into list of row-dicts."""
        results = []
        for p in self.event_paths:
            if not p.exists():
                results.append({})
                continue
            rows: list[dict] = []
            with open(p) as fh:
                lines = fh.readlines()
            if not lines:
                results.append({})
                continue
            header = lines[0].strip().split("\t")
            for line in lines[1:]:
                vals = line.strip().split("\t")
                rows.append(dict(zip(header, vals)))
            results.append({"path": str(p), "rows": rows, "n_events": len(rows)})
        self.events = results
        return results

    def info(self) -> None:
        print(f"FMRIBundle: {self.card.full_name}")
        print(f"  Task       : {self.task}")
        print(f"  Subjects   : {len(self.subjects)}")
        print(f"  TR         : {self.tr} s")
        print(f"  BOLD files : {len(self.bold_paths)}")
        print(f"  Event files: {len(self.event_paths)}")

    def __repr__(self) -> str:
        return (
            f"FMRIBundle({self.card.name!r}, task={self.task!r}, "
            f"n_subjects={len(self.subjects)}, tr={self.tr} s)"
        )


# ── Segmentation bundle ───────────────────────────────────────────────────────

@dataclass
class SegmentationBundle:
    """Loaded medical image segmentation dataset.

    Attributes
    ----------
    card         : Dataset metadata.
    case_ids     : Unique case identifiers.
    image_paths  : Per-case image file paths (list of lists for multimodal).
    mask_paths   : Per-case segmentation mask paths.
    label_map    : {int → str} mapping of mask integer values to region names.
    modalities   : List of modality names (e.g. ['FLAIR', 'T1w', 'T1gd', 'T2w']).
    split        : 'train', 'val', 'test', or 'all'.
    """
    card: DatasetCard
    case_ids: list[str]
    image_paths: list[list[Path]]   # [case][modality] → Path
    mask_paths: list[Path]
    label_map: dict[int, str]
    modalities: list[str]
    split: str = "train"

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)

    def load_pair(self, index: int) -> "tuple[Any, Any]":
        """Load one image-mask pair as numpy arrays.

        Returns (image_array, mask_array).
        For multimodal: image_array shape is [n_modalities, x, y, z].
        """
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            raise ImportError(
                "load_pair() requires nibabel. Install with: pip install 'qortex[mri]'"
            ) from None

        imgs = []
        for p in self.image_paths[index]:
            img = nib.load(str(p))
            imgs.append(img.get_fdata(dtype=np.float32))
        image = np.stack(imgs, axis=0) if len(imgs) > 1 else imgs[0]

        mask_img = nib.load(str(self.mask_paths[index]))
        mask = mask_img.get_fdata(dtype=np.float32)
        return image, mask

    def info(self) -> None:
        print(f"SegmentationBundle: {self.card.full_name}")
        print(f"  Split      : {self.split}")
        print(f"  Cases      : {self.n_cases}")
        print(f"  Modalities : {self.modalities}")
        print(f"  Label map  : {self.label_map}")

    def __repr__(self) -> str:
        return (
            f"SegmentationBundle({self.card.name!r}, split={self.split!r}, "
            f"n_cases={self.n_cases}, modalities={self.modalities})"
        )


# ── Dataset registry ──────────────────────────────────────────────────────────

class DatasetRegistry:
    """Global registry of all known dataset cards."""

    def __init__(self) -> None:
        self._cards: dict[str, DatasetCard] = {}

    def register(self, card: DatasetCard) -> None:
        self._cards[card.name] = card

    def get(self, name: str) -> DatasetCard:
        if name not in self._cards:
            raise KeyError(
                f"Dataset '{name}' not found. "
                f"Available: {sorted(self._cards.keys())}"
            )
        return self._cards[name]

    def list_all(self) -> list[DatasetCard]:
        return sorted(self._cards.values(), key=lambda c: c.name)

    def list_by_modality(self, modality: str) -> list[DatasetCard]:
        return [c for c in self._cards.values() if c.modality == modality]

    def list_by_tutorial(self, tutorial_id: str) -> list[DatasetCard]:
        return [c for c in self._cards.values() if tutorial_id in c.tutorial_ids]

    def summary_table(self) -> str:
        """Compact tabular listing of all datasets."""
        header = f"{'Name':<16} {'Modality':<10} {'Subjects':>8} {'Size (GB)':>10} {'Tasks'}"
        sep = "-" * 72
        rows = [header, sep]
        for c in self.list_all():
            tasks = ", ".join(c.tasks[:2])
            rows.append(
                f"{c.name:<16} {c.modality:<10} {c.n_subjects:>8} "
                f"{c.size_gb_approx:>10.1f}  {tasks}"
            )
        return "\n".join(rows)


# Global registry instance
_REGISTRY = DatasetRegistry()
