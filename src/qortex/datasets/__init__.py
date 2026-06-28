"""qortex.datasets — Keras-style neuroscience dataset loaders.

Each dataset module provides:
  - ``describe()``   → DatasetCard (no download, no heavy deps)
  - ``load_data()``  → typed Bundle (EEGBundle / MRIBundle / FMRIBundle / SegmentationBundle)

Quick start::

    from qortex.datasets import eegbci
    bundle = eegbci.load_data(subjects=[1, 2, 3], runs=[4, 8, 12])
    X, y = bundle.to_windows(window_s=4.0, bandpass=(8.0, 30.0))
    bundle.info()

    from qortex.datasets import sleep_edf
    bundle = sleep_edf.load_data(subjects=[0, 1, 2])
    X, y = bundle.to_windows(window_s=30.0, event_driven=False)

    from qortex.datasets import oasis1
    bundle = oasis1.load_data(local_root="/data/oasis1")
    # label_map = {0: "no_dementia", 1: "dementia"}

    from qortex.datasets import msd_brain
    bundle = msd_brain.load_data(split="train", max_cases=20)
    image, mask = bundle.load_pair(0)

Catalogue API::

    qortex.datasets.list_available()        # list all DatasetCards
    qortex.datasets.describe("eegbci")      # single DatasetCard
    qortex.datasets.load_dataset("eegbci", subjects=[1], runs=[4, 8, 12])

Available datasets
------------------
eegbci      PhysioNet EEG Motor Movement/Imagery (T01, T02)
sleep_edf   Sleep-EDF Expanded — sleep stage classification (T03)
chbmit      CHB-MIT Scalp EEG Seizure Database (T04)
oasis1      OASIS-1 Structural MRI — dementia research (T05)
ixi         IXI Multimodal MRI — age regression / sex classification (T06)
ds000001    OpenfMRI BART task fMRI — event design validation (T07)
msd_brain   MSD Brain Tumour segmentation (T08)
"""

from __future__ import annotations

from qortex.datasets._base import (
    DatasetCard,
    DatasetRegistry,
    EEGBundle,
    FMRIBundle,
    MRIBundle,
    SegmentationBundle,
    _REGISTRY,
)

# ── Lazy dataset module imports ───────────────────────────────────────────────
# Each sub-module registers its CARD in _REGISTRY on import.

from qortex.datasets import (
    chbmit,
    ds000001,
    eegbci,
    ixi,
    msd_brain,
    oasis1,
    sleep_edf,
)


# ── Catalogue functions ────────────────────────────────────────────────────────

def list_available() -> list[DatasetCard]:
    """Return all registered DatasetCards, sorted by name."""
    return _REGISTRY.list_all()


def describe(name: str) -> DatasetCard:
    """Return the DatasetCard for a dataset by name.

    Parameters
    ----------
    name : Dataset identifier, e.g. "eegbci", "sleep_edf", "oasis1".

    Raises
    ------
    KeyError : If the dataset name is not registered.

    Examples
    --------
    >>> card = qortex.datasets.describe("eegbci")
    >>> print(card.n_subjects, card.sampling_hz)
    """
    return _REGISTRY.get(name)


def load_dataset(name: str, **kwargs) -> "EEGBundle | MRIBundle | FMRIBundle | SegmentationBundle":
    """Load a dataset by name with keyword arguments forwarded to load_data().

    Parameters
    ----------
    name   : Dataset identifier (same as used in ``describe()``).
    **kwargs : Forwarded verbatim to the dataset module's ``load_data()`` function.

    Returns
    -------
    A typed Bundle for the requested dataset.

    Examples
    --------
    >>> bundle = qortex.datasets.load_dataset("eegbci", subjects=[1, 2], runs=[4, 8, 12])
    >>> bundle = qortex.datasets.load_dataset("sleep_edf", subjects=[0, 1, 2])
    >>> bundle = qortex.datasets.load_dataset("msd_brain", max_cases=10)
    """
    _modules = {
        "eegbci": eegbci,
        "sleep_edf": sleep_edf,
        "chbmit": chbmit,
        "oasis1": oasis1,
        "ixi": ixi,
        "ds000001": ds000001,
        "msd_brain": msd_brain,
    }
    if name not in _modules:
        available = sorted(_modules.keys())
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return _modules[name].load_data(**kwargs)


def summary() -> str:
    """Print a compact table of all available datasets.

    Examples
    --------
    >>> print(qortex.datasets.summary())
    """
    return _REGISTRY.summary_table()


__all__ = [
    # Bundle types
    "DatasetCard",
    "DatasetRegistry",
    "EEGBundle",
    "FMRIBundle",
    "MRIBundle",
    "SegmentationBundle",
    # Catalogue
    "describe",
    "list_available",
    "load_dataset",
    "summary",
    # Dataset modules (importable as qortex.datasets.eegbci etc.)
    "chbmit",
    "ds000001",
    "eegbci",
    "ixi",
    "msd_brain",
    "oasis1",
    "sleep_edf",
]
