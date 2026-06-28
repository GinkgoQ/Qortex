"""Dataset-level cache directory management for qortex.datasets.

Uses platformdirs (same standard as pip/poetry) when available;
falls back to ~/.cache/qortex/datasets.
"""

from __future__ import annotations

import os
from pathlib import Path


def dataset_cache_dir(dataset_name: str) -> Path:
    """Return (and create) the cache directory for a named dataset.

    Resolution order:
    1. $QORTEX_DATA_DIR / dataset_name
    2. platformdirs.user_cache_dir('qortex') / 'datasets' / dataset_name
    3. ~/.cache/qortex/datasets / dataset_name
    """
    env_root = os.environ.get("QORTEX_DATA_DIR")
    if env_root:
        root = Path(env_root) / dataset_name
    else:
        try:
            import platformdirs  # type: ignore[import]
            root = Path(platformdirs.user_cache_dir("qortex")) / "datasets" / dataset_name
        except ImportError:
            root = Path.home() / ".cache" / "qortex" / "datasets" / dataset_name

    root.mkdir(parents=True, exist_ok=True)
    return root


def mne_data_dir() -> Path:
    """Return the MNE data directory (respects MNE_DATA env var)."""
    env = os.environ.get("MNE_DATA")
    if env:
        return Path(env)
    try:
        import mne  # type: ignore[import]
        return Path(mne.get_config("MNE_DATA", default=str(Path.home() / "mne_data")))
    except (ImportError, Exception):
        return Path.home() / "mne_data"
