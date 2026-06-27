"""AI runtime integration — native PyTorch, MONAI, and TorchEEG connectors.

Provides zero-boilerplate mapping from BIDS entities directly into
framework-native dataset objects.  No intermediate Parquet conversion
step required for MRI-centric workflows.

Usage::

    from qortex.runtime import BIDSImageDataset, BIDSSignalDataset

    # Volumetric MRI → PyTorch map-style Dataset
    ds = BIDSImageDataset(
        bids_root=Path("~/.cache/qortex/ds000001/1.0.0"),
        suffix="T1w",
        label_column="diagnosis",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2)

    # Electrophysiology epochs → PyTorch Dataset
    eeg_ds = BIDSSignalDataset(
        bids_root=Path("~/.cache/qortex/ds004130/1.0.2"),
        modality="eeg",
        label_column="trial_type",
        epoch_duration_s=2.0,
    )
"""

from qortex.runtime.loader import (
    BIDSImageDataset,
    BIDSSignalDataset,
    MONAIDictBuilder,
)
from qortex.runtime.epochs import (
    BIDSEpochDataset,
    TorchEEGBridge,
)

__all__ = [
    "BIDSImageDataset",
    "BIDSSignalDataset",
    "MONAIDictBuilder",
    "BIDSEpochDataset",
    "TorchEEGBridge",
]
