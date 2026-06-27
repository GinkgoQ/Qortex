"""ML framework manifest exporters.

Exports locally-downloaded BIDS datasets into native JSON formats directly
consumable by MONAI and TorchIO without any user-written boilerplate.

Usage::

    from qortex.export import MONAIExporter, TorchIOExporter

    # MONAI PersistentDataset / CacheDataset JSON datalist
    exp = MONAIExporter(bids_root=Path("~/.cache/qortex/datasets/ds000001/1.0.0"))
    manifest_path = exp.export(
        output_dir=Path("./monai_data"),
        datatype="anat",
        suffix="T1w",
        label_source="participants",
        label_column="diagnosis",
    )

    # TorchIO SubjectsDataset JSON
    exp = TorchIOExporter(bids_root=Path("~/.cache/qortex/datasets/ds000001/1.0.0"))
    manifest_path = exp.export(
        output_dir=Path("./torchio_data"),
        modalities={"T1w": "ScalarImage", "T2w": "ScalarImage"},
        label_column="diagnosis",
    )
"""

from qortex.export.monai import MONAIExporter, MONAIDataset
from qortex.export.torchio import TorchIOExporter, TorchIOSubject

__all__ = [
    "MONAIExporter",
    "MONAIDataset",
    "TorchIOExporter",
    "TorchIOSubject",
]
