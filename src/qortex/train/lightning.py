"""PyTorch Lightning DataModule adapter for Qortex Parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class QortexDataModule:
    """LightningDataModule wrapping QortexTorchDataset splits.

    Usage::

        dm = QortexDataModule(data_dir=Path("my_lake/output"))
        dm.setup()
        trainer.fit(model, datamodule=dm)
    """

    framework = "lightning"

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self._data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._train: Any = None
        self._val: Any = None
        self._test: Any = None

    def setup(self, stage: str | None = None) -> None:
        from qortex.train.torch import QortexTorchDataset
        if stage in (None, "fit"):
            self._train = QortexTorchDataset(self._data_dir, split="train")
            self._val = QortexTorchDataset(self._data_dir, split="val")
        if stage in (None, "test"):
            self._test = QortexTorchDataset(self._data_dir, split="test")

    def train_dataloader(self) -> Any:
        from torch.utils.data import DataLoader
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Any:
        from torch.utils.data import DataLoader
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Any:
        from torch.utils.data import DataLoader
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def from_dir(self, data_dir: Path, split: str | None = None) -> "QortexDataModule":
        return QortexDataModule(
            data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
