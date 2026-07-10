from __future__ import annotations

import sys
import types
from pathlib import Path


def test_msd_brain_monai_loader_receives_public_seed(monkeypatch, tmp_path):
    from qortex.datasets import msd_brain

    captured: dict[str, object] = {}

    class FakeDecathlonDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._items = [
                {
                    "image": str(tmp_path / "imagesTr" / "case_000.nii.gz"),
                    "label": str(tmp_path / "labelsTr" / "case_000.nii.gz"),
                }
            ]

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int) -> dict[str, str]:
            return self._items[index]

    monai_mod = types.ModuleType("monai")
    monai_data_mod = types.ModuleType("monai.data")
    monai_data_mod.DecathlonDataset = FakeDecathlonDataset

    monkeypatch.setitem(sys.modules, "monai", monai_mod)
    monkeypatch.setitem(sys.modules, "monai.data", monai_data_mod)

    bundle = msd_brain.load_data(
        local_root=tmp_path,
        split="train",
        max_cases=1,
        download=False,
        seed=123,
    )

    assert captured["root_dir"] == str(tmp_path)
    assert captured["task"] == "Task01_BrainTumour"
    assert captured["section"] == "training"
    assert captured["download"] is False
    assert captured["seed"] == 123
    assert bundle.n_cases == 1
    assert bundle.image_paths == [[Path(tmp_path / "imagesTr" / "case_000.nii.gz")]]
    assert bundle.mask_paths == [Path(tmp_path / "labelsTr" / "case_000.nii.gz")]
