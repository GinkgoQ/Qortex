from __future__ import annotations

import json
import zipfile

import pytest

from qortex.neuroai.models.monai import (
    MONAIBundleAdapter,
    _load_json,
    _safe_extract_zip,
)
from qortex.neuroai.spec import ModelSpec, RuntimeSpec


def test_load_json_rejects_malformed_config(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        _load_json(path)


def test_safe_extract_zip_blocks_path_traversal(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        _safe_extract_zip(archive, tmp_path / "extract")


def test_monai_predict_does_not_fallback_to_full_volume_after_sliding_window_error(monkeypatch):
    import torch

    class _FailingInferers:
        @staticmethod
        def sliding_window_inference(*args, **kwargs):
            raise RuntimeError("sliding window failed")

    fake_monai = type("FakeMonai", (), {"inferers": _FailingInferers})
    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id="local"))
    adapter._model = torch.nn.Identity()
    adapter._device = "cpu"
    adapter._inference_settings = {
        "roi_size": (2, 2, 2),
        "sw_batch_size": 1,
        "overlap": 0.25,
        "argmax_axis": 1,
        "label_map": {},
    }
    monkeypatch.setattr("qortex.neuroai.models.monai._require_monai", lambda: fake_monai)

    with pytest.raises(RuntimeError, match="sliding window failed"):
        adapter.predict(torch.zeros((1, 1, 2, 2, 2)))


def test_monai_load_rejects_state_dict_mismatch(tmp_path, monkeypatch):
    import torch

    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "models").mkdir()
    (bundle / "configs" / "metadata.json").write_text("{}", encoding="utf-8")
    (bundle / "configs" / "inference.json").write_text(
        json.dumps({"network_def": {"spatial_dims": 3, "in_channels": 1, "out_channels": 1}}),
        encoding="utf-8",
    )
    torch.save({"unexpected.weight": torch.ones(1)}, bundle / "models" / "model.pt")

    class _FakeParser(dict):
        def read_config(self, path):
            self["network_def"] = torch.nn.Conv3d(1, 1, kernel_size=1)

        def get_parsed_content(self, key):
            return self[key]

    fake_monai = type("FakeMonai", (), {"bundle": type("B", (), {"ConfigParser": _FakeParser})})
    monkeypatch.setattr("qortex.neuroai.models.monai._require_monai", lambda: fake_monai)
    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id=str(bundle)))

    with pytest.raises(RuntimeError, match="state_dict mismatch"):
        adapter.load(RuntimeSpec(device="cpu"))
